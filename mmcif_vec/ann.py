"""Optional ANN (approximate nearest neighbour) index over chain vectors.

Engine: LanceDB, available only via the optional ``ann`` extra
(``uv sync --extra ann`` / ``pip install 'mmcif-vector-db[ann]'``).
SUBSTITUTION NOTE: none -- LanceDB 0.40.0 resolved and installed cleanly
on this repo's Python 3.14 / Windows environment, so the settled LanceDB
design is implemented as specified (no pure-numpy IVF fallback needed).

LanceDB is imported lazily inside functions/methods only: importing this
module (or ``mmcif_vec.db``) never imports ``lancedb``. Using an ANN
index without the extra installed raises a clear ``RuntimeError``.

The index is a sidecar directory next to the SQLite DB:

    <out_dir>/ann_meta.json   {"engine", "lancedb_version", "dim", "n",
                               "vector", "created_from", "index_type",
                               "num_partitions", "num_sub_vectors",
                               "nprobes", ...}
    <out_dir>/lance/          LanceDB dataset (table ``chains_ann``)

Exactness: this index only *generates candidates*. ``mmcif_vec.db.search``
reranks candidates with the exact block cosines, so ANN scores here are
approximate candidate scores, never reported result scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# vector name -> (chains BLOB column, dimension); mirrors db.VEC_TABLES
# without importing db (keeps this module importable without sqlite-vec).
VECTOR_COLUMNS = {
    "combined": ("combined_vec", 768),
    "seq": ("seq_vec", 256),
    "struct": ("struct_vec", 256),
    "text": ("text_vec", 256),
}

META_NAME = "ann_meta.json"
TABLE_NAME = "chains_ann"
CHUNK = 10_000
# Below this many rows LanceDB index training is unreliable/pointless;
# search then runs LanceDB's exact flat scan, which is still fast.
MIN_INDEX_ROWS = 2048


def _load_lancedb():
    try:
        import lancedb
    except ImportError as e:
        raise RuntimeError(
            "ANN support requires the optional 'ann' extra (lancedb is not "
            "installed). Install it with: uv sync --extra ann  "
            "(or: pip install 'mmcif-vector-db[ann]')"
        ) from e
    return lancedb



class ANNIndex:
    """LanceDB-backed ANN candidate index for one DB vector column."""

    def __init__(self, out_dir, table, meta):
        self.out_dir = Path(out_dir)
        self._table = table
        self.meta = dict(meta)
        self.dim = int(self.meta["dim"])
        self.n = int(self.meta["n"])
        # Query-time probe count (mutable so benchmarks/callers can tune
        # the recall/latency trade-off without rebuilding).
        self.nprobes = self.meta.get("nprobes")

    # ------------------------------------------------------------ build
    @classmethod
    def build(cls, db, out_dir, vector="combined", *, num_partitions=None,
              num_sub_vectors=96, index_type="IVF_FLAT", nprobes=128,
              m=20, ef_construction=300):
        """Build an index over ``vector`` BLOBs from a ``mmcif_vec.db.DB``.

        Reads chains in keyset-paginated chunks (never loads the whole
        corpus at once) and streams each chunk into LanceDB.
        """
        lancedb = _load_lancedb()
        import pyarrow as pa

        if vector not in VECTOR_COLUMNS:
            raise ValueError(
                f"unknown vector {vector!r}; expected one of {sorted(VECTOR_COLUMNS)}")
        col, dim = VECTOR_COLUMNS[vector]
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        lance_dir = out_dir / "lance"

        created_from = None
        try:
            row = db.con.execute("PRAGMA database_list").fetchone()
            created_from = row[2] if row is not None and len(row) > 2 else None
        except Exception:
            created_from = None

        schema = pa.schema([
            pa.field("uid", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ])

        ldb = lancedb.connect(str(lance_dir))
        if TABLE_NAME in list(ldb.list_tables().tables):
            ldb.drop_table(TABLE_NAME)
        table = ldb.create_table(TABLE_NAME, schema=schema, mode="overwrite")

        n = 0
        last_uid = None
        while True:
            if last_uid is None:
                rows = db.con.execute(
                    f"SELECT uid, {col} FROM chains ORDER BY uid LIMIT ?",
                    (CHUNK,)).fetchall()
            else:
                rows = db.con.execute(
                    f"SELECT uid, {col} FROM chains WHERE uid > ? "
                    f"ORDER BY uid LIMIT ?", (last_uid, CHUNK)).fetchall()
            if not rows:
                break
            uids = [r[0] for r in rows]
            vecs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            if vecs.shape[1] != dim:
                raise ValueError(f"{col} vectors must have {dim} dimensions")
            flat = pa.array(vecs.reshape(-1), type=pa.float32())
            batch = pa.table({
                "uid": pa.array(uids, type=pa.string()),
                "vector": pa.FixedSizeListArray.from_arrays(flat, dim),
            }, schema=schema)
            table.add(batch)
            n += len(rows)
            last_uid = uids[-1]
            if len(rows) < CHUNK:
                break

        if num_partitions is None:
            # Tuned on the 500k benchmark (results_store/benchmark_ann.md):
            # IVF_FLAT with 256 partitions + nprobes=128 gives pool
            # recall@40 >= 0.98 at ~25x end-to-end speedup; IVF_PQ on the
            # same corpus collapsed to ~0.15 recall (see benchmark script).
            num_partitions = int(min(256, max(1, round(np.sqrt(n))))) if n else 1
        num_partitions = int(max(1, min(num_partitions, n))) if n else 1

        index_created = False
        effective_index_type = "flat"
        if n >= MIN_INDEX_ROWS:
            kwargs = {"metric": "cosine", "index_type": index_type,
                      "num_partitions": num_partitions}
            if index_type.startswith("IVF_HNSW"):
                kwargs.update({"m": int(m), "ef_construction": int(ef_construction)})
                if "PQ" in index_type or "SQ" in index_type:
                    kwargs["num_sub_vectors"] = int(num_sub_vectors)
            elif index_type != "IVF_FLAT":
                kwargs["num_sub_vectors"] = int(num_sub_vectors)
            table.create_index(**kwargs)
            index_created = True
            effective_index_type = index_type

        meta = {
            "engine": "lancedb",
            "lancedb_version": getattr(lancedb, "__version__", None),
            "dim": dim,
            "n": n,
            "vector": vector,
            "column": col,
            "created_from": created_from,
            "index_type": effective_index_type,
            "index_created": index_created,
            "num_partitions": num_partitions if index_created else None,
            "num_sub_vectors": (int(num_sub_vectors)
                                if index_created and effective_index_type != "IVF_FLAT"
                                and not (effective_index_type.startswith("IVF_HNSW")
                                         and effective_index_type.endswith("FLAT"))
                                else None),
            "nprobes": int(nprobes) if index_created else None,
            "metric": "cosine",
        }
        idx = cls(out_dir, table, meta)
        idx.save()
        return idx

    # ------------------------------------------------------------- save/load
    def save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta)
        meta["nprobes"] = self.nprobes
        (self.out_dir / META_NAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        self.meta = meta
        return self

    @classmethod
    def load(cls, out_dir):
        lancedb = _load_lancedb()
        out_dir = Path(out_dir)
        meta_path = out_dir / META_NAME
        if not meta_path.exists():
            raise FileNotFoundError(f"no ANN index metadata at {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("engine") != "lancedb":
            raise ValueError(f"unsupported ANN engine in {meta_path}: {meta.get('engine')!r}")
        ldb = lancedb.connect(str(out_dir / "lance"))
        table = ldb.open_table(TABLE_NAME)
        return cls(out_dir, table, meta)

    # ------------------------------------------------------------ query
    def query(self, qvec, pool_k, *, nprobes=None):
        """Return [(uid, approx_cosine_score), ...] best-first, up to pool_k."""
        if pool_k <= 0 or self.n == 0:
            return []
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query vector must have {self.dim} dimensions")
        builder = (self._table.search(q.tolist())
                   .metric("cosine").select(["uid", "_distance"]).limit(int(pool_k)))
        probes = self.nprobes if nprobes is None else nprobes
        if probes is not None and self.meta.get("index_created"):
            parts = self.meta.get("num_partitions")
            if parts:
                probes = min(int(probes), int(parts))
            builder = builder.nprobes(int(probes))
        rows = builder.to_list()
        # LanceDB cosine _distance == 1 - cosine_similarity.
        return [(r["uid"], float(1.0 - r["_distance"])) for r in rows]
