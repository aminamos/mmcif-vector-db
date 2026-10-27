"""Optional FAISS ANN index over chain vectors.

Engine: FAISS (``faiss-cpu``), available only via the optional ``faiss``
extra (``uv sync --extra faiss`` / ``pip install 'mmcif-vector-db[faiss]'``).
INSTALL ROUTE VERIFIED: repo route worked -- ``faiss-cpu==1.15.1``
resolved and installed cleanly on this repo's Python 3.14 / Windows
environment (pinned in ``uv.lock``); no WSL fallback was needed. The
module itself is platform-agnostic (no Windows/WSL-specific code).

FAISS is imported lazily inside functions/methods only: importing this
module (or ``mmcif_vec.db`` / ``mmcif_vec.ann``) never imports ``faiss``.
Using a FAISS index without the extra installed raises a clear
``RuntimeError`` naming the install route.

The index is a sidecar directory next to the SQLite DB:

    <out_dir>/faiss_meta.json  {"engine": "faiss", "faiss_version",
                                "factory", "index_factory", "dim", "n",
                                "nlist", "pq_m", "trained", ...}
    <out_dir>/index.faiss      faiss.write_index output
    <out_dir>/uids.npy         int64-id -> uid mapping (row i = faiss id i)

Vectors are L2-normalised at build/query time, so the index uses
``METRIC_INNER_PRODUCT`` and inner product == cosine similarity.

Exactness: this index only *generates candidates*. ``mmcif_vec.db.search``
reranks candidates with the exact block cosines, so FAISS scores here are
approximate candidate scores, never reported result scores.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# Mirrors ann.VECTOR_COLUMNS without importing ann/db.
VECTOR_COLUMNS = {
    "combined": ("combined_vec", 768),
    "seq": ("seq_vec", 256),
    "struct": ("struct_vec", 256),
    "text": ("text_vec", 256),
}

META_NAME = "faiss_meta.json"
INDEX_NAME = "index.faiss"
UIDS_NAME = "uids.npy"
CHUNK = 10_000
# Training sample cap (~262k vectors); IVF/PQ training on more gives
# negligible centroid quality gain for much more RAM/time.
DEFAULT_TRAIN_SAMPLE = 262_144

_FACTORIES = ("ivf_flat", "ivf_pq")


def _load_faiss():
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError(
            "FAISS ANN support requires the optional 'faiss' extra "
            "(faiss-cpu is not installed). Install it with: "
            "uv sync --extra faiss  (or: pip install "
            "'mmcif-vector-db[faiss]')"
        ) from e
    return faiss


def _normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32, copy=False)


def _read_chunks(db, col, dim, limit=None):
    """Yield (uids, normalised (m, dim) float32) in 10k keyset chunks."""
    last_uid = None
    remaining = limit
    while True:
        lim = CHUNK if remaining is None else min(CHUNK, remaining)
        if lim <= 0:
            break
        if last_uid is None:
            rows = db.con.execute(
                f"SELECT uid, {col} FROM chains ORDER BY uid LIMIT ?",
                (lim,)).fetchall()
        else:
            rows = db.con.execute(
                f"SELECT uid, {col} FROM chains WHERE uid > ? "
                f"ORDER BY uid LIMIT ?", (last_uid, lim)).fetchall()
        if not rows:
            break
        uids = [r[0] for r in rows]
        vecs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        if vecs.shape[1] != dim:
            raise ValueError(f"{col} vectors must have {dim} dimensions")
        yield uids, _normalize(vecs)
        last_uid = uids[-1]
        if remaining is not None:
            remaining -= len(rows)
        if len(rows) < lim:
            break


class FaissIndex:
    """FAISS-backed ANN candidate index for one DB vector column."""

    def __init__(self, out_dir, index, uids, meta):
        self.out_dir = Path(out_dir)
        self._index = index
        self._uids = [str(u) for u in uids]
        self.meta = dict(meta)
        self.dim = int(self.meta["dim"])
        self.n = int(self.meta["n"])
        # Query-time probe count (mutable so benchmarks/callers can tune
        # the recall/latency trade-off without rebuilding).
        self.nprobe = self.meta.get("nprobe")

    # ------------------------------------------------------------ build
    @classmethod
    def build(cls, db, out_dir, vector="combined", *, index_factory="ivf_flat",
              nlist=None, pq_m=96, train_sample=DEFAULT_TRAIN_SAMPLE,
              nprobe=None):
        """Build a FAISS index over ``vector`` BLOBs from a ``DB``.

        Reads chains in keyset-paginated 10k chunks (never loads the
        whole corpus at once) in two passes: a capped training sample,
        then one streaming ``add_with_ids`` pass.
        """
        faiss = _load_faiss()
        if index_factory not in _FACTORIES:
            raise ValueError(
                f"unknown index_factory {index_factory!r}; "
                f"expected one of {list(_FACTORIES)}")
        if vector not in VECTOR_COLUMNS:
            raise ValueError(
                f"unknown vector {vector!r}; expected one of {sorted(VECTOR_COLUMNS)}")
        col, dim = VECTOR_COLUMNS[vector]
        pq_m = int(pq_m)
        if index_factory == "ivf_pq" and dim % pq_m != 0:
            raise ValueError(
                f"pq_m={pq_m} must divide dim={dim} (for ivf_pq)")
        train_sample = int(train_sample)
        if train_sample <= 0:
            raise ValueError("train_sample must be positive")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        created_from = None
        try:
            row = db.con.execute("PRAGMA database_list").fetchone()
            created_from = row[2] if row is not None and len(row) > 2 else None
        except Exception:
            created_from = None

        n = int(db.con.execute("SELECT COUNT(*) FROM chains").fetchone()[0])

        if nlist is None:
            nlist = int(min(4096, max(1, round(np.sqrt(n))))) if n else 1
        # FAISS cannot train more IVF cells than vectors.
        nlist = int(max(1, min(int(nlist), n))) if n else 1

        if index_factory == "ivf_flat":
            factory = f"IVF{nlist},Flat"
            meta_pq_m = None
        else:
            factory = f"IVF{nlist},PQ{pq_m}"
            meta_pq_m = pq_m

        index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)

        t_build = time.perf_counter()
        train_seconds = 0.0
        trained = bool(index.is_trained)
        if n and not trained:
            t0 = time.perf_counter()
            sample_parts, got = [], 0
            for _uids, vecs in _read_chunks(db, col, dim, limit=train_sample):
                sample_parts.append(vecs)
                got += len(vecs)
                if got >= train_sample:
                    break
            train_vecs = (np.concatenate(sample_parts) if sample_parts
                          else np.empty((0, dim), dtype=np.float32))
            if len(train_vecs):
                index.train(train_vecs)
            trained = bool(index.is_trained)
            train_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        uids: list[str] = []
        next_id = 0
        for chunk_uids, vecs in _read_chunks(db, col, dim):
            ids = np.arange(next_id, next_id + len(chunk_uids), dtype=np.int64)
            index.add_with_ids(vecs, ids)
            uids.extend(chunk_uids)
            next_id += len(chunk_uids)
        add_seconds = time.perf_counter() - t0
        if next_id != n:
            # Rows changed between COUNT(*) and the add pass; trust the pass.
            n = next_id

        faiss.write_index(index, str(out_dir / INDEX_NAME))
        np.save(out_dir / UIDS_NAME,
                np.array(uids, dtype=str) if uids else np.empty((0,), dtype=str))

        if nprobe is None:
            nprobe = int(min(32, nlist))
        meta = {
            "engine": "faiss",
            "faiss_version": getattr(faiss, "__version__", None),
            "factory": factory,
            "index_factory": index_factory,
            "dim": dim,
            "n": n,
            "vector": vector,
            "column": col,
            "created_from": created_from,
            "nlist": nlist,
            "pq_m": meta_pq_m,
            "trained": trained,
            "train_sample": int(min(train_sample, n)) if n else 0,
            "nprobe": int(nprobe),
            "metric": "inner_product",
            "train_seconds": round(train_seconds, 3),
            "add_seconds": round(add_seconds, 3),
            "build_seconds": round(time.perf_counter() - t_build, 3),
        }
        idx = cls(out_dir, index, uids, meta)
        idx.save()
        return idx

    # ------------------------------------------------------------- save/load
    def save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta)
        meta["nprobe"] = self.nprobe
        (self.out_dir / META_NAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        self.meta = meta
        return self

    @classmethod
    def load(cls, out_dir):
        faiss = _load_faiss()
        out_dir = Path(out_dir)
        meta_path = out_dir / META_NAME
        if not meta_path.exists():
            raise FileNotFoundError(f"no FAISS index metadata at {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("engine") != "faiss":
            raise ValueError(
                f"unsupported FAISS engine in {meta_path}: {meta.get('engine')!r}")
        index_path = out_dir / INDEX_NAME
        if not index_path.exists():
            raise FileNotFoundError(f"no FAISS index file at {index_path}")
        index = faiss.read_index(str(index_path))
        uids_path = out_dir / UIDS_NAME
        uids = (np.load(uids_path).tolist() if uids_path.exists() else [])
        return cls(out_dir, index, uids, meta)

    # ------------------------------------------------------------ query
    def query(self, qvec, pool_k, *, nprobe=None):
        """Return [(uid, approx_cosine_score), ...] best-first, up to pool_k."""
        if pool_k <= 0 or self.n == 0:
            return []
        q = _normalize(np.asarray(qvec, dtype=np.float32).reshape(-1))
        if q.shape[1] != self.dim:
            raise ValueError(f"query vector must have {self.dim} dimensions")
        probes = self.nprobe if nprobe is None else nprobe
        if probes is not None:
            try:
                self._index.nprobe = int(max(1, min(int(probes),
                                                    int(self.meta["nlist"]))))
            except (AttributeError, TypeError, ValueError):
                pass
        scores, ids = self._index.search(q, int(pool_k))
        out = []
        for score, idx_id in zip(scores[0], ids[0]):
            if idx_id < 0 or idx_id >= len(self._uids):
                continue
            out.append((self._uids[int(idx_id)], float(score)))
        return out
