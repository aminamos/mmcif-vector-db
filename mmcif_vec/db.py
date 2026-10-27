"""SQLite vector store: per-chain seq/struct/text/combined vectors.

Vectors live twice, deliberately: as float32 BLOBs on ``chains`` (block
scores, clustering and stats read them byte-identically to before) and in
sqlite-vec ``vec0`` virtual tables (``vec_combined``/``vec_seq``/
``vec_struct``/``vec_text``, each keyed by ``uid``) that generate search
candidates via KNN. Search reranks those candidates with the exact block
cosines the old brute-force scan used, so results are unchanged.

Transactions: the connection runs in autocommit mode; ``DB.transaction()``
is the explicit unit of work (BEGIN IMMEDIATE / COMMIT / ROLLBACK).
``upsert_chain`` joins the ambient transaction when called inside one and
otherwise wraps itself in a single-upsert transaction, so bulk loaders
should wrap their loop in ``with db.transaction():`` — committing per
upsert costs an fsync each (~230-300 vec/s measured), while one commit
per batch/file is orders of magnitude faster. ``DB.ingest`` already wraps
a whole file in one transaction.

Scale note (measured, results_store/benchmark_store.md): vec0 KNN here is
an exact linear scan — search p50 ~6 ms at 2,202 vectors, ~19 ms at 10k,
~179 ms at 100k, ~895 ms (~0.9 s) at 500k on the benchmark machine. It is
chosen for exactness and single-file simplicity; an ANN index is the
documented next step for interactive latency at >500k scale.
"""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import sqlite_vec
from .parse import parse_cif
from .features import seq_vector, struct_vector, text_vector, combined_vector

def _blob(v): return np.asarray(v, dtype=np.float32).tobytes()
def _vec(b): return np.frombuffer(b, dtype=np.float32)

SCHEMA = """
CREATE TABLE IF NOT EXISTS chains(
 uid TEXT PRIMARY KEY, entry_id TEXT, chain_id TEXT, source TEXT, title TEXT,
 method TEXT, sequence TEXT, n_res INTEGER, path TEXT, cluster_id INTEGER,
 seq_vec BLOB, struct_vec BLOB, text_vec BLOB, combined_vec BLOB);
"""

# vec0 tables mirror the BLOB columns: uid is the primary key, the single
# embedding column carries the fixed dimension. Verified against
# sqlite-vec 0.1.9: `INSERT OR REPLACE` is rejected by vec0 (UNIQUE
# constraint on the primary key), so replacement is DELETE + INSERT
# inside the same transaction (see upsert_chain).
VEC_TABLES = {
    "vec_seq": ("seq_vec", 256),
    "vec_struct": ("struct_vec", 256),
    "vec_text": ("text_vec", 256),
    "vec_combined": ("combined_vec", 768),
}
VEC_SCHEMA = [
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0(uid TEXT PRIMARY KEY, embedding float[{dim}])"
    for name, (_, dim) in VEC_TABLES.items()
]

class DB:
    def __init__(self, path="mmcif.db"):
        self._txn_depth = 0
        # Autocommit mode: DB.transaction() owns BEGIN/COMMIT/ROLLBACK.
        self.con = sqlite3.connect(path, isolation_level=None)
        self.con.row_factory = sqlite3.Row
        try:
            self.con.enable_load_extension(True)
            sqlite_vec.load(self.con)
        except Exception as e:
            self.con.close()
            raise RuntimeError(
                f"failed to load the sqlite-vec extension (required for the vector store): {e}"
            ) from e
        existing = {r[0] for r in self.con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        vec_missing = any(name not in existing for name in VEC_TABLES)
        self.con.execute(SCHEMA)
        for stmt in VEC_SCHEMA: self.con.execute(stmt)

        if "chains" in existing and (vec_missing or self._vec_count() != self._chain_count()):
            self._backfill_vec()
    def _chain_count(self):
        return self.con.execute("SELECT COUNT(*) FROM chains").fetchone()[0]
    def _vec_count(self):
        return self.con.execute("SELECT COUNT(*) FROM vec_combined").fetchone()[0]
    @contextmanager
    def transaction(self):
        """Explicit unit of work: BEGIN IMMEDIATE ... COMMIT/ROLLBACK.

        Nested calls (and calls made while the connection is already in a
        transaction) participate in the ambient transaction without
        beginning, committing, or rolling back on their own — an exception
        propagates to the outermost ``transaction()`` block, which rolls
        back everything. There are no savepoints.
        """
        if self._txn_depth or self.con.in_transaction:
            self._txn_depth += 1
            try:
                yield
            finally:
                self._txn_depth -= 1
            return
        self.con.execute("BEGIN IMMEDIATE")
        self._txn_depth = 1
        try:
            yield
        except BaseException:
            self.con.execute("ROLLBACK")
            raise
        else:
            self.con.execute("COMMIT")
        finally:
            self._txn_depth = 0
    def _backfill_vec(self):
        """Populate the vec0 tables from the chains BLOBs.

        Chunked by keyset-sized batches of 1,000 chains, each batch in its
        own transaction (never one commit per row).
        """
        while rows := self.con.execute(
                "SELECT uid, seq_vec, struct_vec, text_vec, combined_vec FROM chains "
                "WHERE uid NOT IN (SELECT uid FROM vec_combined) LIMIT 1000").fetchall():
            with self.transaction():
                for r in rows:
                    for name, (col, _) in VEC_TABLES.items():
                        self.con.execute(
                            f"INSERT INTO {name}(uid, embedding) VALUES (?,?)",
                            (r["uid"], r[col]))
    def upsert_chain(self, uid, entry_id, chain_id, source, title, method,
                     sequence, n_res, path, seq_vec, struct_vec, text_vec, combined_vec):
        """Write one chain's chains row and its four vec0 rows atomically.

        Re-upserting a uid replaces its vectors (vec0 rejects
        INSERT OR REPLACE, so each vec row is DELETE + INSERTed in the
        same transaction as the chains row). Participates in the ambient
        ``DB.transaction()`` when inside one (no commit of its own);
        called standalone, it wraps itself in a single-upsert transaction.
        """
        vecs = {"vec_seq": seq_vec, "vec_struct": struct_vec,
                "vec_text": text_vec, "vec_combined": combined_vec}
        for name, v in vecs.items():
            dim = VEC_TABLES[name][1]
            if np.asarray(v).size != dim:
                raise ValueError(f"{name} vector must have {dim} dimensions")
        with self.transaction():
            self.con.execute(
              "INSERT OR REPLACE INTO chains VALUES (?,?,?,?,?,?,?,?,?,NULL,?,?,?,?)",
              (uid, entry_id, chain_id, source, title, method, sequence,
               n_res, path, _blob(seq_vec), _blob(struct_vec), _blob(text_vec),
               _blob(combined_vec)))
            for name, v in vecs.items():
                self.con.execute(f"DELETE FROM {name} WHERE uid=?", (uid,))
                self.con.execute(
                    f"INSERT INTO {name}(uid, embedding) VALUES (?,?)", (uid, _blob(v)))
    def ingest(self, file) -> int:
        """Ingest one file: all its chains in a single transaction."""
        prepared = []
        for r in parse_cif(file):
            sv, tv = seq_vector(r.sequence), text_vector(r.text)
            stv = struct_vector(r.ca_coords)
            prepared.append((r.uid, r.entry_id, r.chain_id, r.source, r.title,
                             r.method, r.sequence, len(r.sequence), r.path,
                             sv, stv, tv, combined_vector(sv, stv, tv)))
        with self.transaction():
            for args in prepared: self.upsert_chain(*args)
        return len(prepared)
    def rows(self):
        return self.con.execute("SELECT * FROM chains").fetchall()
    def stats(self):
        chains, entries, clusters = self.con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT entry_id), COUNT(DISTINCT cluster_id) "
            "FROM chains").fetchone()
        sources = dict(self.con.execute(
            "SELECT source, COUNT(*) FROM chains GROUP BY source").fetchall())
        return {"chains": chains, "entries": entries,
                "sources": sources, "clusters": clusters}

def cosine(a, b): 
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0

def _rows_by_uid(db: DB, uids):
    if not uids: return []
    by_uid = {r["uid"]: r for r in db.con.execute(
        f"SELECT * FROM chains WHERE uid IN ({','.join('?' * len(uids))})", uids)}
    return [by_uid[u] for u in uids if u in by_uid]

def _knn_rows(db: DB, table: str, key: str, q, pool_size: int):
    """Candidate chains rows via vec0 KNN on `table`, best first."""
    if pool_size <= 0: return []
    if not np.linalg.norm(q):
        # A zero query has cosine 0 against everything; KNN distances
        # would order by row norm instead. Keep the brute-force pool.
        return db.rows()
    hits = db.con.execute(
        f"SELECT uid FROM {table} WHERE embedding MATCH ? AND k = ?",
        (_blob(q), pool_size)).fetchall()
    if not hits: return []
    return _rows_by_uid(db, [h[0] for h in hits])

def search(db: DB, q_seq=None, q_struct=None, q_text=None, k=5, rerank=True, ann=None):
    """Retrieve by combined vector, then rerank by separate seq/struct/text scores.

    Optional ``ann`` (an ``mmcif_vec.ann.ANNIndex`` over combined vectors):
    only used for combined (seq+struct) queries, where candidate uids come
    from ``ann.query`` instead of the exact vec0 scan. All other modes
    ignore it. Fetch/rerank/result format are identical either way.
    """
    if not db._chain_count(): return []
    # build query combined in same block layout
    sv = seq_vector(q_seq) if q_seq else np.zeros(256, dtype=np.float32)
    stv = q_struct if q_struct is not None else np.zeros(256, dtype=np.float32)
    tv = text_vector(q_text) if q_text else np.zeros(256, dtype=np.float32)
    if q_seq is None and q_struct is None:  # text-only: pure text block search
        table, key, q = "vec_text", "text_vec", tv
    elif q_text is None and q_seq:
        if q_struct is not None:
            table, key, q = "vec_combined", "combined_vec", combined_vector(sv, stv, tv)
        else:
            table, key, q = "vec_seq", "seq_vec", sv
    else:
        table, key, q = "vec_combined", "combined_vec", combined_vector(sv, stv, tv)
    pool_size = max(k*4, k) if rerank else k
    if ann is not None and table == "vec_combined" and q_struct is not None and np.linalg.norm(q):
        cand_rows = _rows_by_uid(db, [uid for uid, _ in ann.query(q, pool_size)])
    else:
        cand_rows = _knn_rows(db, table, key, q, pool_size)
    cand = [(cosine(q, _vec(r[key])), r) for r in cand_rows]
    cand.sort(key=lambda t: -t[0])
    pool = cand[:pool_size] if rerank else cand[:k]
    out = []
    for score, r in pool:
        s_seq = cosine(sv, _vec(r["seq_vec"])) if q_seq else None
        s_str = cosine(stv, _vec(r["struct_vec"])) if q_struct is not None else None
        s_txt = cosine(tv, _vec(r["text_vec"])) if q_text else None
        parts = [x for x in (s_seq, s_str, s_txt) if x is not None]
        final = sum(parts)/len(parts) if (rerank and parts) else score
        out.append({"uid": r["uid"], "entry_id": r["entry_id"], "chain_id": r["chain_id"],
          "source": r["source"], "title": r["title"], "n_res": r["n_res"],
          "cluster_id": r["cluster_id"], "sequence": r["sequence"],
          "score": round(final, 4), "seq_score": None if s_seq is None else round(s_seq,4),
          "struct_score": None if s_str is None else round(s_str,4),
          "text_score": None if s_txt is None else round(s_txt,4)})
    out.sort(key=lambda d: -d["score"])
    return out[:k]
