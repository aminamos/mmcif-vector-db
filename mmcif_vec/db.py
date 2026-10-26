"""SQLite vector store: per-chain seq/struct/text/combined vectors + brute-force cosine.

MVP scale (thousands of chains). Swap the scan for sqlite-vec/LanceDB later
behind this same API.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
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

class DB:
    def __init__(self, path="mmcif.db"):
        self.con = sqlite3.connect(path); self.con.row_factory = sqlite3.Row
        self.con.execute(SCHEMA); self.con.commit()
    def ingest(self, file) -> int:
        n = 0
        for r in parse_cif(file):
            sv, tv = seq_vector(r.sequence), text_vector(r.text)
            stv = struct_vector(r.ca_coords)
            self.con.execute(
              "INSERT OR REPLACE INTO chains VALUES (?,?,?,?,?,?,?,?,?,NULL,?,?,?,?)",
              (r.uid, r.entry_id, r.chain_id, r.source, r.title, r.method, r.sequence,
               len(r.sequence), r.path, _blob(sv), _blob(stv), _blob(tv),
               _blob(combined_vector(sv, stv, tv))))
            n += 1
        self.con.commit(); return n
    def rows(self):
        return self.con.execute("SELECT * FROM chains").fetchall()
    def stats(self):
        r = self.rows()
        return {"chains": len(r), "entries": len({x['entry_id'] for x in r}),
                "sources": {s: sum(1 for x in r if x['source']==s) for s in {x['source'] for x in r}},
                "clusters": len({x['cluster_id'] for x in r if x['cluster_id'] is not None})}

def cosine(a, b): 
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0

def search(db: DB, q_seq=None, q_struct=None, q_text=None, k=5, rerank=True):
    """Retrieve by combined vector, then rerank by separate seq/struct/text scores."""
    rows = db.rows()
    if not rows: return []
    # build query combined in same block layout
    sv = seq_vector(q_seq) if q_seq else np.zeros(256, dtype=np.float32)
    stv = q_struct if q_struct is not None else np.zeros(256, dtype=np.float32)
    tv = text_vector(q_text) if q_text else np.zeros(256, dtype=np.float32)
    if q_seq is None and q_struct is None:  # text-only: pure text block search
        cand = [(cosine(tv, _vec(r["text_vec"])), r) for r in rows]
    elif q_text is None and q_seq:
        from .features import combined_vector
        q = combined_vector(sv, stv, tv) if q_struct is not None else sv
        key = "combined_vec" if q_struct is not None else "seq_vec"
        cand = [(cosine(q, _vec(r[key])), r) for r in rows]
    else:
        from .features import combined_vector
        q = combined_vector(sv, stv, tv)
        cand = [(cosine(q, _vec(r["combined_vec"])), r) for r in rows]
    cand.sort(key=lambda t: -t[0])
    pool = cand[:max(k*4, k)] if rerank else cand[:k]
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
