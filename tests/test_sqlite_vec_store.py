import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from mmcif_vec.cluster import cluster
from mmcif_vec.db import DB, SCHEMA, _blob, _vec, cosine, search
from mmcif_vec.features import combined_vector, seq_vector, struct_vector, text_vector
from mmcif_vec.parse import parse_cif

EX = Path(__file__).parents[1] / "examples"
VEC_TABLES = ("vec_seq", "vec_struct", "vec_text", "vec_combined")


def _ingest_examples(db):
    for f in sorted(EX.glob("*.cif")):
        db.ingest(f)
    return db


def _brute_force(db, q_seq=None, q_struct=None, q_text=None, k=5, rerank=True):
    """The pre-sqlite-vec scan, verbatim in behaviour, as reference."""
    rows = db.rows()
    if not rows:
        return []
    sv = seq_vector(q_seq) if q_seq else np.zeros(256, dtype=np.float32)
    stv = q_struct if q_struct is not None else np.zeros(256, dtype=np.float32)
    tv = text_vector(q_text) if q_text else np.zeros(256, dtype=np.float32)
    if q_seq is None and q_struct is None:
        cand = [(cosine(tv, _vec(r["text_vec"])), r) for r in rows]
    elif q_text is None and q_seq:
        q = combined_vector(sv, stv, tv) if q_struct is not None else sv
        key = "combined_vec" if q_struct is not None else "seq_vec"
        cand = [(cosine(q, _vec(r[key])), r) for r in rows]
    else:
        q = combined_vector(sv, stv, tv)
        cand = [(cosine(q, _vec(r["combined_vec"])), r) for r in rows]
    cand.sort(key=lambda t: -t[0])
    pool = cand[:max(k * 4, k)] if rerank else cand[:k]
    out = []
    for score, r in pool:
        s_seq = cosine(sv, _vec(r["seq_vec"])) if q_seq else None
        s_str = cosine(stv, _vec(r["struct_vec"])) if q_struct is not None else None
        s_txt = cosine(tv, _vec(r["text_vec"])) if q_text else None
        parts = [x for x in (s_seq, s_str, s_txt) if x is not None]
        final = sum(parts) / len(parts) if (rerank and parts) else score
        out.append({"uid": r["uid"], "score": round(final, 4),
                    "seq_score": None if s_seq is None else round(s_seq, 4),
                    "struct_score": None if s_str is None else round(s_str, 4),
                    "text_score": None if s_txt is None else round(s_txt, 4)})
    out.sort(key=lambda d: -d["score"])
    return out[:k]


def _assert_same_topk(new, ref, ref_all):
    """Top-k equivalence allowing tie-order/tie-membership differences.

    Exact when reference scores are distinct; when scores tie (within
    1e-6, e.g. chains of one entry sharing a text vector), any tied uid
    may occupy the tied slot, so validate each returned uid's score
    against the full brute-force ranking instead of uid identity.
    """
    assert len(new) == len(ref)
    assert [h["score"] for h in new] == [h["score"] for h in ref]
    by_uid = {h["uid"]: h for h in ref_all}
    assert all(by_uid[h["uid"]]["score"] == h["score"] for h in new)
    scores = [h["score"] for h in ref]
    new_uids = [h["uid"] for h in new]
    ref_uids = [h["uid"] for h in ref]
    if all(abs(a - b) > 1e-6 for a, b in zip(scores, scores[1:])):
        # No in-list ties: only a boundary tie (a uid outside ref's top-k
        # scoring equal to the last in) may substitute for the last slot.
        assert new_uids[:-1] == ref_uids[:-1]
        assert new_uids[-1] == ref_uids[-1] or (
            by_uid[new_uids[-1]]["score"] == scores[-1])
    else:
        assert set(new_uids) == set(ref_uids) or all(
            by_uid[u]["score"] in scores for u in new_uids)


def test_search_matches_brute_force_all_modes(tmp_path):
    db = _ingest_examples(DB(str(tmp_path / "t.db")))
    rec = parse_cif(EX / "2a3s.cif")[0]
    queries = [
        # combined (seq+struct, similar-style), seq-only, text-only
        dict(q_seq=rec.sequence, q_struct=struct_vector(rec.ca_coords)),
        dict(q_seq=rec.sequence),
        dict(q_text="myocyte nuclear factor"),
        dict(q_text="foxk1 transcription factor forkhead"),
    ]
    n = db.stats()["chains"]
    for q in queries:
        new = search(db, k=3, **q)
        ref = _brute_force(db, k=3, **q)
        ref_all = _brute_force(db, k=n, **q)
        assert new, q
        _assert_same_topk(new, ref, ref_all)
        by_uid = {h["uid"]: h for h in ref_all}
        for h_new in new:
            h_ref = by_uid[h_new["uid"]]
            assert h_new["seq_score"] == h_ref["seq_score"]
            assert h_new["struct_score"] == h_ref["struct_score"]
            assert h_new["text_score"] == h_ref["text_score"]


def _build_old_format_db(path):
    """A pre-sqlite-vec DB file: chains table with BLOBs, no vec tables."""
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    n = 0
    for f in sorted(EX.glob("*.cif")):
        for r in parse_cif(f):
            sv, tv = seq_vector(r.sequence), text_vector(r.text)
            stv = struct_vector(r.ca_coords)
            con.execute(
                "INSERT OR REPLACE INTO chains VALUES (?,?,?,?,?,?,?,?,?,NULL,?,?,?,?)",
                (r.uid, r.entry_id, r.chain_id, r.source, r.title, r.method,
                 r.sequence, len(r.sequence), r.path, _blob(sv), _blob(stv),
                 _blob(tv), _blob(combined_vector(sv, stv, tv))))
            n += 1
    con.commit()
    con.close()
    return n


def test_migration_backfills_vec_tables_and_search_and_cluster(tmp_path):
    path = str(tmp_path / "old.db")
    n = _build_old_format_db(path)
    # sanity: the hand-built file really has no vec tables
    con = sqlite3.connect(path)
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert not any(t in names for t in VEC_TABLES)

    db = DB(path)  # opening migrates + backfills
    names = {r[0] for r in db.con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert all(t in names for t in VEC_TABLES)
    for t in VEC_TABLES:
        assert db.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == n
    assert db.stats()["chains"] == n

    hits = search(db, q_text="myocyte nuclear factor", k=3)
    assert hits and "2A3S" in hits[0]["uid"]
    _assert_same_topk(hits, _brute_force(db, q_text="myocyte nuclear factor", k=3),
                      _brute_force(db, q_text="myocyte nuclear factor", k=n))

    groups = cluster(db, k=2)
    assert sum(len(v) for v in groups.values()) == n


def test_upsert_chain_replaces_vectors(tmp_path):
    db = DB(str(tmp_path / "t.db"))
    q_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"
    sv = seq_vector(q_seq)
    other = seq_vector("WWWWYYYYMMMMKKKKRRRRHHHHCCCCWW")
    zero = np.zeros(256, dtype=np.float32)

    def chain(uid, vec):
        db.upsert_chain(uid, uid.split(":")[0], "A", "PDB", "t", "X-RAY",
                        q_seq, len(q_seq), "x.cif", vec, zero, zero,
                        combined_vector(vec, zero, zero))

    chain("AAA:A", sv)
    chain("BBB:A", other)
    assert search(db, q_seq=q_seq, k=1)[0]["uid"] == "AAA:A"

    # Re-upsert the same uid with a mutated vector: row count stays put
    # and KNN now reflects the new vector, not the old one.
    q2_seq = "GGGGSSSSAAAACCCCDDDDEEEEFFFFGG"
    mutated = seq_vector(q2_seq)
    chain("AAA:A", mutated)
    assert db.stats()["chains"] == 2
    assert search(db, q_seq=q2_seq, k=1)[0]["uid"] == "AAA:A"
    hits = db.con.execute(
        "SELECT uid, distance FROM vec_seq WHERE embedding MATCH ? AND k = 2",
        (_blob(mutated),)).fetchall()
    assert hits[0][0] == "AAA:A" and hits[0][1] < 1e-6
    assert {h[0] for h in hits} == {"AAA:A", "BBB:A"}


def test_cluster_on_new_db(tmp_path):
    db = _ingest_examples(DB(str(tmp_path / "t.db")))
    groups = cluster(db, k=2)
    assert sum(len(v) for v in groups.values()) == db.stats()["chains"]
    assert db.stats()["clusters"] == 2

def test_transaction_rolls_back_upserts(tmp_path):
    db = DB(str(tmp_path / "t.db"))
    vec = seq_vector("ACDEFGHIKLMNPQRSTVWYACDEFGHIKL")
    zero = np.zeros(256, dtype=np.float32)
    args = ("AAA:A", "AAA", "A", "PDB", "t", "X-RAY", "ACDE", 4, "x.cif",
            vec, zero, zero, combined_vector(vec, zero, zero))
    with pytest.raises(RuntimeError, match="boom"):
        with db.transaction():
            db.upsert_chain(*args)
            # visible inside the transaction, gone after rollback
            assert db.stats()["chains"] == 1
            raise RuntimeError("boom")
    assert db.stats()["chains"] == 0
    for t in VEC_TABLES:
        assert db.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0
    # Standalone upsert still commits on its own.
    db.upsert_chain(*args)
    assert db.stats()["chains"] == 1
    for t in VEC_TABLES:
        assert db.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 1
    # A committed ambient transaction persists all its upserts at once.
    with db.transaction():
        db.upsert_chain("BBB:A", "BBB", "A", "PDB", "t", "X-RAY", "ACDE", 4,
                        "x.cif", vec, zero, zero, combined_vector(vec, zero, zero))
    assert db.stats()["chains"] == 2
