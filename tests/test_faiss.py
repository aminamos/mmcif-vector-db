import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from mmcif_vec.db import DB, search
from mmcif_vec.features import combined_vector, seq_vector, struct_vector, text_vector
from mmcif_vec.parse import parse_cif

ROOT = Path(__file__).parents[1]
SAMPLES = ROOT / "samples"


def _ingest_samples(db):
    for f in sorted(SAMPLES.glob("*.cif")):
        db.ingest(f)
    return db


@pytest.fixture(scope="module")
def faiss_setup(tmp_path_factory):
    pytest.importorskip("faiss")
    from mmcif_vec.faiss_ann import FaissIndex

    tmp = tmp_path_factory.mktemp("faiss")
    db = _ingest_samples(DB(str(tmp / "t.db")))
    idx = FaissIndex.build(db, tmp / "faiss")
    return db, idx, tmp / "faiss"


def test_faiss_query_pool_contains_self_and_top1_matches_exact(faiss_setup):
    db, idx, _ = faiss_setup
    rec = parse_cif(SAMPLES / "1ubq.cif")[0]
    q_struct = struct_vector(rec.ca_coords)
    q = combined_vector(seq_vector(rec.sequence), q_struct, text_vector(""))
    pool = idx.query(q, pool_k=20)
    assert pool, "FAISS query returned no candidates"
    assert rec.uid in [uid for uid, _ in pool]
    scores = [s for _, s in pool]
    assert scores == sorted(scores, reverse=True)

    exact = search(db, q_seq=rec.sequence, q_struct=q_struct, k=5)
    faiss_hits = search(db, q_seq=rec.sequence, q_struct=q_struct, k=5, ann=idx)
    assert exact and faiss_hits
    assert faiss_hits[0]["uid"] == exact[0]["uid"] == rec.uid


def test_search_candidate_pool_larger_than_4k_honoured_by_ann(faiss_setup):
    db, idx, _ = faiss_setup
    rec = parse_cif(SAMPLES / "1ubq.cif")[0]
    q_struct = struct_vector(rec.ca_coords)
    seen = []

    class RecordingAnn:
        def query(self, qvec, pool_k):
            seen.append(pool_k)
            return idx.query(qvec, pool_k)

    # k=5 -> default pool would be 20; candidate_pool=37 must reach ann.query.
    hits = search(db, q_seq=rec.sequence, q_struct=q_struct, k=5,
                  ann=RecordingAnn(), candidate_pool=37)
    assert seen == [37]
    assert len(hits) == 5
    assert hits[0]["uid"] == rec.uid
    # Default (candidate_pool=None) keeps the historical max(4k, k) pool.
    seen.clear()
    search(db, q_seq=rec.sequence, q_struct=q_struct, k=5, ann=RecordingAnn())
    assert seen == [20]

def test_faiss_meta_roundtrip(faiss_setup):
    db, idx, out_dir = faiss_setup
    from mmcif_vec.faiss_ann import FaissIndex

    meta_path = out_dir / "faiss_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["engine"] == "faiss"
    assert meta["faiss_version"]
    assert meta["factory"].startswith("IVF")
    assert meta["dim"] == 768
    assert meta["n"] == db.stats()["chains"]
    assert meta["nlist"] >= 1
    assert meta["pq_m"] is None  # ivf_flat default
    assert meta["trained"] is True
    assert (out_dir / "index.faiss").exists()

    loaded = FaissIndex.load(out_dir)
    assert loaded.meta["factory"] == idx.meta["factory"]
    assert loaded.n == idx.n
    rec = parse_cif(SAMPLES / "1ubq.cif")[0]
    q = combined_vector(seq_vector(rec.sequence),
                        struct_vector(rec.ca_coords), text_vector(""))
    assert rec.uid in [uid for uid, _ in loaded.query(q, pool_k=20)]


def test_faiss_pq_m_must_divide_dim(faiss_setup):
    pytest.importorskip("faiss")
    from mmcif_vec.faiss_ann import FaissIndex

    db, _, _ = faiss_setup
    with pytest.raises(ValueError, match="must divide"):
        FaissIndex.build(db, faiss_setup[2].parent / "bad",
                         index_factory="ivf_pq", pq_m=100)


def test_db_import_and_error_without_faiss():
    # Fresh interpreter: importing db/faiss_ann must not import faiss,
    # and using FAISS without the engine must raise a clear error.
    code = (
        "import sys\n"
        "sys.modules['faiss'] = None\n"  # make `import faiss` fail
        "import mmcif_vec.db\n"
        "import mmcif_vec.faiss_ann as fa\n"
        "assert sys.modules.get('faiss') is None\n"
        "try:\n"
        "    fa.FaissIndex.load('nonexistent-faiss-dir')\n"
        "except RuntimeError as e:\n"
        "    assert 'faiss' in str(e) and '--extra faiss' in str(e), str(e)\n"
        "else:\n"
        "    raise SystemExit('expected RuntimeError without faiss')\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_import_db_does_not_import_faiss():
    code = (
        "import sys\n"
        "import mmcif_vec.db\n"
        "import mmcif_vec.faiss_ann\n"
        "assert 'faiss' not in sys.modules, 'faiss imported eagerly'\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
