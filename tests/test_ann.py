import subprocess
import sys
from pathlib import Path

import numpy as np
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
def ann_setup(tmp_path_factory):
    pytest.importorskip("lancedb")
    from mmcif_vec.ann import ANNIndex

    tmp = tmp_path_factory.mktemp("ann")
    db = _ingest_samples(DB(str(tmp / "t.db")))
    idx = ANNIndex.build(db, tmp / "ann")
    return db, idx, tmp / "ann"


def test_ann_query_pool_contains_self_and_top1_matches_exact(ann_setup):
    db, idx, _ = ann_setup
    rec = parse_cif(SAMPLES / "1ubq.cif")[0]
    q_struct = struct_vector(rec.ca_coords)
    q = combined_vector(seq_vector(rec.sequence), q_struct, text_vector(""))
    pool = idx.query(q, pool_k=20)
    assert pool, "ANN query returned no candidates"
    assert rec.uid in [uid for uid, _ in pool]
    # best-first ordering by approximate score
    scores = [s for _, s in pool]
    assert scores == sorted(scores, reverse=True)

    exact = search(db, q_seq=rec.sequence, q_struct=q_struct, k=5)
    ann_hits = search(db, q_seq=rec.sequence, q_struct=q_struct, k=5, ann=idx)
    assert exact and ann_hits
    assert ann_hits[0]["uid"] == exact[0]["uid"] == rec.uid


def test_search_ann_result_shape_identical_to_exact(ann_setup):
    db, idx, _ = ann_setup
    rec = parse_cif(SAMPLES / "2a3s.cif")[0]
    q = dict(q_seq=rec.sequence, q_struct=struct_vector(rec.ca_coords), k=5)
    exact = search(db, **q)
    ann_hits = search(db, ann=idx, **q)
    assert ann_hits
    assert len(ann_hits) == len(exact)
    for h_ann, h_exact in zip(ann_hits, exact):
        assert list(h_ann.keys()) == list(h_exact.keys())
        assert {k: type(v) for k, v in h_ann.items()} == {
            k: type(v) for k, v in h_exact.items()}
    # small corpus uses exact flat scan, so results should agree fully
    assert [h["uid"] for h in ann_hits] == [h["uid"] for h in exact]


def test_ann_ignored_for_text_only_mode(ann_setup):
    db, idx, _ = ann_setup
    exact = search(db, q_text="myocyte nuclear factor", k=3)
    ann_hits = search(db, q_text="myocyte nuclear factor", k=3, ann=idx)
    assert [h["uid"] for h in ann_hits] == [h["uid"] for h in exact]


def test_db_and_ann_import_without_lancedb():
    # Fresh interpreter: importing db (and ann) must not import lancedb,
    # and using ANN without the engine must raise a clear error.
    code = (
        "import sys\n"
        "sys.modules['lancedb'] = None\n"  # make `import lancedb` fail
        "import mmcif_vec.db\n"
        "assert 'lancedb' not in [m for m in sys.modules if m == 'lancedb' and sys.modules[m] is not None]\n"
        "import mmcif_vec.ann as ann\n"
        "try:\n"
        "    ann.ANNIndex.load('nonexistent-ann-dir')\n"
        "except RuntimeError as e:\n"
        "    assert 'ann' in str(e) and 'lancedb' in str(e), str(e)\n"
        "else:\n"
        "    raise SystemExit('expected RuntimeError without lancedb')\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_import_db_does_not_import_lancedb():
    code = (
        "import sys\n"
        "import mmcif_vec.db\n"
        "import mmcif_vec.ann\n"
        "assert 'lancedb' not in sys.modules, 'lancedb imported eagerly'\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
