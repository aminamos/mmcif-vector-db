import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from mmcif_vec.parse import parse_cif
from mmcif_vec.db import DB, search
from mmcif_vec.features import struct_vector
from mmcif_vec.cluster import cluster

EX = Path(__file__).parents[1] / "examples"

def test_parse_pdb_and_afdb():
    assert parse_cif(EX/"2a3s.cif")[0].source == "PDB"
    af = parse_cif(EX/"AF-P42128-F1-model.cif")
    assert af and af[0].source == "AFDB" and len(af[0].ca_coords) > 100

def test_ingest_search_cluster_rag(tmp_path):
    db = DB(str(tmp_path/"t.db"))
    for f in EX.glob("*.cif"): db.ingest(f)
    assert db.stats()["chains"] >= 4
    # text search finds myocyte nuclear factor
    hits = search(db, q_text="myocyte nuclear factor", k=3)
    assert hits and "2A3S" in hits[0]["uid"]
    # structure+sequence similarity: 2A3S query should rank itself / AF Foxk1 high
    r = parse_cif(EX/"2a3s.cif")[0]
    sim = search(db, q_seq=r.sequence, q_struct=struct_vector(r.ca_coords), k=4)
    assert sim[0]["uid"].startswith("2A3S")
    assert sim[0]["struct_score"] is not None and sim[0]["seq_score"] is not None
    groups = cluster(db, k=2)
    assert sum(len(v) for v in groups.values()) == db.stats()["chains"]
