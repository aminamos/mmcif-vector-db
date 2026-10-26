"""Focused _atom_site handling tests: altloc, calcium, models, numbering."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from mmcif_vec.parse import parse_cif

FIXTURE = """data_TEST
_entry.id TEST
_struct.title 'atom_site fixture'
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
_atom_site.label_entity_id
ATOM 1 CA . ALA A X 1 101 ? 0.0 0.0 0.0 1.00 10.0 1 1
ATOM 2 CA B ALA A X 1 101 ? 9.0 9.0 9.0 1.00 10.0 1 1
ATOM 3 CA . GLY A X 2 102 ? 3.8 0.0 0.0 1.00 11.0 1 1
ATOM 4 CA . SER A X 3 103 ? 7.6 0.0 0.0 1.00 12.0 1 1
ATOM 5 CA . THR A X 4 104 ? 11.4 0.0 0.0 1.00 13.0 1 1
ATOM 6 CA . VAL A X 5 105 ? 15.2 0.0 0.0 1.00 14.0 1 1
ATOM 7 CA . LEU A X 6 106 ? 19.0 0.0 0.0 0.00 15.0 1 1
HETATM 8 CA . CA A X 7 107 ? 99.0 99.0 99.0 1.00 20.0 1 1
ATOM 9 CA . ALA A X 1 101 ? 50.0 50.0 50.0 1.00 10.0 2 1
"""

def test_atom_site_gaps(tmp_path):
    p = tmp_path / "test.cif"
    p.write_text(FIXTURE)
    (rec,) = parse_cif(p)
    # altloc B duplicate, zero-occupancy LEU, calcium HETATM, model 2: all excluded
    assert rec.sequence == "AGSTV"
    assert len(rec.ca_coords) == 5
    assert rec.ca_coords[0] == (0.0, 0.0, 0.0)  # model 1 alt A, not 9/50
    # both numbering schemes retained (SIFTS maps between them)
    assert rec.label_range == (1, 5)
    assert rec.auth_range == ("101", "105")
    assert rec.auth_chain_id == "X"
    assert rec.residues[0].b_factor == 10.0

def test_real_samples_still_parse():
    ex = Path(__file__).parents[1] / "samples"
    rec = parse_cif(ex / "1egw.cif")[0]
    assert len(rec.sequence) == len(rec.ca_coords)
    assert rec.label_range is not None
