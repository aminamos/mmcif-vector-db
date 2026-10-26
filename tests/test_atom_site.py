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

# All-atom fixture: header must declare type_symbol for the parser to read it.
ALL_ATOM = """data_ALLATOM
_entry.id ALLATOM
_struct.title 'all-atom fixture'
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
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
ATOM 1 N N . ALA A X 1 101 ? 0.0 1.0 0.0 1.00 9.0 1 1
ATOM 2 C CA . ALA A X 1 101 ? 1.5 0.0 0.0 1.00 10.0 1 1
ATOM 3 C C . ALA A X 1 101 ? 2.5 1.0 0.0 1.00 11.0 1 1
ATOM 4 C CB . ALA A X 1 101 ? 1.5 -1.5 0.0 1.00 12.0 1 1
ATOM 5 C CB B ALA A X 1 101 ? 9.0 9.0 9.0 1.00 12.0 1 1
ATOM 6 C CB . ALA A X 1 101 ? 8.0 8.0 8.0 1.00 12.0 1 1
ATOM 7 O O . ALA A X 1 101 ? 2.5 2.0 0.0 0.00 11.0 1 1
ATOM 8 N N . GLY A X 2 102 ? 3.8 0.0 0.0 1.00 10.0 1 1
ATOM 9 C CA . GLY A X 2 102 ? 5.3 0.0 0.0 1.00 10.0 1 1
HETATM 10 S SD . MSE A X 3 103 ? 7.0 0.0 0.0 1.00 10.0 1 1
HETATM 11 C CA . MSE A X 3 103 ? 8.5 0.0 0.0 1.00 10.0 1 1
ATOM 12 C CA . MSE A X 4 104 ? 11.0 0.0 0.0 1.00 10.0 1 1
ATOM 13 C CB . MSE A X 4 104 ? 12.0 0.0 0.0 1.00 10.0 1 1
ATOM 14 C CA . SER A X 5 105 ? 15.0 0.0 0.0 1.00 10.0 1 1
ATOM 15 C CA . THR A X 6 106 ? 19.0 0.0 0.0 1.00 10.0 1 1
"""

def test_all_atom_retention_and_filters(tmp_path):
    p = tmp_path / "allatom.cif"
    p.write_text(ALL_ATOM)
    (rec,) = parse_cif(p)
    # HETATM MSE (residue 3) is dropped by the ATOM-only rule despite
    # MSE->M in AA3; ATOM MSE (residue 4) is retained as M. Documented
    # current semantics, not a silent change.
    assert rec.sequence == "AGMST"
    ala, gly, mse = rec.residues[:3]
    # side-chain atoms retained in row order; altloc-B CB filtered,
    # duplicate '.' CB deduped (first seen), zero-occupancy O dropped
    assert [a.name for a in ala.atoms] == ["N", "CA", "C", "CB"]
    assert [a.element for a in ala.atoms] == ["N", "C", "C", "C"]
    assert ala.atoms[3].coord == (1.5, -1.5, 0.0)
    assert ala.atoms[0].b_factor == 9.0 and ala.atoms[0].occupancy == 1.0
    # Residue.coord / b_factor remain the CA values, not the first (N) atom's
    assert ala.coord == (1.5, 0.0, 0.0) == ala.atoms[1].coord
    assert ala.b_factor == 10.0
    assert rec.ca_coords[0] == (1.5, 0.0, 0.0)
    assert [a.name for a in gly.atoms] == ["N", "CA"]
    assert [a.name for a in mse.atoms] == ["CA", "CB"] and mse.aa == "M"

def test_real_sample_atoms_populated():
    ex = Path(__file__).parents[1] / "samples"
    rec = parse_cif(ex / "1ubq.cif")[0]
    assert all(r.atoms for r in rec.residues)
    assert any(a.name == "CB" for r in rec.residues for a in r.atoms)
    # CA coord still equals the CA atom's coord for every residue with a CA
    for r in rec.residues:
        ca = [a for a in r.atoms if a.name == "CA"]
        if ca:
            assert r.coord == ca[0].coord

def test_real_samples_still_parse():
    ex = Path(__file__).parents[1] / "samples"
    rec = parse_cif(ex / "1egw.cif")[0]
    assert len(rec.sequence) == len(rec.ca_coords)
    assert rec.label_range is not None
