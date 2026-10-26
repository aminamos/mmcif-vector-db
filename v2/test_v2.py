import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from v2.core import load_chains, find_families
from v2.complexes import build_complexes, describe

SAMPLES = Path(__file__).parents[1] / "samples"

def _group_of(families, uid):
    for i, fam in enumerate(families):
        if any(c.uid == uid for c in fam):
            return i
    raise AssertionError(uid)

def test_v2_families():
    chains = load_chains([SAMPLES])
    assert len(chains) == 20  # 30 raw chains deduped to 20
    fams = find_families(chains)
    # Foxk1 together, including full-length AFDB vs short PDB domains
    assert _group_of(fams, "2A3S:A") == _group_of(fams, "2D2W:A") == _group_of(fams, "AF-P42128-F1:A")
    # MEF2 main chains together
    assert _group_of(fams, "1EGW:E") == _group_of(fams, "1N6J:C") == _group_of(fams, "1TQE:E")
    # globin alpha with alpha, beta with beta (different chains, not forced together)
    assert _group_of(fams, "1A3N:A") == _group_of(fams, "1HHO:A")
    assert _group_of(fams, "1A3N:B") == _group_of(fams, "1HHO:B")
    # unrelated proteins stay on their own, not absorbed into a family
    for uid in ("1LYZ:A", "1EMA:A", "1UBQ:A", "6LU7:A"):
        fam = fams[_group_of(fams, uid)]
        assert len(fam) == 1, (uid, [c.uid for c in fam])

def test_v2_complexes():
    chains = load_chains([SAMPLES])
    fams = find_families(chains)
    comps, groups = build_complexes([SAMPLES], fams)
    by_entry = {c["entry_id"]: c for c in comps}
    # 1A3N deposits 2 alpha + 2 beta; 1HHO deposits 1 + 1 (the rest is
    # symmetry-generated). Same recipe in different proportions.
    assert "Family 3 x2" in describe(by_entry["1A3N"])
    assert set(by_entry["1A3N"]["recipe"]) == set(by_entry["1HHO"]["recipe"])
    # and the two haemoglobins group as the same kind of complex
    joined = [g for g in groups if len(g) > 1 and g[0]["entry_id"] in ("1A3N", "1HHO")]
    assert joined and {c["entry_id"] for c in joined[0]} == {"1A3N", "1HHO"}
