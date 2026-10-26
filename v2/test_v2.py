import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from v2.core import load_chains, find_families

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
