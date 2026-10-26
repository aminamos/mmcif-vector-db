"""CPU-side nvidia_encoder tests: atom37 build, pooling, availability.

No torch / GPU here: the heavy stack is lazy-imported, so these run in
the repo uv env. GPU encode is proven separately in the WSL venv.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from mmcif_vec import nvidia_encoder as ne
from mmcif_vec.parse import Residue, parse_cif

EX = Path(__file__).parents[1] / "samples"


def test_atom37_from_real_chain():
    rec = parse_cif(EX / "1ubq.cif")[0]
    coords, mask, rtype, pdb_idx = ne.build_atom37(rec.residues)
    assert coords.shape == (76, 37, 3) and mask.shape == (76, 37)
    assert rtype.shape == (76,) and pdb_idx.shape == (76,)
    # CA present everywhere, coords match the parser's CA coord
    ca = ne.ATOM37_INDEX["CA"]
    assert mask[:, ca].all()
    np.testing.assert_allclose(coords[:, ca], np.array(rec.ca_coords), atol=1e-5)
    # first residue is MET (M): OpenFold index 12, has CB; GLY has no CB
    assert rtype[0] == 12
    gly = [i for i, r in enumerate(rec.residues) if r.aa == "G"]
    assert gly and not mask[gly[0], ne.ATOM37_INDEX["CB"]]


def test_batch_arrays_centered_nm_and_breaks():
    rec = parse_cif(EX / "1ubq.cif")[0]
    arr = ne.build_batch_arrays(rec.residues)
    ca_nm = arr["coords_nm"][:, ne.ATOM37_INDEX["CA"], :]
    np.testing.assert_allclose(ca_nm.mean(axis=0), [0, 0, 0], atol=1e-5)
    # nm = Angstrom/10 before centering: a CA-CA step is ~0.38 nm
    assert arr["chain_breaks_per_residue"].dtype == bool
    assert not arr["chain_breaks_per_residue"][-1]
    assert arr["chains"].tolist() == [0] * 76


def test_pooling_variants():
    rng = np.random.default_rng(0)
    lat = rng.normal(size=(10, 8)).astype(np.float32)
    # default pooling is the validated mean-only (8-d)
    assert ne.FIXED_DIM == 8
    default = ne.pool_latents(lat)
    assert default.shape == (8,)
    np.testing.assert_allclose(default, lat.mean(axis=0), rtol=1e-6)
    assert ne.pool_latents(lat, "mean").shape == (8,)
    assert ne.pool_latents(lat, "mean_max").shape == (16,)
    pooled = ne.pool_latents(lat, "mean_max_std")
    assert pooled.shape == (24,)
    np.testing.assert_allclose(pooled[:8], lat.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(pooled[16:], lat.std(axis=0), rtol=1e-6)
    vec = ne._l2norm(pooled)


def test_not_available_without_stack_or_ckpt():
    rec = parse_cif(EX / "1ubq.cif")[0]
    with pytest.raises(ne.NotAvailable):
        ne.encode_chain(rec.residues, model=None, ckpt_path=None)


def test_over_limit_never_truncates():
    residues = [Residue(i + 1, str(i + 1), "?", "ALA", "A", (0.0, 0.0, 0.0)) for i in range(600)]

    class FakeModel:
        _mmcif_ckpt_path = "AE1_ucond_512.ckpt"

    with pytest.raises(ValueError, match="512"):
        ne.encode_residue_latents(residues, FakeModel(), device="cpu")
