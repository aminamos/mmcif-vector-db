"""Rigid-motion invariance of the local (coordinate-consuming) features.

``struct_vector`` is built only from CA-CA distances (pairwise / lagged
distance histograms, radius of gyration, end-to-end distance, contact
fraction), so it must be invariant to any proper rotation + translation
of the coordinates.  ``seq_vector`` / ``text_vector`` take no coordinates
at all (sequence string / metadata text only), so they are invariant by
construction; only the coordinate-consuming path (``struct_vector`` and
the ``combined_vector`` built on it) is exercised here.

Known limitation (chirality blindness): the distance histograms are also
invariant under *reflection* (det = -1), i.e. a structure and its mirror
image get identical vectors.  That is asserted explicitly below, not
hidden: distance-only features cannot distinguish enantiomorphic folds.
"""
from pathlib import Path

import numpy as np
import pytest

from mmcif_vec.features import combined_vector, seq_vector, struct_vector, text_vector
from mmcif_vec.parse import parse_cif

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
FILES = ["1ubq.cif", "1lyz.cif", "1a3n.cif"]


def _rotation(rng: np.random.Generator, det: float = 1.0) -> np.ndarray:
    """Random orthogonal matrix with prescribed determinant sign, via QR."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))  # sign fix: unique Q with positive diag(R)
    if np.linalg.det(q) * det < 0:
        q[:, 0] *= -1
    return q


def _transform(ca, rotation, translation):
    return (np.asarray(ca, dtype=np.float64) @ rotation.T + translation).tolist()


@pytest.mark.parametrize("filename", FILES)
def test_struct_vector_invariant_to_random_proper_rotations(filename):
    record = parse_cif(SAMPLES / filename)[0]
    base_struct = struct_vector(record.ca_coords)
    # seq/text vectors take no coordinates (invariant by construction);
    # they are computed once and reused for every rotated copy.
    base_combined = combined_vector(
        seq_vector(record.sequence), base_struct, text_vector(record.text))

    for seed in range(5):
        rng = np.random.default_rng(2024 + seed)
        rotation = _rotation(rng, det=+1.0)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        translation = rng.standard_normal(3) * 100.0

        rotated = struct_vector(_transform(record.ca_coords, rotation, translation))
        assert np.max(np.abs(rotated - base_struct)) < 1e-5

        combined = combined_vector(
            seq_vector(record.sequence), rotated, text_vector(record.text))
        np.testing.assert_array_equal(combined, base_combined)


@pytest.mark.parametrize("filename", FILES)
def test_struct_vector_invariant_to_translation_only(filename):
    record = parse_cif(SAMPLES / filename)[0]
    base = struct_vector(record.ca_coords)
    moved = struct_vector(_transform(record.ca_coords, np.eye(3), np.array([37.5, -12.25, 8.0])))
    assert np.max(np.abs(moved - base)) < 1e-5


@pytest.mark.parametrize("filename", FILES)
def test_struct_vector_reflection_invariant_chirality_blindness(filename):
    # Reflection (det = -1) is NOT a proper rotation, yet the distance-based
    # histogram features are reflection-invariant too: chirality blindness,
    # a known limitation of these features.
    record = parse_cif(SAMPLES / filename)[0]
    base = struct_vector(record.ca_coords)
    rng = np.random.default_rng(99)
    reflection = _rotation(rng, det=-1.0)
    assert np.linalg.det(reflection) == pytest.approx(-1.0)
    reflected = struct_vector(
        _transform(record.ca_coords, reflection, rng.standard_normal(3) * 10.0))
    assert np.max(np.abs(reflected - base)) < 1e-5
