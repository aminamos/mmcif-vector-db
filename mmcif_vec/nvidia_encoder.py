"""OPTIONAL NVIDIA La-Proteina autoencoder encoder — not the default.

Measured outcome (2026-10-26, see EVALUATION.md / results/BENCHMARK_RUN.md):
this encoder does NOT beat the local encoder and is not the default.
On samples/ it scores 19/22 (local 21/22) and on samples_hard/ 16/17
leave-entry-out (local 16/17, 17/17 leave-self-out) with the validated
pooling below. It stays an optional encoder behind the same contract.

Contract (same as the local encoder, see docs/nvidia-autoencoder-draft.md):

    encode_chain(residues) -> np.ndarray[float32, 8], L2-normalised

`residues` is a list of ``mmcif_vec.parse.Residue`` (all-atom, as populated
by ``parse_cif``) or a ``ChainRecord``. Per-residue latents are the
autoencoder's deterministic posterior ``mean`` (never the stochastic
``z_latent`` sample), shape (n_res, 8). Aggregation in ``encode_chain``
is mean-pooling over residues of that latent-mean -> 8-d, L2-normalised.
Mean is the validated aggregation: mean+max+std (24-d) scored only
10/22 + 8/17 and mean+max (16-d) 9/22 + 8/17 — max/std destroy signal.
Pooling variants for benchmarks live in ``POOLINGS``.

Residue limits (never silently truncate — a clear ``ValueError`` is
raised instead): AE1 (``AE1_ucond_512.ckpt``) <= 512 residues; use AE2
(``AE2_ucond_800.ckpt``) for longer chains (<= 896 residues, e.g. the
719-residue AF-P42128-F1 chain).

Batch construction mirrors la-proteina's own pipeline, reimplemented
from source (no graphein re-parse of files):

- ``proteinfoundation/datasets/pdb_data.py``: ``protein_to_pyg`` yields
  atom37 coords/mask in PDB order, reordered to OpenFold order by
  ``PDB_TO_OPENFOLD_INDEX_TENSOR`` in ``__getitem__``; ``residue_type``
  is ``openfold.np.residue_constants.resname_to_idx`` per residue;
  ``residue_pdb_idx`` is the integer auth seq id; ``coord_mask`` is
  ``coords != fill_value`` per atom. We build the same atom37 arrays
  directly in OpenFold order from parsed residues.
- transforms (``configs/dataset/pdb/pdb_train_ucond.yaml`` order,
  minus the training-only random ``GlobalRotationTransform``, omitted
  so encoding is deterministic): ``CoordsToNanometers``
  (``coords_nm = coords / 10``), ``CenterStructureTransform`` (subtract
  the mean CA ``coords_nm``), ``ChainBreakPerResidueTransform``
  (CA--CA distance on Angstrom ``coords`` > 4.0, last residue False).
  Their transform classes are not instantiated: with current
  torch_geometric they are abstract (only ``__call__`` overridden).
- dense collation (``utils/dense_padding_data_loader.py``) pads a
  single graph to ``[1, n, ...]`` and attaches ``mask_dict[attr]`` =
  ``padded != padding_value``; ``AutoEncoder.encode`` then sets
  ``batch["mask"] = mask_dict["coords"][..., 0, 0]``.
- encoder features (``configs/nn_ae/nn_130m.yaml``): ``coords`` (A,
  used by backbone/side-chain angle features), ``coords_nm``,
  ``coord_mask``, ``residue_type``, ``residue_pdb_idx``,
  ``chain_breaks_per_residue`` and ``chains`` (per-residue chain index,
  0 for a single chain — ``ChainIdxSeqFeat`` raises without it).

This module imports only numpy at module level. torch / la-proteina are
lazy-imported inside functions; when unavailable, ``NotAvailable`` is
raised (never a raw ImportError traceback). The heavy stack lives in
its own WSL2/Linux CUDA environment outside this repo (validated on
WSL2 Ubuntu-22.04, torch 2.7.0+cu118) — see
docs/nvidia-autoencoder-draft.md and ``scripts/encode_nvidia.py``.
NEVER add torch to this repo's deps.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

FIXED_DIM = 8
LATENT_DIM = 8

# OpenFold atom37 order (openfold.np.residue_constants.atom_types);
# index 0=N, 1=CA, 2=C, 3=CB as la-proteina's features assume.
ATOM37_TYPES = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]
ATOM37_INDEX = {name: i for i, name in enumerate(ATOM37_TYPES)}

# OpenFold resname_to_idx (restype order A R N D C Q E G H I L K M F P S T W Y V).
# MSE (selenomethionine) maps to MET, as in OpenFold.
RESNAME_TO_IDX = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLN": 5,
    "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11,
    "MET": 12, "MSE": 12, "PHE": 13, "PRO": 14, "SER": 15, "THR": 16,
    "TRP": 17, "TYR": 18, "VAL": 19, "SEC": 4,
}
_AA1_TO_IDX = {}
for _name, _idx in RESNAME_TO_IDX.items():
    _AA1_TO_IDX.setdefault(_name, _idx)
AA1_TO_IDX = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7,
    "H": 8, "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19, "U": 4,
}

#: checkpoint filename fragment -> max residues (never truncate past it)
MODEL_MAX_RESIDUES = {"AE1": 512, "AE2": 896}
DEFAULT_MAX_RESIDUES = 512


class NotAvailable(RuntimeError):
    pass


_NOT_AVAILABLE_MSG = (
    "NVIDIA La-Proteina autoencoder not available: torch / la-proteina "
    "(proteinfoundation) could not be imported in this environment. "
    "It lives in its own pinned WSL venv, never in this repo's uv env. "
    "See docs/nvidia-autoencoder-draft.md and run scripts/encode_nvidia.py "
    "inside that venv (PYTHONPATH=<la-proteina>:<repo root>)."
)


def _residues_of(residues):
    """Accept a ChainRecord or a plain list of Residue."""
    if hasattr(residues, "residues"):
        return list(residues.residues)
    return list(residues)


def build_atom37(residues):
    """Pure-numpy atom37 construction from parsed residues.

    Returns (coords, coord_mask, residue_type, residue_pdb_idx):
    coords float32 (n, 37, 3) in Angstrom, OpenFold atom order;
    coord_mask bool (n, 37); residue_type int64 (n,) OpenFold indices;
    residue_pdb_idx int64 (n,) auth seq id when numeric, else the
    label seq id, else 1-based position (la-proteina parses it as
    ``int(residue_id.split(':')[2])``, i.e. the PDB/auth numbering).

    Residues with an empty ``atoms`` list contribute only a CA atom at
    ``Residue.coord`` so older / hand-built residues still encode.
    """
    residues = _residues_of(residues)
    n = len(residues)
    coords = np.zeros((n, 37, 3), dtype=np.float32)
    coord_mask = np.zeros((n, 37), dtype=bool)
    residue_type = np.zeros((n,), dtype=np.int64)
    residue_pdb_idx = np.zeros((n,), dtype=np.int64)
    for i, res in enumerate(residues):
        comp = (getattr(res, "comp_id", "") or "").strip().upper()
        if comp in RESNAME_TO_IDX:
            residue_type[i] = RESNAME_TO_IDX[comp]
        else:
            residue_type[i] = AA1_TO_IDX.get(getattr(res, "aa", "A"), 0)
        pdb_idx = None
        for cand in (getattr(res, "auth_seq_id", None), getattr(res, "label_seq_id", None)):
            try:
                pdb_idx = int(str(cand).strip())
                break
            except (TypeError, ValueError):
                continue
        residue_pdb_idx[i] = pdb_idx if pdb_idx is not None else i + 1
        atoms = getattr(res, "atoms", None) or []
        if not atoms:
            coords[i, ATOM37_INDEX["CA"]] = np.asarray(res.coord, dtype=np.float32)
            coord_mask[i, ATOM37_INDEX["CA"]] = True
            continue
        for atom in atoms:
            j = ATOM37_INDEX.get(atom.name)
            if j is None or coord_mask[i, j]:
                continue
            coords[i, j] = np.asarray(atom.coord, dtype=np.float32)
            coord_mask[i, j] = True
    return coords, coord_mask, residue_type, residue_pdb_idx


def chain_breaks_per_residue(coords) -> np.ndarray:
    """ChainBreakPerResidueTransform: CA dist > 4.0 A, last = False."""
    coords = np.asarray(coords)
    n = len(coords)
    breaks = np.zeros((n,), dtype=bool)
    if n > 1:
        ca = coords[:, ATOM37_INDEX["CA"], :].astype(np.float64)
        breaks[:-1] = np.linalg.norm(ca[1:] - ca[:-1], axis=1) > 4.0
    return breaks


def build_batch_arrays(residues):
    """Numpy batch arrays mirroring la-proteina transforms (see module docstring)."""
    coords, coord_mask, residue_type, residue_pdb_idx = build_atom37(residues)
    coords_nm = coords.astype(np.float64) / 10.0  # CoordsToNanometers
    if len(coords_nm):
        # CenterStructureTransform: centre on mean CA coords_nm.
        coords_nm = coords_nm - coords_nm[:, ATOM37_INDEX["CA"], :].mean(axis=0)[None, None, :]
    return {
        "coords": coords,
        "coords_nm": coords_nm.astype(np.float32),
        "coord_mask": coord_mask,
        "residue_type": residue_type,
        "residue_pdb_idx": residue_pdb_idx,
        "chain_breaks_per_residue": chain_breaks_per_residue(coords),
        "chains": np.zeros((len(residue_type),), dtype=np.int64),
        "seq_pos": np.arange(len(residue_type), dtype=np.int64)[:, None],
    }


def _build_batch(residues, torch, device):
    arr = build_batch_arrays(residues)
    batch = {
        k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in arr.items()
    }
    # Dense-collation mask_dict: (padded != padding_value). Single
    # unpadded chain -> all True; residue mask derives from coords.
    batch["mask_dict"] = {
        "coords": torch.ones_like(batch["coords"], dtype=torch.bool),
        "coord_mask": torch.ones_like(batch["coord_mask"], dtype=torch.bool),
        "coords_nm": torch.ones_like(batch["coords_nm"], dtype=torch.bool),
        "residue_type": torch.ones((1, len(arr["residue_type"])), dtype=torch.bool, device=device),
        "residue_pdb_idx": torch.ones((1, len(arr["residue_type"])), dtype=torch.bool, device=device),
        "chain_breaks_per_residue": torch.ones((1, len(arr["residue_type"])), dtype=torch.bool, device=device),
        "chains": torch.ones((1, len(arr["residue_type"])), dtype=torch.bool, device=device),
    }
    return batch


def pool_latents(latents: np.ndarray, pooling: str = "mean") -> np.ndarray:
    """Pool (n, 8) per-residue latents to a fixed vector (not normalised)."""
    x = np.asarray(latents, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"latents must be a non-empty (n, d) array, got {x.shape}")
    mean = x.mean(axis=0)
    if pooling == "mean":
        return mean
    if pooling == "max":
        return x.max(axis=0)
    if pooling == "mean_max":
        return np.concatenate([mean, x.max(axis=0)])
    if pooling == "mean_max_std":
        return np.concatenate([mean, x.max(axis=0), x.std(axis=0)])
    raise ValueError(f"unknown pooling {pooling!r}; choose from {sorted(POOLINGS)}")


POOLINGS = {
    "mean": lambda x: pool_latents(x, "mean"),
    "max": lambda x: pool_latents(x, "max"),
    "mean_max": lambda x: pool_latents(x, "mean_max"),
    "mean_max_std": lambda x: pool_latents(x, "mean_max_std"),
}
#: output dim per pooling (latent dim 8)
POOLING_DIMS = {"mean": 8, "max": 8, "mean_max": 16, "mean_max_std": 24}


def _l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def _import_torch_and_ae():
    try:
        import torch  # noqa: F401
        from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder  # noqa: F401
    except Exception as e:  # ImportError and its many heavy-stack cousins
        raise NotAvailable(_NOT_AVAILABLE_MSG) from e
    return torch, AutoEncoder


def _max_residues_for(ckpt_path, model=None) -> int:
    if model is not None and getattr(model, "_mmcif_max_residues", None):
        return int(model._mmcif_max_residues)
    name = Path(str(ckpt_path)).name.upper() if ckpt_path else ""
    for frag, limit in MODEL_MAX_RESIDUES.items():
        if frag in name:
            return limit
    return DEFAULT_MAX_RESIDUES


_MODEL_CACHE: dict = {}


def load_model(ckpt_path, device="cuda"):
    """Load (and cache) the frozen La-Proteina AutoEncoder from a checkpoint.

    Returns the model in eval mode on ``device``, tagged with
    ``_mmcif_max_residues`` / ``_mmcif_ckpt_path`` for limit checks.
    Raises NotAvailable when torch / la-proteina are not importable.
    """
    ckpt_path = str(ckpt_path)
    key = (str(Path(ckpt_path).resolve()) if Path(ckpt_path).exists() else ckpt_path, str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch, AutoEncoder = _import_torch_and_ae()
    if not Path(ckpt_path).exists():
        raise NotAvailable(f"La-Proteina checkpoint not found: {ckpt_path}. "
                           "See docs/nvidia-autoencoder-draft.md for download locations.")
    model = AutoEncoder.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model._mmcif_max_residues = _max_residues_for(ckpt_path)
    model._mmcif_ckpt_path = ckpt_path
    _MODEL_CACHE[key] = model
    return model


def encode_residue_latents(residues, model, device="cuda") -> np.ndarray:
    """(n_res, 8) float32 deterministic latents: the encoder's ``mean``.

    Never uses ``z_latent`` (a stochastic sample). Raises ValueError
    for chains longer than the model's residue limit (AE1 <= 512,
    AE2 <= 896) — never silently truncates.
    """
    residues = _residues_of(residues)
    if not residues:
        raise ValueError("encode_residue_latents: no residues")
    limit = _max_residues_for(getattr(model, "_mmcif_ckpt_path", None), model)
    if len(residues) > limit:
        raise ValueError(
            f"chain has {len(residues)} residues, exceeding the {limit}-residue "
            f"limit of this checkpoint; use the AE2 checkpoint (<=896 residues) "
            f"for longer chains instead of truncating")
    try:
        import torch
    except Exception as e:
        raise NotAvailable(_NOT_AVAILABLE_MSG) from e
    batch = _build_batch(residues, torch, device)
    with torch.no_grad():
        if hasattr(model, "encode"):
            out = model.encode(batch)
        elif hasattr(model, "encoder"):
            # Raw EncoderTransformer: AutoEncoder.encode sets batch["mask"] first.
            batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
            out = model.encoder(batch)
        else:
            raise NotAvailable(f"model {type(model)!r} has neither .encode nor .encoder")
    mean = out["mean"] if isinstance(out, dict) else out[1]
    arr = mean.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape != (len(residues), LATENT_DIM):
        # Still return what the model produced if latent dim differs by
        # checkpoint, but the settled AE checkpoints are dim 8.
        if arr.ndim != 2 or arr.shape[0] != len(residues):
            raise ValueError(f"unexpected encoder mean shape {arr.shape} for {len(residues)} residues")
    return arr.astype(np.float32, copy=False)


def encode_chain(residues, model=None, device="cuda", ckpt_path=None, pooling="mean") -> np.ndarray:
    """Chain -> (8,) float32 L2-normalised vector (mean pooling, validated).

    If ``model`` is None, a checkpoint is loaded from ``ckpt_path`` or
    the ``MMCIF_LA_PROTEINA_CKPT`` env var. Raises NotAvailable (not
    ImportError) when torch / la-proteina / a checkpoint are unavailable.
    """
    if model is None:
        ckpt_path = ckpt_path or os.environ.get("MMCIF_LA_PROTEINA_CKPT")
        if not ckpt_path:
            # Probe importability first so the failure is NotAvailable
            # with guidance, whether the stack or just the ckpt is missing.
            _import_torch_and_ae()
            raise NotAvailable(
                "No La-Proteina model or checkpoint given: pass model=..., "
                "ckpt_path=..., or set MMCIF_LA_PROTEINA_CKPT. "
                "See docs/nvidia-autoencoder-draft.md.")
        model = load_model(ckpt_path, device=device)
    latents = encode_residue_latents(residues, model, device=device)
    return _l2norm(pool_latents(latents, pooling))
