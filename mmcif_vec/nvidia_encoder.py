"""OPTIONAL NVIDIA autoencoder encoder (La-Proteina / RFD4-Proteina) — not the MVP default.

The friend asked for NVIDIA first; the MVP does NOT depend on it.
Reason: 256M-param autoencoder, CUDA-oriented checkpoints, and generation
weights — heavy to ship before clustering/RAG value is proven.

Swap point: features.seq_vector/struct_vector/combined_vector have the same
contract this module would fulfil (CIF chain -> fixed-dim float32 vector).
To enable, in the RosettaCommons/RFD4-Proteina (or NVIDIA-Digital-Bio/
la-proteina) checkout, load the frozen autoencoder, encode a chain, and
mean-pool its per-residue latents (sequence + side-chain) alongside the
explicit CA trace. Then pass those vectors into db.DB.ingest via a custom
encoder. Until `encode_chain()` below can import that code + weights, it
raises NotAvailable so the fallback path is explicit, never silent.
"""
from __future__ import annotations
import numpy as np

class NotAvailable(RuntimeError):
    pass

def encode_chain(sequence: str, ca_coords: list) -> np.ndarray:
    try:
        import la_proteina  # noqa: F401  (NVIDIA-Digital-Bio/la-proteina)
    except ImportError as e:
        raise NotAvailable(
            "NVIDIA La-Proteina/RFD4-Proteina autoencoder not installed. "
            "MVP uses features.py (sequence k-mers + invariant CA fingerprint). "
            "Install the RosettaCommons/RFD4-Proteina checkout + weights and "
            "implement this function to swap it in.") from e
    raise NotAvailable("la_proteina importable but encoder wiring not implemented yet")
