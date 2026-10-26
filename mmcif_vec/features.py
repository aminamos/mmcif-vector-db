"""Sequence + structure vectors that are comparable across PDB/AFDB mmCIFs.

Structure features are translation/rotation invariant: no raw xyz is embedded.
Representative atoms = CA trace only (first model).
"""
from __future__ import annotations
import hashlib, re
import numpy as np

DIM = 256
AA = "ACDEFGHIKLMNPQRSTVWY"

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)

def _hash_into(v: np.ndarray, token: str, weight: float = 1.0):
    h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
    v[h % len(v)] += weight

def seq_vector(seq: str) -> np.ndarray:
    """AA composition + hashed 2/3-mers."""
    seq = "".join(c for c in seq.upper() if c in AA) or "A"
    v = np.zeros(DIM, dtype=np.float64)
    for i, c in enumerate(seq): v[AA.index(c)] += 1.0
    for k in (2, 3):
        for i in range(len(seq) - k + 1): _hash_into(v, seq[i:i+k])
    return _norm(v)

def text_vector(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float64)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        _hash_into(v, tok); _hash_into(v, "w:" + tok[:4])
    return _norm(v)

def _hist(vals, bins, lo, hi):
    if len(vals) == 0: return np.zeros(bins)
    h, _ = np.histogram(vals, bins=bins, range=(lo, hi))
    return h / max(h.sum(), 1)

def struct_vector(ca: list) -> np.ndarray:
    """Invariant CA-trace fingerprint: pairwise + local distance histograms."""
    x = np.array(ca, dtype=np.float64)
    if len(x) < 5: return np.zeros(DIM, dtype=np.float32)
    n = len(x)
    # pairwise distances (sample for long chains)
    idx = np.arange(n) if n <= 400 else np.linspace(0, n-1, 400).astype(int)
    xs = x[idx]
    d = np.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
    pair = _hist(d[np.triu_indices(len(xs), 1)], 64, 0, 64)
    feats = [pair]
    for lag, bins in ((2, 16), (3, 16), (4, 16), (8, 16)):
        if n > lag: feats.append(_hist(np.linalg.norm(x[lag:] - x[:-lag], axis=1), bins, 0, 32))
    rg = np.sqrt(((x - x.mean(0))**2).sum(1).mean())
    contacts = (d < 8.0).sum() / max(len(xs)**2, 1)
    scalars = np.array([np.log1p(n)/8, rg/max(n,1), np.linalg.norm(x[-1]-x[0])/max(n,1), contacts])
    v = np.concatenate(feats + [_hist(scalars, 10, 0, 1) if False else np.pad(scalars, (0, 10-len(scalars)))])
    out = np.zeros(DIM); out[:min(DIM, len(v))] = v[:DIM]
    return _norm(out)

def combined_vector(seq: np.ndarray, struct: np.ndarray, text: np.ndarray,
                      w_seq=0.45, w_struct=0.45, w_text=0.10) -> np.ndarray:
    return _norm(np.concatenate([seq*np.sqrt(w_seq), struct*np.sqrt(w_struct), text*np.sqrt(w_text)]))
