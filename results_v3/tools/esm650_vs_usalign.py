#!/usr/bin/env python
"""ESM-650M cosine vs USalign TM on the 2,000 tm_pairs.tsv pairs.

encoder_vs_usalign.py hardcodes its encoder set/output, so this small script
reuses pearson/spearman/load_vectors from v3/evaluate_v3.py for esm650 only
and writes results_v3/esm650_vs_usalign.json (does not overwrite the prior
encoder_vs_usalign.json).
Run from repo root: uv run python results_v3/tools/esm650_vs_usalign.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v3"))
from evaluate_v3 import load_vectors, pearson, spearman  # noqa: E402

RESULTS = REPO / "results_v3"

def main() -> int:
    pairs, usalign = [], []
    with open(RESULTS / "tm_pairs.tsv", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("error"):
                continue
            pairs.append((row["a"], row["b"]))
            usalign.append(float(row["usalign_tm"]))
    us = np.asarray(usalign, dtype=np.float64)
    vectors = load_vectors(RESULTS / "esm2_t33_650m.npz")
    cos = np.empty(len(pairs), dtype=np.float64)
    for i, (a, b) in enumerate(pairs):
        va, vb = vectors[a].astype(np.float64), vectors[b].astype(np.float64)
        cos[i] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    out = {
        "source_pairs": "results_v3/tm_pairs.tsv",
        "n_pairs": len(pairs),
        "sampling": "2,000 uniform random unordered distinct-domain cross-entry pairs, seed 7 (tm_validation.py)",
        "encoder": "esm650",
        "model": "facebook/esm2_t33_650M_UR50D",
        "vectors": "results_v3/esm2_t33_650m.npz",
        "dim": int(next(iter(vectors.values())).shape[0]),
        "vs_usalign": {"n_pairs": len(pairs), "pearson": pearson(cos, us), "spearman": spearman(cos, us)},
    }
    (RESULTS / "esm650_vs_usalign.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"esm650 vs USalign: P={out['vs_usalign']['pearson']:.4f} S={out['vs_usalign']['spearman']:.4f} n={len(pairs)}")
    print(f"wrote {RESULTS / 'esm650_vs_usalign.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
