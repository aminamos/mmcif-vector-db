#!/usr/bin/env python
"""Encoder cosine vs USalign (and vs Foldseek gold) on the tm_pairs subsample.

Reads results_v3/tm_pairs.tsv (a, b, usalign_tm; 2,000 uniform random
cross-entry pairs, seed 7, see tm_validation.py / tm_validation.json) plus
the three encoder NPZs, and writes results_v3/encoder_vs_usalign.json with,
per encoder, Pearson/Spearman of cosine similarity against:
  * USalign symmetric TM (mean of the two length-normalised TM-scores), and
  * Foldseek symmetric TM (mean(qtmscore, ttmscore), dedupe by max bits)
on exactly the same pairs. The Foldseek-vs-USalign agreement on those pairs
is recomputed here too, so the JSON is self-contained.

Pearson/Spearman (and the NPZ loader) are reused from v3/evaluate_v3.py -
numpy only, no scipy. The 270 MB Foldseek TSV is streamed once and only
rows for the 2,000 target pairs are retained.

Run from the repo root:  uv run python results_v3/tools/encoder_vs_usalign.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v3"))
from evaluate_v3 import _normalise_gold_id, load_vectors, pearson, spearman  # noqa: E402

RESULTS = REPO / "results_v3"
PAIRS_TSV = RESULTS / "tm_pairs.tsv"
GOLD_TSV = RESULTS / "foldseek_gold.tsv"
OUT = RESULTS / "encoder_vs_usalign.json"
ENCODERS = {"local": RESULTS / "local.npz", "ae2": RESULTS / "ae2.npz", "esm": RESULTS / "esm_t12.npz"}


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def main() -> int:
    pairs: list[tuple[str, str]] = []
    usalign: list[float] = []
    with open(PAIRS_TSV, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("error"):
                continue
            try:
                tm = float(row["usalign_tm"])
            except (TypeError, ValueError, KeyError):
                continue
            pairs.append(_key(row["a"], row["b"]))
            usalign.append(tm)
    usalign_arr = np.asarray(usalign, dtype=np.float64)
    wanted = set(pairs)
    print(f"usable USalign pairs: {len(pairs)}")

    # Stream the Foldseek gold TSV once, keeping only wanted pairs (max-bits dedupe).
    best: dict[tuple[str, str], tuple[float, float]] = {}  # key -> (tm, bits)
    with open(GOLD_TSV, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        iq, it, iqt, itt, ib = (header.index(c) for c in ("query", "target", "qtmscore", "ttmscore", "bits"))
        for row in reader:
            if len(row) <= max(iq, it, iqt, itt, ib):
                continue
            key = _key(_normalise_gold_id(row[iq]), _normalise_gold_id(row[it]))
            if key not in wanted:
                continue
            tm = 0.5 * (float(row[iqt]) + float(row[itt]))
            bits = float(row[ib])
            if key not in best or bits > best[key][1]:
                best[key] = (tm, bits)
    foldseek = np.asarray([best[k][0] if k in best else np.nan for k in pairs], dtype=np.float64)
    both = np.isfinite(foldseek)
    print(f"pairs also present in Foldseek gold: {int(both.sum())}/{len(pairs)}")

    out: dict = {
        "source_pairs": "results_v3/tm_pairs.tsv",
        "n_pairs": len(pairs),
        "sampling": "2,000 uniform random unordered distinct-domain cross-entry pairs, seed 7 (tm_validation.py)",
        "usalign_tm": {
            "mean": float(usalign_arr.mean()),
            "median": float(np.median(usalign_arr)),
            "fraction_ge_0.7": float((usalign_arr >= 0.7).mean()),
        },
        "foldseek_vs_usalign": {
            "n_pairs": int(both.sum()),
            "pearson": pearson(foldseek[both], usalign_arr[both]),
            "spearman": spearman(foldseek[both], usalign_arr[both]),
        },
        "encoders": {},
    }
    for name, npz in ENCODERS.items():
        vectors = load_vectors(npz)
        cos = np.empty(len(pairs), dtype=np.float64)
        for idx, (a, b) in enumerate(pairs):
            va, vb = vectors[a].astype(np.float64), vectors[b].astype(np.float64)
            cos[idx] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
        out["encoders"][name] = {
            "dim": int(next(iter(vectors.values())).shape[0]),
            "vs_usalign": {"n_pairs": len(pairs), "pearson": pearson(cos, usalign_arr), "spearman": spearman(cos, usalign_arr)},
            "vs_foldseek_same_pairs": {
                "n_pairs": int(both.sum()),
                "pearson": pearson(cos[both], foldseek[both]),
                "spearman": spearman(cos[both], foldseek[both]),
            },
        }
        e = out["encoders"][name]
        print(f"{name}: vs USalign P={e['vs_usalign']['pearson']:.3f} S={e['vs_usalign']['spearman']:.3f} | "
              f"vs Foldseek P={e['vs_foldseek_same_pairs']['pearson']:.3f} S={e['vs_foldseek_same_pairs']['spearman']:.3f}")
    print(f"Foldseek vs USalign: P={out['foldseek_vs_usalign']['pearson']:.3f} S={out['foldseek_vs_usalign']['spearman']:.3f}")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
