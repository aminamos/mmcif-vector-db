#!/usr/bin/env python
"""Encode every chain of a samples dir with the NVIDIA La-Proteina autoencoder.

Intended to run INSIDE the WSL la-proteina venv, e.g.:

  PYTHONPATH=<la-proteina>:<repo root> ~/laproteina-venv/bin/python \
    scripts/encode_nvidia.py --samples samples \
    --ckpt /path/to/AE1_ucond_512.ckpt \
    --out results/nvidia_ae1.npz --sidecar results/nvidia_ae1.json

Writes an NPZ mapping uid -> 8-d mean-pooled L2-normalised chain vector (plus
``<uid>__latents`` (n_res, 8) arrays with --latents) and a JSON
sidecar: per-chain seconds, failures with reasons (chains longer than
the checkpoint limit, e.g. >512 for AE1, are recorded as failures and
do NOT crash the run — rerun those with AE2 / --max-residues), peak
GPU memory, checkpoint path and package versions.

This script imports cleanly without torch (``--help`` works in the
repo's CPU-only uv env): all heavy imports happen inside main().
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="samples", help="dir (or file) of .cif samples")
    ap.add_argument("--ckpt", required=True, help="La-Proteina AE checkpoint (.ckpt)")
    ap.add_argument("--out", default="results/nvidia_ae1.npz", help="output NPZ path")
    ap.add_argument("--sidecar", default=None, help="output JSON sidecar (default: <out stem>.json)")
    ap.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    ap.add_argument("--latents", action="store_true",
                    help="also store per-residue latents as <uid>__latents in the NPZ")
    ap.add_argument("--max-residues", type=int, default=None,
                    help="override the checkpoint residue limit (default: AE1 512 / AE2 896 from filename)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Heavy / optional imports live here so --help and import work CPU-only.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import json
    import time

    import numpy as np

    from mmcif_vec import nvidia_encoder
    from mmcif_vec.parse import parse_cif

    try:
        model = nvidia_encoder.load_model(args.ckpt, device=args.device)
    except nvidia_encoder.NotAvailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.max_residues:
        model._mmcif_max_residues = int(args.max_residues)

    import torch

    samples = Path(args.samples)
    files = sorted(samples.rglob("*.cif")) if samples.is_dir() else [samples]

    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    vectors: dict[str, np.ndarray] = {}
    per_chain: dict[str, dict] = {}
    failures: dict[str, str] = {}
    t_all = time.perf_counter()
    for f in files:
        try:
            records = parse_cif(f)
        except Exception as e:
            failures[str(f)] = f"parse failed: {type(e).__name__}: {e}"
            continue
        for rec in records:
            t0 = time.perf_counter()
            try:
                if args.latents:
                    lat = nvidia_encoder.encode_residue_latents(rec.residues, model, device=args.device)
                    vectors[f"{rec.uid}__latents"] = lat
                    vectors[rec.uid] = nvidia_encoder._l2norm(
                        nvidia_encoder.pool_latents(lat, "mean"))
                else:
                    vectors[rec.uid] = nvidia_encoder.encode_chain(
                        rec.residues, model=model, device=args.device)
                per_chain[rec.uid] = {"seconds": time.perf_counter() - t0,
                                      "n_res": len(rec.residues)}
            except Exception as e:  # per-chain failure: record, never crash the run
                failures[rec.uid] = f"{type(e).__name__}: {e}"
                per_chain[rec.uid] = {"seconds": time.perf_counter() - t0,
                                      "n_res": len(rec.residues), "error": failures[rec.uid]}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **vectors)

    peak_mb = None
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    def _version(mod_name):
        try:
            mod = __import__(mod_name)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return None

    sidecar = {
        "encoder": "nvidia la-proteina autoencoder (mean latents, concat mean/max/std, L2)",
        "ckpt": str(args.ckpt),
        "device": str(args.device),
        "total_seconds": time.perf_counter() - t_all,
        "peak_gpu_mem_mb": peak_mb,
        "n_encoded": len(per_chain) - len(failures),
        "n_failed": len(failures),
        "per_chain": per_chain,
        "failures": failures,
        "versions": {"torch": _version("torch"), "numpy": np.__version__,
                     "proteinfoundation": _version("proteinfoundation")},
    }
    sidecar_path = Path(args.sidecar) if args.sidecar else out.with_suffix(".json")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(sidecar, indent=1))
    print(f"wrote {out} ({sidecar['n_encoded']} chains, {len(failures)} failures) + {sidecar_path}")
    for uid, why in failures.items():
        print(f"  FAILED {uid}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
