#!/usr/bin/env python
"""Re-key ENTRY:chain NPZ to manifest domain_ids (entry part, asserted)."""
import argparse, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

manifest = json.loads(Path(args.manifest).read_text())
want = {d["domain_id"] for d in manifest["domains"]}
out = {}
with np.load(args.inp, allow_pickle=False) as z:
    keys = list(z.files)
    for k in keys:
        if k.endswith("__latents"):
            continue
        entry = k.split(":")[0]
        assert entry in want, f"key {k!r} entry {entry!r} not in manifest"
        assert entry not in out, f"duplicate domain after rekey: {entry} from {k}"
        out[entry] = np.asarray(z[k], dtype=np.float32)
assert set(out) == want, f"key mismatch: got {len(out)} want {len(want)} missing={sorted(want-set(out))[:10]} extra={sorted(set(out)-want)[:10]}"
dims = {v.shape for v in out.values()}
assert len(dims) == 1, dims
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, **out)
print(f"OK {args.out}: {len(out)} vectors dim={dims.pop()} keys==manifest")
