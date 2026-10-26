#!/usr/bin/env python
import json
from pathlib import Path
import numpy as np
manifest=json.loads(Path("v3/manifest.json").read_text())
want={d["domain_id"] for d in manifest["domains"]}
out={}
for p in sorted(Path("results_v3/.ae2_shards").glob("shard_*.npz")):
    with np.load(p) as z:
        for k in z.files:
            if k.endswith("__latents"): continue
            entry=k.split(":")[0]
            assert entry in want, (p,k)
            assert entry not in out, f"dup {entry}"
            out[entry]=np.asarray(z[k],dtype=np.float32)
    print(p, len(out))
assert set(out)==want, f"{len(out)} vs {len(want)} missing={sorted(want-set(out))[:10]}"
dims={v.shape for v in out.values()}; assert len(dims)==1, dims
np.savez("results_v3/ae2.npz", **out)
print(f"OK ae2.npz {len(out)} dim={dims.pop()}")
# aggregated summary
summ=json.loads(Path("results_v3/.ae2_shards/summary.json").read_text())
agg={"shards":summ["shards"],"total_wall_seconds_shards":sum(s["wall_seconds"] for s in summ["shards"]),
"total_seconds_sidecars":sum(s.get("total_seconds_sidecar") or 0 for s in summ["shards"]),
"max_peak_gpu_mem_mb":max(s.get("peak_gpu_mem_mb") or 0 for s in summ["shards"]),
"total_failures":sum(s.get("n_failed") or 0 for s in summ["shards"]),
"merged_vectors":len(out)}
Path("results_v3/ae2_summary.json").write_text(json.dumps(agg,indent=1))
print(json.dumps(agg,indent=1))
