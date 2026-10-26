#!/usr/bin/env bash
# AE2 sharded encode, resumable: skip shard if NPZ validates.
set -u
REPO=/mnt/e/development/mmcif-vector-db
LAP=/mnt/e/development/laproteina-eval/la-proteina
CKPT=/mnt/e/development/laproteina-eval/checkpoints_laproteina/AE2_ucond_800.ckpt
PY=/root/laproteina-venv/bin/python
export PYTHONPATH="$LAP:$REPO"
cd "$REPO"
SUMMARY=results_v3/.ae2_shards/summary.json
$PY - "$SUMMARY" <<'PYEOF'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
if not p.exists(): p.write_text(json.dumps({"shards":[]},indent=1))
PYEOF
for shard in $(ls -d v3/.shards/shard_* | sort); do
  name=$(basename "$shard")
  out="results_v3/.ae2_shards/${name}.npz"
  side="results_v3/.ae2_shards/${name}.json"
  expected=$(ls "$shard"/*.cif | wc -l)
  # validate existing
  if [ -f "$out" ]; then
    ok=$($PY - "$out" "$expected" <<'PYEOF'
import numpy as np,sys
try:
    with np.load(sys.argv[1]) as z:
        n=len([k for k in z.files if not k.endswith("__latents")])
    print("1" if n==int(sys.argv[2]) else "0")
except Exception as e:
    print("0")
PYEOF
)
    if [ "$ok" = "1" ]; then echo "SKIP $name (valid, $expected)"; continue; fi
    echo "RE-RUN $name (existing NPZ invalid)"
  fi
  echo "=== $name expected=$expected $(date -Iseconds) ==="
  start=$(date +%s)
  $PY scripts/encode_nvidia.py --samples "$shard" --ckpt "$CKPT" --out "$out" --sidecar "$side" > "results_v3/.ae2_shards/${name}.log" 2>&1
  rc=$?
  end=$(date +%s)
  wall=$((end-start))
  cat "results_v3/.ae2_shards/${name}.log"
  $PY - "$SUMMARY" "$name" "$wall" "$rc" "$side" "$out" <<'PYEOF'
import json,sys
from pathlib import Path
import numpy as np
summ_p, name, wall, rc, side, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6])
data=json.loads(Path(summ_p).read_text()) if Path(summ_p).exists() else {"shards":[]}
entry={"shard":name,"wall_seconds":wall,"returncode":rc}
if side.exists():
    s=json.loads(side.read_text())
    entry.update({"n_encoded":s.get("n_encoded"),"n_failed":s.get("n_failed"),"failures":s.get("failures"),"peak_gpu_mem_mb":s.get("peak_gpu_mem_mb"),"total_seconds_sidecar":s.get("total_seconds")})
if out.exists():
    with np.load(out) as z: entry["npz_keys"]=len(z.files)
data["shards"]=[e for e in data["shards"] if e["shard"]!=name]+[entry]
Path(summ_p).write_text(json.dumps(data,indent=1))
print(entry)
PYEOF
done
echo "AE2 shards done $(date -Iseconds)"
