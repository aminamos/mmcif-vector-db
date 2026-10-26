#!/usr/bin/env python
"""2,000 uniform random cross-entry pairs (seed 7): USalign symmetric TM vs Foldseek gold.
Resumable: pairs already in tm_pairs.tsv are skipped. Parallel x4 via threads+subprocess."""
import csv, json, random, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

REPO = Path("/mnt/e/development/mmcif-vector-db")
USALIGN = "/root/tmtools/USalign"
MANIFEST = REPO/"v3/manifest.json"
DOMAINS = REPO/"v3/domains"
GOLD = REPO/"results_v3/foldseek_gold.tsv"
PAIRS_TSV = REPO/"results_v3/tm_pairs.tsv"
OUT = REPO/"results_v3/tm_validation.json"
N_PAIRS = 2000

manifest = json.loads(MANIFEST.read_text())
doms = manifest["domains"]
ids = sorted(d["domain_id"] for d in doms)
pdb = {d["domain_id"]: d["pdb_id"] for d in doms}

rng = random.Random(7)
pairs = []
seen = set()
while len(pairs) < N_PAIRS:
    a, b = rng.sample(ids, 2)
    if pdb[a] == pdb[b]:
        continue
    key = (a, b) if a < b else (b, a)
    if key in seen:
        continue
    seen.add(key); pairs.append(key)
print(f"sampled {len(pairs)} pairs", flush=True)

# gold lookup
gold = {}
with open(GOLD, newline="") as fh:
    r = csv.DictReader(fh, delimiter="\t")
    for row in r:
        q = Path(row["query"]).stem; t = Path(row["target"]).stem
        if q == t: continue
        key = (q, t) if q < t else (t, q)
        try:
            tm = 0.5*(float(row["qtmscore"])+float(row["ttmscore"])); bits=float(row["bits"])
        except Exception: continue
        if key not in gold or bits > gold[key][1]:
            gold[key] = (tm, bits)
print(f"gold distinct pairs loaded {len(gold)}", flush=True)

# existing results
done = {}
if PAIRS_TSV.exists():
    with open(PAIRS_TSV, newline="") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            try: done[(row["a"], row["b"])] = float(row["usalign_tm"])
            except Exception: pass
print(f"resume: {len(done)} already done", flush=True)

tm_re1 = re.compile(r"TM-score=\s*([0-9.]+).*normalized by length of Structure_1", re.S)
tm_re2 = re.compile(r"TM-score=\s*([0-9.]+).*normalized by length of Structure_2", re.S)

def run_pair(key):
    a, b = key
    fa = str(DOMAINS/f"{a}.cif"); fb = str(DOMAINS/f"{b}.cif")
    try:
        p = subprocess.run([USALIGN, fa, fb], capture_output=True, text=True, timeout=300)
        out = p.stdout + p.stderr
        m1 = tm_re1.search(out); m2 = tm_re2.search(out)
        if not m1 or not m2:
            return key, None, f"parse fail rc={p.returncode}: {out[:300]}"
        return key, 0.5*(float(m1.group(1))+float(m2.group(1))), ""
    except Exception as e:
        return key, None, f"{type(e).__name__}: {e}"

todo = [p for p in pairs if p not in done]
print(f"todo {len(todo)}", flush=True)
failures = {}
t0 = time.time()
fh = open(PAIRS_TSV, "a", newline="")
w = csv.writer(fh, delimiter="\t")
if PAIRS_TSV.stat().st_size == 0:
    w.writerow(["a","b","usalign_tm","error"]); fh.flush()
with ThreadPoolExecutor(max_workers=4) as ex:
    for i, (key, tm, err) in enumerate(ex.map(run_pair, todo)):
        if tm is None:
            failures[str(key)] = err
            w.writerow([key[0], key[1], "", err])
        else:
            done[key] = tm
            w.writerow([key[0], key[1], f"{tm:.6f}", ""])
        if (i+1) % 100 == 0:
            fh.flush(); print(f"{i+1}/{len(todo)} elapsed={time.time()-t0:.0f}s", flush=True)
fh.close()
print(f"USalign done elapsed={time.time()-t0:.0f}s failures={len(failures)}", flush=True)

# compare
xs, ys, present = [], [], 0
for key in pairs:
    if key in gold:
        present += 1
        if key in done:
            xs.append(gold[key][0]); ys.append(done[key])
xs = np.array(xs); ys = np.array(ys)
def pearson(x,y):
    if len(x)<3: return None
    dx=x-x.mean(); dy=y-y.mean(); d=np.sqrt((dx*dx).sum()*(dy*dy).sum())
    return float(dx@dy/d) if d>0 else None
def rankavg(x):
    order=np.argsort(x, kind="mergesort"); ranks=np.empty(len(x)); sx=x[order]
    i=0
    while i<len(x):
        j=i
        while j+1<len(x) and sx[j+1]==sx[i]: j+=1
        ranks[order[i:j+1]]=(i+j)/2+1; i=j+1
    return ranks
res = {
  "n_pairs_sampled": len(pairs), "seed": 7,
  "sampling": "uniform random unordered distinct-domain pairs, cross-entry (pdb_id differs), rng=random.Random(7) sample without pair duplication",
  "usalign_binary": USALIGN,
  "usalign_symmetric_tm": "mean of TM-score normalized by Chain_1 and by Chain_2 from a single USalign run per pair",
  "foldseek_gold_tm": "mean(qtmscore,ttmscore), dedupe by max bits (evaluator convention)",
  "n_present_in_foldseek_tsv": int(present),
  "fraction_present_in_foldseek_tsv": present/len(pairs) if pairs else None,
  "n_compared": int(len(xs)),
  "pearson": pearson(xs,ys),
  "spearman": pearson(rankavg(xs), rankavg(ys)) if len(xs)>=3 else None,
  "n_usalign_failures": len(failures),
  "usalign_failures_sample": dict(list(failures.items())[:20]),
  "wall_seconds_usalign_phase": time.time()-t0,
}
OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
