#!/usr/bin/env python
"""ESM2-t12 (facebook/esm2_t12_35M_UR50D) mean-pooled residue embeddings.
Incremental: flush NPZ every 100 new domains; skip domain_ids already present.
Sequences taken from v3 manifest (authoritative domain sequences)."""
import argparse, json, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sidecar", required=True)
ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
ap.add_argument("--batch-size", type=int, default=8)
args = ap.parse_args()

import torch
from transformers import AutoModel, AutoTokenizer

t_all = time.perf_counter()
manifest = json.loads(Path(args.manifest).read_text())
domains = manifest["domains"]
seqs = {d["domain_id"]: d["sequence"] for d in domains}
order = [d["domain_id"] for d in domains]

vecs = {}
outp = Path(args.out)
if outp.exists():
    with np.load(outp, allow_pickle=False) as z:
        vecs = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
    print(f"resume: {len(vecs)} already present", flush=True)

todo = [d for d in order if d not in vecs]
print(f"todo {len(todo)} / {len(order)}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModel.from_pretrained(args.model).to(device).eval()
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

failures = {}
B = args.batch_size

def flush():
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, **vecs)

with torch.no_grad():
    for start in range(0, len(todo), B):
        batch_ids = todo[start:start+B]
        batch_seqs = [seqs[d] for d in batch_ids]
        try:
            enc = tok(batch_seqs, return_tensors="pt", padding=True, truncation=False).to(device)
            out_h = model(**enc).last_hidden_state  # (B, L, D) includes BOS/EOS + pad
            attn = enc["attention_mask"]
            # exclude BOS (pos0) and EOS: real token mask minus specials
            # ESM tokenizer: pad token masked by attn; set first token and last non-pad token to 0
            mask = attn.clone()
            mask[:, 0] = 0
            lengths = attn.sum(1)  # includes BOS+EOS
            for i in range(len(batch_ids)):
                mask[i, lengths[i]-1] = 0  # EOS
            m = mask.unsqueeze(-1).float()
            summed = (out_h * m).sum(1)
            counts = m.sum(1).clamp(min=1)
            pooled = (summed / counts).cpu().numpy().astype(np.float32)
            # L2 normalise to match other encoders' cosine-ready form? Keep raw mean then L2 (cosine in evaluator normalises anyway; store L2 for consistency)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.where(norms > 0, norms, 1.0)
            for d, v in zip(batch_ids, pooled):
                vecs[d] = v.astype(np.float32)
        except Exception as e:
            for d in batch_ids:
                failures[d] = f"{type(e).__name__}: {e}"
            print(f"FAILED batch at {start}: {e}", flush=True)
        if (start // B) % 13 == 0 or start + B >= len(todo):
            pass
        done_new = start + len(batch_ids)
        if done_new % 100 < B or done_new >= len(todo):
            flush()
            print(f"progress {len(vecs)}/{len(order)} elapsed={time.perf_counter()-t_all:.1f}s", flush=True)
flush()
peak = None
if device == "cuda":
    peak = torch.cuda.max_memory_allocated() / (1024*1024)
sidecar = {
    "encoder": f"ESM2 t12 ({args.model}) mean-pooled residue last_hidden_state, L2-normalised",
    "model": args.model, "device": device, "batch_size": B,
    "total_seconds": time.perf_counter() - t_all,
    "peak_gpu_mem_mb": peak,
    "n_encoded": len(vecs), "n_expected": len(order),
    "failures": failures,
    "dim": int(next(iter(vecs.values())).shape[0]) if vecs else None,
    "versions": {"torch": torch.__version__},
}
try:
    import transformers; sidecar["versions"]["transformers"] = transformers.__version__
except Exception: pass
Path(args.sidecar).write_text(json.dumps(sidecar, indent=1))
print(json.dumps(sidecar, indent=1))
assert len(vecs) == len(order), f"{len(vecs)} != {len(order)}"
assert set(vecs) == set(order)
print("ESM DONE")
