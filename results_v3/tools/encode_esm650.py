#!/usr/bin/env python
"""ESM2-t33 650M (facebook/esm2_t33_650M_UR50D) mean-pooled residue embeddings.

Incremental: flush NPZ every 100 new domains; skip domain_ids already present.
Sequences from v3 manifest. Mean-pool last_hidden_state over residue tokens
only (exclude BOS/EOS/pad), L2-normalise, store float32 (1280-d).
GPU safety: fp16 autocast + inference_mode + TF32, length-sorted batching.
"""
import argparse, json, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sidecar", required=True)
ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
ap.add_argument("--batch-size", type=int, default=16)
args = ap.parse_args()

import torch
from transformers import AutoModel, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

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
# length-sorted (desc) so padded waste + peak activation are predictable
todo.sort(key=lambda d: len(seqs[d]), reverse=True)
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
    tmp = outp.with_suffix(".tmp.npz")
    np.savez(tmp, **vecs)
    tmp.replace(outp)

since_flush = 0
with torch.inference_mode():
    for start in range(0, len(todo), B):
        batch_ids = todo[start:start + B]
        batch_seqs = [seqs[d] for d in batch_ids]
        try:
            enc = tok(batch_seqs, return_tensors="pt", padding=True, truncation=False).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out_h = model(**enc).last_hidden_state  # (B, L, D) incl BOS/EOS+pad
            attn = enc["attention_mask"]
            mask = attn.clone()
            mask[:, 0] = 0  # BOS
            lengths = attn.sum(1)
            for i in range(len(batch_ids)):
                mask[i, lengths[i] - 1] = 0  # EOS
            m = mask.unsqueeze(-1).to(out_h.dtype)
            pooled = ((out_h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.where(norms > 0, norms, 1.0)
            for d, v in zip(batch_ids, pooled):
                vecs[d] = v.astype(np.float32)
            since_flush += len(batch_ids)
        except Exception as e:
            for d in batch_ids:
                failures[d] = f"{type(e).__name__}: {e}"
            print(f"FAILED batch at {start}: {e}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
        if since_flush >= 100 or start + B >= len(todo):
            flush()
            since_flush = 0
            print(f"progress {len(vecs)}/{len(order)} elapsed={time.perf_counter()-t_all:.1f}s", flush=True)
flush()
peak = None
if device == "cuda":
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
sidecar = {
    "encoder": f"ESM2 t33 650M ({args.model}) mean-pooled residue last_hidden_state, L2-normalised",
    "model": args.model, "device": device, "batch_size": B,
    "dtype": "float16 autocast (inference_mode, TF32 on); stored float32 L2-normalised",
    "total_seconds": time.perf_counter() - t_all,
    "wall_time_s": time.perf_counter() - t_all,
    "peak_gpu_mem_mb": peak,
    "peak_vram_mb": peak,
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
print("ESM650 DONE")
