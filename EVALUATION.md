# Evaluation — does it work?

Sample set (`samples/`, 13 entries, 30 chains): related families MEF2
(1EGW/1TQE/1N6J), Foxk1 (2A3S/2D2W + AFDB AF-P42128-F1), globin (1A3N/1HHO),
plus unrelated 1LYZ, 1EMA, 1UBQ, 4INS, 6LU7. Embedded locally on an M3
MacBook Air: 30 chains in ~1s, no GPU/API.

## Retrieval (similar: sequence+structure, reranked) — works

Leave-self-out, top-1 must be same family:

- **21/22 family queries correct.** 2A3S->2D2W (0.996), MEF2 chains -> MEF2,
  globin alpha->alpha and beta->beta across 1A3N/1HHO.
- Only miss: AF-P42128-F1 (full-length 719-res AFDB model) vs its 101-res
  PDB domain — length mismatch, known v1 limitation.

## Clustering (k-means k=4 over combined vector) — coarse only

- Purity 0.77. Insulin chains cluster cleanly; MEF2 cluster absorbs GFP
  (1EMA) and ubiquitin (1UBQ); globin alpha chains cluster with Foxk1.
- Reweighting sequence vs structure (0.2–1.0 swept) did not fix this:
  small all-alpha proteins have genuinely similar CA-distance histograms.
- Verdict: use retrieval/rerank for finding relatives; do not trust
  k=4 clusters as families. A learned encoder (NVIDIA La-Proteina /
  RFD4-Proteina latents, Jev-style reranking) is the upgrade path.

## Unrelated discrimination — weak scores, correct ordering

An unrelated query (1LYZ) still gets top scores ~0.87 against MEF2.
Cosine scores are uncalibrated: trust rank order within a query,
not absolute thresholds across queries.

Reproduce: `uv run mmcif-vec --db sample.db ingest samples/` then
`similar` / `cluster` as in README.

## NVIDIA La-Proteina evaluation (2026-10-26) — rejected

Full run record: `results/BENCHMARK_RUN.md`. Draft/spec:
`docs/nvidia-autoencoder-draft.md`.

### Protocol recap

- Encoders compared behind the same `chain -> vector` contract on
  identical inputs: `samples/` (13 entries, 30 chains, 22 family
  queries, 8 singletons) and `samples_hard/` (12 entries, 17 chains,
  17 family queries, 0 singletons; same-fold families with low pairwise
  sequence identity — see `samples_hard/manifest.json`).
- Metric: leave-entry-out top-1 family accuracy (primary; other chains
  from the query's entry excluded) and leave-self-out top-1
  (secondary), cosine over each dataset's candidate pool via
  `scripts/benchmark_encoders.py`.
- NVIDIA vectors: per-residue 8-d deterministic posterior `mean`
  latents from the La-Proteina autoencoder, pooled per chain
  (pooling ablation below), L2-normalised. Controls: ESM2-t12 35M
  mean-pooled residue embeddings and Foldseek `easy-search`
  (bits ranking), on the same sets.

### Results — samples/ (22 family queries)

| Encoder | Dim | Leave-entry-out | Leave-self-out | Encode wall | Peak GPU | Failures |
|---|---:|---:|---:|---:|---:|---|
| Local `combined_vector` | 768 | 21/22 | 21/22 | 0.04 s (CPU) | — | none |
| NVIDIA AE1 (`AE1_ucond_512`) mean+max+std | 24 | 9/21 | 9/21 | 2.12 s | 1.4 GB | AF-P42128-F1:A: 719 res > AE1's 512 limit (recorded, not truncated) |
| NVIDIA AE2 (`AE2_ucond_800`) mean+max+std | 24 | 10/22 | 10/22 | 1.69 s | 3.2 GB | none |
| NVIDIA AE2 mean-only (ablation) | 8 | 19/22 | 19/22 | derived | — | none |
| NVIDIA AE2 mean+max (ablation) | 16 | 9/22 | 9/22 | derived | — | none |
| ESM2-t12 35M mean-pooled (control) | 480 | 22/22 | 22/22 | 0.79 s | 232 MB | none |
| Foldseek easy-search (control) | — | 19/22 | 19/22 | ~1.5 s search (CPU) | — | 3 short MEF2 peptide chains (1N6J:E, 1TQE:G, 1TQE:J) returned no hit, counted as misses |

### Results — samples_hard/ (17 family queries)

| Encoder | Dim | Leave-entry-out | Leave-self-out | Encode wall | Peak GPU | Failures |
|---|---:|---:|---:|---:|---:|---|
| Local `combined_vector` | 768 | 16/17 | 17/17 | 0.04 s (CPU) | — | none |
| NVIDIA AE1 mean+max+std | 24 | 6/17 | 6/17 | 1.25 s | 1.2 GB | none (all hard chains ≤512 res) |
| NVIDIA AE2 mean+max+std | 24 | 8/17 | 8/17 | 1.37 s | 1.2 GB | none |
| NVIDIA AE2 mean-only (ablation) | 8 | 16/17 | 17/17 | derived | — | none |
| NVIDIA AE2 mean+max (ablation) | 16 | 8/17 | 8/17 | derived | — | none |
| ESM2-t12 35M mean-pooled (control) | 480 | 17/17 | 17/17 | 0.72 s | 151 MB | none |
| Foldseek easy-search (control) | — | 17/17 | 17/17 | ~1.5 s search (CPU) | — | none |

Encode wall excludes ~20 s checkpoint load/imports (observed end-to-end
WSL wall incl. load: AE1 samples 28.6 s, AE2 samples 22.2 s, AE1 hard
22.4 s, AE2 hard 22.2 s). Artifacts: `results/*.npz` vectors (with
per-residue `__latents` for the NVIDIA runs), `results/benchmark_*.{json,txt}`
per-encoder scores, `results/foldseek_allvall.tsv` pairwise control table.

### Pooling ablation conclusion

Pooling decides the NVIDIA result. On AE2 latents: mean+max+std (24-d,
the originally shipped pooling) scores 10/22 + 8/17, mean+max (16-d)
9/22 + 8/17, and **mean-only (8-d) 19/22 + 16/17** (sole hard miss
1GFL:A → 1A5R:A). Max/std pooling destroys most of the signal; the plain
per-residue mean is the only validated aggregation, so `mean` is now
the default in `mmcif_vec/nvidia_encoder.py` (`FIXED_DIM = 8`, other
poolings retained in `POOLINGS` for benchmarks). Latents are always the
deterministic posterior `mean` — never the stochastic `z_latent` sample.

### Environment

- GPU: NVIDIA RTX 4070 SUPER 12 GB, power-capped at 120 W; driver stack
  under WSL2 Ubuntu-22.04; torch 2.7.0+cu118, transformers 4.48.3.
  No new GPU error events were recorded during the runs.
- Checkpoints: `AE1_ucond_512.ckpt` (≤512 residues) and
  `AE2_ucond_800.ckpt` (≤896 residues) from Hugging Face
  `nvidia/NV-La-Proteina-Ucond-v1` (ungated, NVIDIA Open Model
  License). 130M-parameter encoder; checkpoints total ~4.1 GB.
- The heavy stack lives in a WSL2/Linux CUDA venv outside this repo;
  this repo's `uv` env stays CPU-only and torch-free. Chains over a
  checkpoint's residue limit are rejected with a clear error, never
  truncated.

### Verdict vs the draft's acceptance criteria

Acceptance required: strictly better than the local encoder on the hard
set **and** not worse on the bundled set. NVIDIA meets **neither** with
any pooling: the best variant (AE2 mean-only, 8-d) ties the local
encoder on hard leave-entry-out (16/17 vs 16/17) and loses on the
bundled set (19/22 vs 21/22); the 24-d pooling loses everywhere
(10/22, 8/17). **Verdict: REJECT the swap. The local encoder stays the
default**; the NVIDIA encoder remains optional code behind the same
contract. ESM2-t12 (22/22 + 17/17 at 151–232 MB GPU) is the cheap
upgrade path if retrieval quality ever becomes the bottleneck — as a
separate proposal for an optional pretrained sequence encoder, not a
reason to adopt La-Proteina or to train our own on a 47-chain benchmark.
