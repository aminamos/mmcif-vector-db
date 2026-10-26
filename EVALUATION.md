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

## v3 SCOPe benchmark (2026-10-26) — authoritative, supersedes the 47-chain numbers

Artifacts: `results_v3/evaluation.{json,md}` (headline), `results_v3/tm_validation.json`
plus `results_v3/tm_pairs.tsv` (USalign validation), `results_v3/encoder_vs_usalign.json`
(recomputed by `results_v3/tools/encoder_vs_usalign.py`), encoder vectors
`results_v3/{local,ae2,esm_t12}.npz`, harness `v3/evaluate_v3.py`, dataset builder
`scripts/build_v3_dataset.py`, unit set `v3/manifest.json`.

### Dataset

SCOPe 2.01 (`dir.cla.scope.2.01-stable`) domains — defined residue segments, not
whole entries — sampled with seed 42: **2,202 domains / 310 superfamilies / 235
folds / 1,669 PDB entries, lengths 40–511 residues**. Filters were fixed a priori
(in `v3/manifest.json`): single auth chain, length 40–512, superfamilies with
≥4 eligible domains, cap 8 per superfamily, standard amino acids only. Sliced
CIFs live in `v3/domains/` (~202 MB, gitignored, regenerable via
`scripts/build_v3_dataset.py`; the builder is resumable — existing sliced files
are re-validated and reused — and deterministic: same seed + cache ⇒ same
manifest domain list).

### Protocol (7 points, as implemented in `v3/evaluate_v3.py`)

1. **Unit set fixed a priori** by the manifest before any encoding or gold
   computation; every unordered pair of distinct domains is a potential pair.
2. **Gold is exhaustive**: Foldseek `easy-search --exhaustive-search` all-vs-all
   over the 2,202 sliced domains.
3. **100% coverage, nothing imputed**: 2,423,301/2,423,301 pairs covered
   (headline cross-entry subset: 2,422,451/2,422,451). Missing gold would be
   excluded and only reported as coverage — in the event, there is none.
4. **Symmetrisation**: gold TM = mean(`qtmscore`, `ttmscore`); duplicate rows
   for an unordered pair de-duplicated keeping max `bits`.
5. **Pearson and Spearman** both reported (numpy-only implementations, average
   ranks for ties).
6. **Protein-bootstrap CIs**: 95% percentile CIs from 1,000 resamples over
   domains (not pairs), seed 42.
7. **Headline = cross-entry pairs only** (domains sharing a PDB entry excluded);
   the including-same-entry analysis is reported as a sensitivity check.

Gold distribution (headline): mean symmetric TM 0.110, TM ≥ 0.7 fraction 0.17%.
Labels for retrieval/classification are same-superfamily (6,190 positive vs
2,416,261 negative headline pairs); retrieval is leave-entry-out top-1 over
2,139 queries (63 queries excluded for having no cross-entry same-superfamily
candidate, never scored as misses).

### Headline results (cross-entry, vs Foldseek gold)

| Encoder | Dim | Pearson [95% CI] | Spearman [95% CI] | top-1 | AUPR | AUROC |
|---|---:|---|---|---:|---:|---:|
| Local `combined_vector` | 768 | 0.159 [0.138, 0.181] | 0.120 [0.090, 0.150] | 0.771 | 0.468 | 0.906 |
| NVIDIA La-Proteina AE2, mean-only | 8 | 0.104 [0.095, 0.114] | 0.084 [0.074, 0.096] | 0.305 | 0.103 | 0.813 |
| ESM2-t12 35M, mean-pooled | 480 | 0.087 [0.068, 0.107] | 0.084 [0.064, 0.104] | 0.892 | 0.498 | 0.897 |

Sensitivity (same-entry pairs included, 2,423,301 pairs): local Pearson 0.164 /
Spearman 0.121; AE2 0.108 / 0.085; ESM 0.091 / 0.084 — conclusions unchanged.

### Gold validation caveat (USalign)

Foldseek TM is a proxy. On 2,000 uniform random cross-entry pairs (seed 7),
USalign (compiled from Zhang-group source after the prebuilt binary 404'd;
symmetric TM = mean of the two length-normalised scores) agrees with Foldseek
gold at only **Pearson 0.478 / Spearman 0.304**. Foldseek TM is therefore a
noisy proxy on this mostly-unrelated population, and that noise attenuates
every encoder-vs-gold correlation above; the encoder *rankings* are what the
headline table supports, not the absolute r values as measures of encoder
quality. USalign TM on these random pairs: mean 0.280, median 0.278,
TM ≥ 0.7 in only 0.35% — the correlation task is dominated by unrelated pairs,
which is why all correlations are low while retrieval (below) tells a
different story.

Encoder cosine vs USalign on the same 2,000 pairs (independently recomputed,
`results_v3/encoder_vs_usalign.json`), with vs-Foldseek on identical pairs for
comparison:

| Encoder | vs USalign Pearson | vs USalign Spearman | vs Foldseek Pearson | vs Foldseek Spearman |
|---|---:|---:|---:|---:|
| Local | 0.166 | 0.138 | 0.161 | 0.123 |
| AE2 (8-d) | 0.050 | 0.038 | 0.075 | 0.061 |
| ESM2-t12 | 0.061 | 0.049 | 0.071 | 0.053 |

The ordering (local > ESM > AE2 on TM correlation) is the same against the
independent USalign gold as against Foldseek.

### Runtime, storage, resumability

All runs local on one RTX 4070 SUPER @ 120 W — no rented GPUs, no new GPU
error events. Local encoder: 17 s on CPU for all 2,202 domains. AE2: 321 s
wall sharded (129 s encode-only, 2.1 GB peak GPU). ESM2-t12: 29 s (534 MB
peak). Storage: sliced domains ~202 MB and the 2.6 GB entry cache
(`v3/.cache_entries/`) are gitignored and deletable/regenerable; the 270 MB
Foldseek gold TSV and Foldseek DB/temp dirs are likewise gitignored
(regenerable via `results_v3/tools/run_foldseek.sh`). Committed artifacts are
small: NPZs 7.0 MB (local) / 4.6 MB (ESM) / 0.6 MB (AE2), manifest 1.3 MB,
everything else ≤ 53 KB. The dataset builder, AE2 sharding, and USalign
validation are all resumable.

### Retraction of the v1-era correlations

The 47-chain hits-only correlations reported during the v1/NVIDIA evaluation
(local 0.72, AE2 0.49, ESM 0.40) were **range-restricted estimates**: they were
computed only over pairs Foldseek already reported as hits on a 47-chain set,
which inflates correlation by construction. They are **retracted and
superseded** by the v3 numbers above (exhaustive pairs, 100% coverage, no
hit-selection). The v1 top-1 retrieval tables elsewhere in this document stand
as small-set retrieval results, but no correlation claim should be quoted
from the 47-chain era.

### Verdict

**The local encoder remains the default**: best TM correlation against both
gold standards, best AUROC (0.906), competitive AUPR, zero dependencies, 17 s
on CPU. **ESM2-t12 is the retrieval winner** (top-1 0.892, best AUPR 0.498)
and is the named optional-upgrade candidate if retrieval quality becomes the
bottleneck — an optional pretrained sequence encoder, adopted only as a
separate proposal. **La-Proteina AE2 is rejected decisively at scale**: worst
retrieval (top-1 0.305, AUPR 0.103) and worst-or-tied-worst correlation of the
three, with by far the heaviest stack (4.1 GB checkpoints, WSL2/CUDA venv,
321 s sharded encode). The small-set tie that kept AE2 alive in the v1
evaluation does not survive a 2,202-domain benchmark.
