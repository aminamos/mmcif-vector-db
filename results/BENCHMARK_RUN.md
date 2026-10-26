# NVIDIA La-Proteina Benchmark Run — 2026-10-26

GPU pre-flight: RTX 4070 SUPER power limit verified **120.00 W** before any GPU work (already applied).
External watchdog `train_watchdog.ps1` started as Windows PID **64164**, stopped at teardown.
Note: the watchdog polls `nvidia-smi` and cannot kill WSL python by name — the 120W cap is the real guard.
No other GPU compute process was running (only Brave/dwm graphics processes, present before and after).

Protocol: `scripts/benchmark_encoders.py` — leave-ENTRY-out top-1 (primary), leave-self-out (secondary),
cosine over the NPZ candidate pool of that dataset. NVIDIA vectors: 24-d `concat(mean, max, std)` over
per-residue 8-d latent `mean`, L2-normalised, `--latents` stored in the same NPZ.
Encode wall = sidecar `total_seconds` (encode loop, excl. ~20 s checkpoint load/imports; observed
end-to-end WSL wall incl. load: AE1 samples 28.6 s, AE2 samples 22.2 s, AE1 hard 22.4 s, AE2 hard 22.2 s).

## samples/ (13 entries, 30 chains, 22 family queries, 8 singletons)

| encoder | dim | leave-entry-out top-1 | leave-self-out | overall incl singletons | total encode wall (s) | peak GPU MB | failures |
|---|---|---|---|---|---|---|---|
| local combined_vector | 768 | 21/22 = 0.955 | 21/22 = 0.955 | 0.700 | 0.04 | — (CPU) | none |
| nvidia AE1 (`AE1_ucond_512.ckpt`) | 24 | 9/21 = 0.429 | 9/21 = 0.429 | 0.310 | 2.12 | 1397.6 | AF-P42128-F1:A: 719 residues > 512 AE1 limit (recorded, not truncated) |
| nvidia AE2 (`AE2_ucond_800.ckpt`) | 24 | 10/22 = 0.455 | 10/22 = 0.455 | 0.333 | 1.69 | 3248.8 | none |
| nvidia AE2 mean-only (ablation, from stored latents) | 8 | 19/22 = 0.864 | 19/22 = 0.864 | 0.633 | — (derived) | — | none |
| nvidia AE2 mean_max (ablation, from stored latents) | 16 | 9/22 = 0.409 | 9/22 = 0.409 | 0.300 | — (derived) | — | none |
| ESM2 t12 35M mean-pooled (control) | 480 | 22/22 = 1.000 | 22/22 = 1.000 | 0.733 | 0.79 | 232.4 | none |
| Foldseek easy-search (control, bits ranking) | — | 19/22 = 0.864 | 19/22 = 0.864 | 0.633 | ~1.5 (search) | — (CPU) | 3 short MEF2 peptide chains (1N6J:E, 1TQE:G, 1TQE:J) returned no Foldseek hit, counted as misses |

AE2 is the clean single-checkpoint like-for-like on samples (all 30 chains). AE1's denominator is 21
because AF-P42128-F1:A could not be encoded.

## samples_hard/ (12 entries, 17 chains, 17 family queries, 0 singletons)

| encoder | dim | leave-entry-out top-1 | leave-self-out | overall incl singletons | total encode wall (s) | peak GPU MB | failures |
|---|---|---|---|---|---|---|---|
| local combined_vector | 768 | 16/17 = 0.941 | 17/17 = 1.000 | 0.941 | 0.04 | — (CPU) | none |
| nvidia AE1 | 24 | 6/17 = 0.353 | 6/17 = 0.353 | 0.353 | 1.25 | 1218.9 | none (all hard chains ≤512 res) |
| nvidia AE2 | 24 | 8/17 = 0.471 | 8/17 = 0.471 | 0.471 | 1.37 | 1218.9 | none |
| Foldseek easy-search (control) | — | 17/17 = 1.000 | 17/17 = 1.000 | 1.000 | ~1.5 (search, combined run) | — (CPU) | none |

## Pooling ablation (samples, AE2 latents)

`mean_max_std` (the shipped 24-d pooling) = 10/22, `mean_max` (16-d) = 9/22, **mean-only (8-d) = 19/22**.
The pooling choice matters decisively: max/std pooling destroys most of the signal; the plain
per-residue mean is nearly as good as Foldseek (19/22) and far above the shipped pooling.
(Files: `results/nvidia_ae2_mean8.npz`, `results/nvidia_ae2_meanmax16.npz`.)

## Controls

- **Foldseek** (static Linux build 463739e, `foldseek-linux-avx2.tar.gz`): `easy-search` all-vs-all over
  samples + samples_hard structures, `--format-output query,target,alntmscore,bits`. Hits were mapped
  auth-chain → label-chain via the repo parser, NMR models deduplicated by max bits, self/same-entry
  excluded exactly as leave-entry-out. Pairwise table: `results/foldseek_allvall.tsv`
  (raw: `results/foldseek_allvall_raw.tsv`); numbers: `results/foldseek_summary.json`.
  Samples 19/22, hard 17/17, combined-pool 28/39.
- **ESM**: `transformers==4.48.3` was already installed in the WSL venv (per ENV.md), so **no pip install
  was needed and torch 2.7.0+cu118 was untouched**. One attempt, `facebook/esm2_t12_35M_UR50D`,
  mean-pooled residue embeddings on GPU: **22/22** on samples (`results/esm_t12.npz`).

## Teardown / GPU health

- Watchdog PID 64164 stopped. No GPU compute process remains (only pre-existing Brave/dwm graphics).
  Power limit still 120.00 W.
- `check_gpu_events.ps1` output (verbatim verdict): **"FAIL: 11 suspect event(s). Training on this box is NOT stable yet."**
  All 11 events pre-date this run (latest 2026-10-26 12:21 nvlddmkm Id 14; the rest are Sept–Oct
  Kernel-Power Id 41 / nvlddmkm from before the 120W-cap session at 15:40–15:42). No new GPU event
  was recorded during or after this benchmark session.

## Bugs fixed during the run

No repo driver/harness bugs blocked the run; `scripts/encode_nvidia.py` and
`scripts/benchmark_encoders.py` were used unmodified. Two throwaway-script bugs of my own
(both in temporary files, deleted after use):
1. Pooling-ablation snippet sliced `"__latents"` as 10 chars instead of 9, collapsing chain UIDs
   to 13 entries (0 family queries); fixed to `k[:-9]`, yielding all 30 chains.
2. Foldseek evaluator initially mapped Foldseek IDs naively (`ENTRY_CHAIN` label-style); Foldseek
   emits auth chain IDs (and bare IDs for single-chain files, `_MODEL_n` for NMR ensembles).
   Fixed by mapping auth→label chains through `mmcif_vec.parse` records and deduplicating
   NMR models by max bits; also restricted the candidate pool per dataset to match the
   encoder harness (an unrestricted combined pool understated hard-set Foldseek as 9/17;
   the like-for-like within-hard number is 17/17).

## Addendum — hard-set completion (2026-10-26, later session)

Completes the two gaps above: mean-only pooling on the hard set and ESM2-t12 on the hard set.
Prior numbers above are unchanged. GPU pre-flight re-verified: power limit **120.00 W**, no
concurrent GPU compute process (only pre-existing Brave/dwm graphics).

| encoder | dim | leave-entry-out top-1 | leave-self-out | overall incl singletons | total encode wall (s) | peak GPU MB | failures |
|---|---|---|---|---|---|---|---|
| nvidia AE2 mean-only (ablation, from stored hard latents) | 8 | 16/17 = 0.941 | 17/17 = 1.000 | 0.941 | — (derived) | — | none |
| nvidia AE2 mean_max (ablation, from stored hard latents) | 16 | 8/17 = 0.471 | 8/17 = 0.471 | 0.471 | — (derived) | — | none |
| ESM2 t12 35M mean-pooled (control) | 480 | 17/17 = 1.000 | 17/17 = 1.000 | 1.000 | 0.72 | 151.4 | none |

- **AE2 ablations**: `results/nvidia_ae2_hard.npz` already contained all 17 `__latents` keys, so no
  re-encode was needed. Mean-only (8-d) and mean_max (16-d) vectors were derived from those latents
  exactly as for samples (per-residue mean / concat(mean, max), L2-normalised; full concat(mean, max,
  std) reproduction of the stored 24-d vectors checked, max diff 1.5e-08). Files:
  `results/nvidia_ae2_hard_mean8.npz`, `results/nvidia_ae2_hard_meanmax16.npz`; benchmarks
  `results/benchmark_nvidia_ae2_hard_mean8.{json,txt}`, `results/benchmark_nvidia_ae2_hard_meanmax16.{json,txt}`.
  The samples pooling conclusion holds on the hard set: mean-only (16/17, sole miss 1GFL:A → 1A5R:A)
  nearly matches Foldseek/ESM, while mean_max (8/17, same 9 misses as the 24-d pooling) does not.
- **ESM hard**: `samples_hard` chains encoded with `facebook/esm2_t12_35M_UR50D`, mean-pooled residue
  embeddings (last hidden state, BOS/EOS excluded), L2-normalised, on GPU in the WSL venv
  (torch 2.7.0+cu118, transformers 4.48.3, `/root/laproteina-venv`). Before the hard run the same
  method re-encoded `samples/` and reproduced `results/esm_t12.npz` (max abs diff 1.2e-07, same
  232.4 MB peak), confirming it is the method that produced the samples control. Files:
  `results/esm_t12_hard.npz` + sidecar `results/esm_t12_hard.json`; benchmark
  `results/benchmark_esm_t12_hard.{json,txt}`. ESM matches Foldseek at 17/17 on the hard set.
