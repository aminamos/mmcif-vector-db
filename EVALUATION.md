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

## Vector store (sqlite-vec + optional LanceDB ANN) — 2026-10-27

Artifacts: `results_store/benchmark_store.{md,json}` (exact store),
`results_store/benchmark_ann.{md,json}` (ANN), harnesses
`scripts/benchmark_store.py` / `scripts/benchmark_ann.py`. The large bench
DBs and ANN indexes under `results_store/.bench_dbs/` and
`results_store/.ann_indexes/` are gitignored and regenerable.

### Architecture

- One SQLite file. The `chains` table keeps chain metadata plus the four
  vectors as float32 BLOBs (`seq`/`struct`/`text` 256-d, `combined` 768-d);
  four sqlite-vec `vec0` tables (`vec_combined` 768, `vec_seq`/`vec_struct`/
  `vec_text` 256) keyed by `uid` mirror them for KNN candidate generation.
  `sqlite-vec>=0.1` is a core dependency; the extension loads at DB open
  with a clear error if unavailable.
- Search is unchanged in result semantics: vec0 KNN generates a candidate
  pool, then candidates are reranked with the identical exact block
  cosines (mean of seq/struct/text scores) the old brute-force scan used,
  and result scores are those exact rerank scores.
- Old pre-vec0 databases migrate on open: the vec0 tables are created and
  backfilled from the existing `chains` BLOBs in chunked transactions, so
  existing DBs open and search without re-ingest.
- Writes: the connection is autocommit; `DB.transaction()` is the explicit
  unit of work. `upsert_chain` joins the ambient transaction (no per-call
  commit), and `ingest` wraps a whole file in one transaction.

### Exact-store benchmark

Methodology caveat: corpora above the real set are **synthetic** — real v3
combined vectors plus Gaussian jitter (σ=0.05), renormalised. This measures
store mechanics (insert, KNN scan, rerank, on-disk size), not encoder
quality. 50 fixed queries, k=10, per size.

| Size | Insert (vec/s) | Search p50 | Search p95 | Top-10 agreement vs brute force |
|---|---:|---:|---:|---:|
| real-2202 | 1,480 | 6.0 ms | 6.2 ms | 1.000 |
| 10k | 1,414 | 17.5 ms | 19.6 ms | 1.000 |
| 100k | 1,335 | 176 ms | 182 ms | 0.998 |
| 500k | 1,297 | 871 ms | 882 ms | 0.998 |
| 2M | 1,280 | 3,538 ms | 3,616 ms | 0.994 |

The 2M row was **measured**, not projected (~26 min insert at 1,280 vec/s;
DB 29,199 MB). Agreement below 1.0 is exclusively adjacent near-tie swaps
(rerank deltas ≤5e-5; identical at the 4-decimal precision search reports).

Insert batching fix: the original per-row-commit upsert measured
~230–297 vec/s. Wrapping ingest in one transaction per file (chunked
transactions for migration/backfill) raised this to ~1.28–1.48k vec/s,
essentially flat to 2M with no throughput cliff.

Honesty note: **vec0 KNN here is a linear scan.** Exact search latency
scales linearly with corpus size, and a chunked NumPy brute force is ~10×
faster at large sizes (338 ms vs 3,538 ms at 2M). vec0 is kept for
exactness (bit-identical rerank semantics) and single-file simplicity —
not for speed. That is the trade the ANN path below exists to address.

### ANN (LanceDB, optional `ann` extra)

LanceDB is an optional dependency (`uv sync --extra ann`, resolved to
LanceDB 0.40.0), imported lazily: the exact path works without it, and
using `--ann` without the extra raises a clear error. The index is a
sidecar directory (`ann_meta.json` + Lance dataset) built with
`build-ann --out DIR` from an existing DB, in chunks; ANN candidates
(combined mode only) go through the same exact rerank as vec0 candidates.
Corpora under 2,048 rows skip IVF training and use LanceDB's flat exact
scan. IVF_FLAT with 256 partitions, cosine.

Benchmarked **at 500k only** — ANN at 2M was not benchmarked, and no 2M
ANN number is claimed. Build: 16.3 s; index 3,079 MB. Tuning (50 queries,
k=10, candidate pool 40; exact search p50 at 500k: 883 ms):

| nprobes | Pool recall@40 | Top-10 agreement | search(ann) p50 |
|---:|---:|---:|---:|
| 20 | 0.6845 | 0.658 | 18.0 ms |
| 40 | 0.8755 | 0.854 | 22.7 ms |
| 80 | 0.967 | 0.950 | 29.2 ms |
| 128 | 0.998 | 0.998 | 35.6 ms |

nprobes=128 is the default: recall 0.998, 35.6 ms vs 883 ms exact =
**24.8× faster** at 500k. IVF_PQ and IVF_HNSW_FLAT were also tried and
lost on recall (see `results_store/benchmark_ann.md`/harness). On the
real-2202 corpus, ANN top-1 equals exact top-1 for 20/20 sampled queries.

### Guidance

- **Exact search is the default** and stays correct at every scale; use
  it unless latency says otherwise.
- Use `--ann` for interactive latency above roughly **100–500k vectors**,
  where exact search passes ~0.2–0.9 s per query.
- Storage: the SQLite DB costs **~14.6 KB/vector** (~2.4× the raw float32
  payload, across `chains` BLOBs + four vec0 tables) — ~29 GB at 2M
  vectors, excluding the corpus CIF files themselves. The 500k ANN
  sidecar adds ~3.1 GB.

## FAISS ANN backend (optional `faiss` extra) — 2026-10-27

Artifacts: `results_store/benchmark_faiss.{md,json}`, harness
`scripts/benchmark_faiss.py`, implementation `mmcif_vec/faiss_ann.py`
(`FaissIndex`). Indexes under `results_store/.faiss_indexes/` are
gitignored and regenerable (same for `.bench_dbs/` / `.ann_indexes/`).

FAISS is an optional dependency (`uv sync --extra faiss`, faiss-cpu
1.15.1), imported lazily like LanceDB: the exact and LanceDB paths work
without it. Build with `build-faiss --out DIR [--factory ivf_flat|ivf_pq]
[--nlist N] [--pq-m 96]`, query with `similar`/`search --faiss DIR
[--pool N]` (`--ann` and `--faiss` are mutually exclusive). Factories are
`IVF<nlist>,Flat` (default `nlist ≈ sqrt(N)`, capped 4096) and
`IVF<nlist>,PQ<pq_m>` (default PQ96), inner-product over the 768-d
combined vector, trained on ≤262k vectors. FAISS (like LanceDB) only
supplies candidates for combined queries; candidates go through the same
exact block-cosine rerank as vec0. `db.search(..., candidate_pool=...)`
(CLI `--pool`) overrides the historical `max(4k, k)` pre-rerank pool —
backward compatible (`None` keeps the old pool) and useful for
approximate candidate sources whose top-`4k` misses rerankable hits.

### Benchmark @2M (measured, synthetic-jitter corpus as in the store bench)

50 fixed queries, k=10, candidate pool 40 unless noted; exact search
p50 3,537 ms at 2M.

| Config | nprobe | Pool recall | Top-10 agree | Cand p50 | Search p50 | Index |
|---|---:|---:|---:|---:|---:|---:|
| IVF4096,Flat | 128 (3.1% of cells) | 0.7005 @40 | 0.702 | 12.5 ms | 47 ms | 6.25 GB |
| IVF512,Flat (tuned) | 256 (50% of cells) | 0.9985 @40 | 0.996 | 86 ms | 116 ms | 6.24 GB |
| IVF4096,PQ96 | 128, pool=400 | 0.2045 @400 | 0.436 | 11 ms | 171 ms | 301 MB |

**Chosen operating point: IVF512,Flat at nprobe=256** — recall@40 0.9985,
top-10 agreement 0.996, search p50 116 ms vs 3,537 ms exact (~30×), at
the cost of a 6.24 GB index and probing 50% of cells. **IVF4096,PQ96 is
not viable as the sole candidate source**: even with `candidate_pool=400`
recall@400 is only 0.2045 (top-10 0.436 after exact rerank); its 301 MB
index (~20× smaller) does not compensate. LanceDB comparison: at 500k
LanceDB reached recall 0.998 at 36 ms probing 128/256 = 50% of
partitions — FAISS matches that recall only at a comparable probed
fraction (IVF512 nprobe=256), on a 4× larger corpus; neither engine gets
near-exact recall for free at low probe fractions on this corpus.

## Invariance checks — 2026-10-27

Artifacts: `results_store/invariance_check.json`, `tests/test_invariance.py`.

- **Local `struct_vector`: exactly invariant.** Under random proper
  rotations + translations (and in the test suite, 5 seeds × 3 chains)
  the max absolute difference is 1.9e-9 — CIF coordinate rounding only.
  This is by construction (CA distance histograms, no raw xyz). Caveat:
  it is also **reflection-invariant**, i.e. chirality-blind — a structure
  and its mirror image get identical struct vectors. That is a documented
  limitation of distance-histogram features, not a bug.
- **NVIDIA AE2: NOT rotation-invariant (measured).** Method: 1UBQ chain
  encoded, then re-encoded after a random rotation (seed 12345) in the
  WSL2/CUDA AE2 stack. Rotated-vs-original pooled cosine is **0.8621**
  (not ~1.0), and per-residue latent drift reaches max L2 **4.976**
  (8-d posterior-mean latents). Implication: AE2 latents would need
  canonicalisation (or rotation augmentation at encode time) before any
  database use — another reason, beyond the v3 verdict above, that AE2
  stays rejected as a stored-vector encoder.

## v3 stratified correlations — 2026-10-27

Artifacts: `results_v3/evaluation_stratified.{json,md}` (harness
`v3/evaluate_v3.py`, strata over headline cross-entry pairs, no
bootstrap). Headline all-pairs numbers are unchanged (local Pearson
0.159 etc., table above).

| Stratum | n pairs | Local P / S | AE2 P / S | ESM2-t12 P / S |
|---|---:|---|---|---|
| Gold TM ≥ 0.5 (`candidate_high`) | 6,866 | 0.770 / 0.800 | 0.519 / 0.573 | 0.600 / 0.832 |
| Same superfamily | 6,190 | 0.718 / 0.805 | 0.488 / 0.576 | 0.558 / 0.824 |

The pair universe explains most of the gap between the low headline
correlations and the higher candidate-restricted numbers quoted in the
literature: over all 2.42M pairs the task is dominated by unrelated pairs
(gold TM ≥ 0.7 in 0.17%), whereas restricted to plausible candidates
(TM ≥ 0.5) every encoder correlates far better. Within candidate pairs,
ESM2-t12 ranks best (Spearman 0.832) while the local encoder has the best
Pearson (0.770) — consistent with the headline verdict (ESM the retrieval
winner, local the best overall TM correlation).
