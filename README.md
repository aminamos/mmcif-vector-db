# mmcif-vector-db

Put mmCIF files (PDB + AlphaFold DB) into a vector database that considers
**sequence and structure relative to other mmCIFs**. The unit is one protein
chain: each chain gets comparable vectors, and the system clusters, reranks,
and retrieves chains by those vectors. It is a retrieval substrate — the
long-term direction is Prompteina-style text → molecular-complex retrieval.

## Quickstart

```bash
uv sync
uv run mmcif-vec --db mmcif.db ingest samples/
uv run mmcif-vec --db mmcif.db stats
uv run mmcif-vec --db mmcif.db similar samples/2a3s.cif
uv run mmcif-vec --db mmcif.db search --text "myocyte nuclear factor" --k 5
uv run mmcif-vec --db mmcif.db cluster --k 3
uv run mmcif-vec --db mmcif.db rag "myocyte nuclear factor DNA binding"
```

`samples/` is bundled and runnable as-is. It includes PDB 2A3S and the
AlphaFold model AF-P42128-F1 (the same Foxk1 protein from two sources) —
with `cluster --k 3` as above, they land in the same cluster.

Optional ANN sidecars, for large corpora:

```bash
uv sync --extra ann
uv run mmcif-vec --db mmcif.db build-ann --out mmcif_ann
uv run mmcif-vec --db mmcif.db similar samples/2a3s.cif --ann mmcif_ann

uv sync --extra faiss
uv run mmcif-vec --db mmcif.db build-faiss --out mmcif_faiss
uv run mmcif-vec --db mmcif.db similar samples/2a3s.cif --faiss mmcif_faiss --pool 40
uv run mmcif-vec --db mmcif.db search --text "myocyte nuclear factor" --faiss mmcif_faiss --pool 40
```

`--db` defaults to `mmcif.db`. `similar` / `search` accept `--k`,
`--ann <dir>`, `--faiss <dir>`, and `--pool <n>` (candidate pool fetched
before exact reranking); `search` takes `--text` and/or `--seq`.

## How it works

### Parser

gemmi, reading the `_atom_site` category: first model only; altloc
`.` / `?` / `A` only; zero-occupancy atoms dropped; only `ATOM` rows with a
standard/modified amino-acid `label_comp_id` become residues (so waters,
ligands, and calcium ions are excluded, and MSE counts as M when it appears
as an `ATOM` row). Both numbering schemes are retained (`label_seq_id` and
`auth_seq_id` + insertion code), and every retained atom of each residue is
kept in `Residue.atoms`, with CA as the representative coordinate.

### Encoders

The local and NVIDIA encoders sit behind one contract: one L2-normalised vector per chain.

- **Local (default).** Three 256-d blocks — sequence (amino-acid composition
  + hashed 2/3-mers), structure (CA pairwise + local distance histograms,
  Rg, contacts; no raw xyz), and text (hashed title/method/source tokens) —
  concatenated with weights 0.45 / 0.45 / 0.10 into a 768-d combined vector.
  CPU-only, no model downloads.
- **NVIDIA La-Proteina AE** (`mmcif_vec/nvidia_encoder.py`, optional):
  mean-pooled 8-d latents. Evaluated and **rejected** — it is not the
  default, and it needs a WSL2/CUDA stack and checkpoints that live outside
  this repo.
- **ESM2** (35M and 650M) was evaluated as a control on the same benchmark.
  It is not wired in as an encoder module; the 650M model gave the best
  retrieval numbers measured (see Benchmark).

### Stores and retrieval

Chains live in one SQLite file: a `chains` table (metadata plus
seq/struct/text/combined float32 BLOBs) and four sqlite-vec `vec0` KNN
tables (`vec_combined` 768-d, the rest 256-d). Search generates candidates
(vec0, LanceDB, or FAISS), then reranks them with exact block cosines; every
hit reports its sequence, structure, and text scores separately. `search()`
also accepts `candidate_pool=` to widen the pre-rerank pool. Clustering is
k-means (cosine, k-means++) over the combined vector, written back as
`cluster_id`. `rag` emits LLM-ready context blocks per retrieved chain.

- **Exact (vec0)** — default. A linear scan, so results are exact; fine up
  to ~100k vectors.
- **LanceDB** (`uv sync --extra ann`, `build-ann` / `--ann`) — for
  interactive latency above ~100–500k vectors.
- **FAISS** (`uv sync --extra faiss`, `build-faiss` / `--faiss` / `--pool`)
  — the 2M-vector tier. FAISS IVF-PQ is not viable as the sole candidate
  source; use IVF-Flat indexes with exact reranking.

## Benchmark summary

v3 benchmark: 2,202 SCOPe domains, exhaustive Foldseek gold with 100%
coverage of 2,423,301 pairs, validated against USalign. Headline
(cross-entry pairs vs Foldseek gold; P = Pearson, S = Spearman):

| Encoder | Dim | P | S | top-1 | AUPR | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Local (default) | 768 | 0.159 | 0.120 | 0.771 | 0.468 | 0.906 |
| NVIDIA La-Proteina AE2, mean-only | 8 | 0.104 | 0.084 | 0.305 | 0.103 | 0.813 |
| ESM2-t12 35M | 480 | 0.087 | 0.084 | 0.892 | 0.498 | 0.897 |
| ESM2-t33 650M | 1280 | 0.057 | 0.034 | 0.903 | 0.528 | 0.895 |

All-pairs correlations are dominated by unrelated pairs; restricted to
candidate pairs (gold TM ≥ 0.5), the local encoder reaches P 0.770 / S 0.800.

Invariance: the local structure vector is exactly rotation/translation
invariant by construction (distance histograms only — which also makes it
chirality-blind, a known limitation). The NVIDIA AE2 latents are measurably
**not** rotation-invariant (rotated-vs-original cosine 0.862).

Store scale (corpora above the real 2,202-vector set are synthetic
benchmark corpora):

| Corpus | Insert | Exact search p50 | ANN search p50 | DB size |
|---|---:|---:|---:|---:|
| real-2202 | 1,480 vec/s | 6.0 ms | — | 38 MB |
| 500k (synthetic) | 1,297 vec/s | 871 ms | 35.6 ms, LanceDB recall 0.998 | 7.3 GB |
| 2M (synthetic) | 1,280 vec/s | 3,537 ms | 116 ms, FAISS recall@40 0.9985 (IVF512,Flat, nprobe 256) | 29.2 GB |

Measured insert rate is ~1.3k vectors/s and the exact DB is ~14.6 KB/vector
(29 GB at 2M vectors, excluding the CIF files themselves).

Protocol, confidence intervals, and full evidence: [EVALUATION.md](EVALUATION.md).

## Live demo

A hosted copy runs at <https://mmcif-vectorize.a-8c6.workers.dev>
(Worker code lives in the separate `mmcif-vectorize-worker` project, not here).

- `GET /` and `GET /health` are open and report the indexes and counts:
  `local` — 2,202 real SCOPe domain vectors (local encoder, 768-d);
  `esm650` — the same 2,202 real domains (ESM2-650M, 1280-d);
  `bench2m` — 2,000,000 **synthetic** benchmark vectors (not real proteins;
  exists only to prove scale).
- `POST /search` and `POST /similar` require an API key — ask the
  maintainer. No key is published here.

## Project layout

- `mmcif_vec/parse.py` — gemmi `_atom_site` parser → per-chain records
- `mmcif_vec/features.py` — local sequence/structure/text vectors, 768-d combined
- `mmcif_vec/db.py` — SQLite + sqlite-vec store, exact search and rerank
- `mmcif_vec/ann.py` / `mmcif_vec/faiss_ann.py` — optional LanceDB / FAISS tiers
- `mmcif_vec/cluster.py` — k-means over combined vectors
- `mmcif_vec/cli.py` — `mmcif-vec` CLI (`ingest`, `similar`, `search`, `cluster`, `rag`, `stats`, `build-ann`, `build-faiss`)
- `mmcif_vec/nvidia_encoder.py` — optional NVIDIA La-Proteina encoder (rejected; not default)
- `samples/`, `samples_hard/`, `examples/` — bundled mmCIF files for the quickstart and checks
- `v3/`, `scripts/` — v3 benchmark dataset builder and evaluation harness
- `results_v3/`, `results_store/`, `results/` — benchmark outputs behind EVALUATION.md
- `tests/` — pytest suite
- `docs/` — design notes (including the NVIDIA encoder contract)

## Tests

```bash
uv run --group dev pytest -q   # 54 passing
```

## Roadmap

- kNN-graph clustering at scale (ANN candidates → connected components)
- Bulk PDB / AlphaFold DB ingest pipeline
- Optional wired ESM encoder (separate proposal)
- SIFTS UniProt/Pfam metadata joins
- Prompteina text ↔ latent alignment direction

Full evaluation history and methodology: [EVALUATION.md](EVALUATION.md).
