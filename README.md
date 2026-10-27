# mmcif-vector-db — MVP

RAG over PDB + AFDB: put an mmCIF file in a vector database, considering
**sequence and structure relative to other mmCIFs**. Cluster, rerank, retrieve.

Prompteina / RFD4-Proteina (text prompt -> molecular complex) is the long-term
direction. This MVP is the retrieval substrate underneath it.

## Start here: v2 (recommended)

One command, nothing to tune — it groups the sample proteins into
families, leaves unrelated proteins alone, and names one representative
to look at per family:

```bash
uv run python -m v2 build samples/
uv run --group dev pytest -q   # 3 tests
```

See `v2/README.md` for what v2 fixes over v1 and its sample results.
The rest of this page documents the v1 baseline underneath it.

## Design

- **Unit = one protein chain** (AFDB is per-chain; PDB entries split by chain).
- **Parse:** gemmi. Representative atoms = `CA` only, first model.
- **Vectors per chain** (all 256-d, L2-normalised, comparable across PDB/AFDB):
  - `seq_vec` — AA composition + hashed 2/3-mers
  - `struct_vec` — translation/rotation-invariant CA fingerprint:
    pairwise + local (i..i+2/3/4/8) distance histograms, Rg, contacts.
    No raw xyz — rotated/translated copies stay neighbours.
  - `text_vec` — hashed title/method/source tokens
  - `combined_vec` — concat(seq, struct, text) weighted 0.45/0.45/0.10
- **Store:** one SQLite file — `chains` table (metadata + seq/struct/text/combined float32 BLOBs) plus four sqlite-vec `vec0` KNN tables (`vec_combined` 768-d, others 256-d). vec0 generates candidates; the exact block-cosine rerank below is unchanged, so results match the old brute-force scan. Optional ANN sidecars for interactive latency at scale: LanceDB (`--ann`) or FAISS (`--faiss`, third tier). Old DBs migrate/backfill on open. See [EVALUATION.md](EVALUATION.md#vector-store-sqlite-vec--optional-lancedb-ann--2026-10-27).
- **Cluster:** k-means (cosine, k-means++) over `combined_vec` -> `cluster_id`.
- **Rerank:** retrieve by combined vector, rerank by mean(seq, struct, text)
  scores; every hit reports all three scores separately.
- **RAG:** `rag` emits LLM-ready context blocks per retrieved chain.

## Encoder comparison (v1, preliminary — 47-chain smoke set)

Leave-entry-out top-1 family retrieval; `samples/` = 22 queries,
`samples_hard/` = 17 queries. **Preliminary/smoke-set numbers only** — the
authoritative benchmark is the v3 table below. Detail: [EVALUATION.md](EVALUATION.md).

| Encoder | Dim | samples top-1 | hard top-1 | Runtime | GPU/CPU | Dependency cost |
|---|---:|---:|---:|---|---|---|
| Local (default) | 768 | 21/22 | 16/17 | 0.04 s | CPU | none |
| NVIDIA La-Proteina AE2, mean-only | 8 | 19/22 | 16/17 | ~1.7 s encode | GPU, ~3.2 GB peak | 4.1 GB checkpoints, WSL2/Linux CUDA stack outside repo |
| NVIDIA AE2, mean+max+std | 24 | 10/22 | 8/17 | ~1.7 s encode | GPU, ~3.2 GB peak | same as above |
| ESM2-t12 35M (control) | 480 | 22/22 | 17/17 | 0.79 s / 0.72 s | GPU, 232 / 151 MB | pretrained model + transformers |
| Foldseek (control) | — | 19/22 | 17/17 | ~1.5 s search | CPU | static Foldseek binary |

## Encoder comparison (v3, authoritative — 2,202 SCOPe domains)

Exhaustive Foldseek gold (2,422,451 cross-entry pairs, 100% coverage, nothing
imputed), superseding the 47-chain numbers above — including a plain retraction
of the v1-era hits-only correlations (0.72/0.49/0.40, range-restricted). Full
protocol, CIs, USalign gold-validation caveat, and sensitivity analysis:
[EVALUATION.md](EVALUATION.md#v3-scope-benchmark-2026-10-26--authoritative-supersedes-the-47-chain-numbers).

| Encoder | Dim | Pearson | Spearman | top-1 | AUPR | AUROC | Cost |
|---|---:|---:|---:|---:|---:|---:|---|
| Local (default) | 768 | 0.159 | 0.120 | 0.771 | 0.468 | 0.906 | 17 s CPU, no dependencies |
| ESM2-t12 35M | 480 | 0.087 | 0.084 | 0.892 | 0.498 | 0.897 | 29 s GPU, 534 MB peak, pretrained model + transformers |
| NVIDIA La-Proteina AE2, mean-only | 8 | 0.104 | 0.084 | 0.305 | 0.103 | 0.813 | 321 s GPU sharded, 2.1 GB peak, 4.1 GB checkpoints + WSL2/CUDA stack |

Stratified (gold TM ≥ 0.5 pairs, n=6,866): local P/S 0.770/0.800, ESM2-t12 0.600/0.832 — see EVALUATION.md.

### Should we use the NVIDIA encoder?

(a) **Can we go without it? Yes.** The local encoder equals or beats it
everywhere that counts, with zero dependencies; it stays the default.
(b) **Can we build our own encoder? We could** — the contract is one
function, so a small learned/contrastive encoder is feasible.
(c) **Should we? No — not now.** The v3 benchmark (2,202 SCOPe domains,
exhaustive gold) settles it: the local encoder still stands as the default
(best TM correlation and AUROC, zero dependencies), ESM2-t12 is the optional
upgrade path if retrieval quality becomes the bottleneck (best top-1, 0.892 —
as a separate proposal for an optional pretrained sequence encoder), and
NVIDIA La-Proteina is out decisively at scale (worst retrieval and
correlation, heaviest stack). Not training our own either.

## Run

```bash
cd mmcif-vector-db
uv run mmcif-vec --db mmcif.db ingest examples/
uv run mmcif-vec --db mmcif.db cluster --k 2
uv run mmcif-vec --db mmcif.db similar examples/2a3s.cif
uv run mmcif-vec --db mmcif.db search --text "myocyte nuclear factor"
uv run mmcif-vec --db mmcif.db rag "myocyte nuclear factor DNA binding"
# Optional ANN sidecars (LanceDB / FAISS) for large corpora:
uv sync --extra ann
uv run mmcif-vec --db mmcif.db build-ann --out mmcif_ann
uv run mmcif-vec --db mmcif.db similar examples/2a3s.cif --ann mmcif_ann
uv sync --extra faiss
uv run mmcif-vec --db mmcif.db build-faiss --out mmcif_faiss
uv run mmcif-vec --db mmcif.db similar examples/2a3s.cif --faiss mmcif_faiss --pool 40
uv run mmcif-vec --db mmcif.db search --text "myocyte nuclear factor" --faiss mmcif_faiss
uv run --group dev pytest -q
```

Examples: PDB 2A3S / 1N6J / 1EGW, AFDB AF-P42128-F1 (same Foxk1 protein as
2A3S — they should retrieve/cluster together on seq+struct, across sources).

## Store scale (measured; detail in [EVALUATION.md](EVALUATION.md#vector-store-sqlite-vec--optional-lancedb-ann--2026-10-27))

Exact search is the default (vec0 is a linear scan — exact, single-file).
`--ann` (LanceDB IVF_FLAT, nprobes=128) is for interactive latency above
~100–500k vectors; `--faiss` (FAISS) is the @2M tier: chosen operating
point IVF512,Flat nprobe=256 gives recall@40 0.9985 at search p50 116 ms
at 2M (6.24 GB index). FAISS IVF_PQ is **not recommended as the sole
candidate source** (recall@400 0.2045 even with `--pool 400`).

| Size | Insert (vec/s) | Exact search p50 | ANN search p50 | DB size |
|---|---:|---:|---:|---:|
| real-2202 | 1,480 | 6.0 ms | — | 38 MB |
| 10k | 1,414 | 17.5 ms | — | 147 MB |
| 100k | 1,335 | 176 ms | — | 1.5 GB |
| 500k | 1,297 | 871 ms | 35.6 ms (recall 0.998, 24.8×) | 7.3 GB |
| 2M | 1,280 | 3,538 ms | not benchmarked | 29.2 GB |

## Next (not MVP)

- Optional pretrained sequence encoder (e.g. ESM2-t12) if retrieval quality becomes the bottleneck — separate proposal; NVIDIA La-Proteina evaluated and rejected (above)
- Bulk PDB + AFDB ingest
- SIFTS UniProt/Pfam metadata joins; Prompteina text<->latent alignment
