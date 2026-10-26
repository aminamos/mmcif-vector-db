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
- **Store:** SQLite, vectors as float32 blobs, brute-force cosine (MVP scale).
- **Cluster:** k-means (cosine, k-means++) over `combined_vec` -> `cluster_id`.
- **Rerank:** retrieve by combined vector, rerank by mean(seq, struct, text)
  scores; every hit reports all three scores separately.
- **RAG:** `rag` emits LLM-ready context blocks per retrieved chain.

## Encoder comparison (measured 2026-10-26)

Leave-entry-out top-1 family retrieval; `samples/` = 22 queries,
`samples_hard/` = 17 queries. Detail: [EVALUATION.md](EVALUATION.md).

| Encoder | Dim | samples top-1 | hard top-1 | Runtime | GPU/CPU | Dependency cost |
|---|---:|---:|---:|---|---|---|
| Local (default) | 768 | 21/22 | 16/17 | 0.04 s | CPU | none |
| NVIDIA La-Proteina AE2, mean-only | 8 | 19/22 | 16/17 | ~1.7 s encode | GPU, ~3.2 GB peak | 4.1 GB checkpoints, WSL2/Linux CUDA stack outside repo |
| NVIDIA AE2, mean+max+std | 24 | 10/22 | 8/17 | ~1.7 s encode | GPU, ~3.2 GB peak | same as above |
| ESM2-t12 35M (control) | 480 | 22/22 | 17/17 | 0.79 s / 0.72 s | GPU, 232 / 151 MB | pretrained model + transformers |
| Foldseek (control) | — | 19/22 | 17/17 | ~1.5 s search | CPU | static Foldseek binary |

### Should we use the NVIDIA encoder?

(a) **Can we go without it? Yes.** The local encoder equals or beats it
everywhere that counts, with zero dependencies; it stays the default.
(b) **Can we build our own encoder? We could** — the contract is one
function, so a small learned/contrastive encoder is feasible.
(c) **Should we? No — not now.** The benchmark is 47 chains total, and
ESM2-t12 off-the-shelf already scores 22/22 + 17/17 at 151–232 MB GPU.
If retrieval quality ever becomes the bottleneck, the cheap step is
wiring a small pretrained sequence encoder as an *optional* encoder
(separate proposal), not training our own and not adopting La-Proteina:
4.1 GB checkpoints, a Linux/CUDA-only stack, and a 130M encoder whose
pooled latents merely tie hand-crafted features.

## Run

```bash
cd mmcif-vector-db
uv run mmcif-vec --db mmcif.db ingest examples/
uv run mmcif-vec --db mmcif.db cluster --k 2
uv run mmcif-vec --db mmcif.db similar examples/2a3s.cif
uv run mmcif-vec --db mmcif.db search --text "myocyte nuclear factor"
uv run mmcif-vec --db mmcif.db rag "myocyte nuclear factor DNA binding"
uv run --group dev pytest -q
```

Examples: PDB 2A3S / 1N6J / 1EGW, AFDB AF-P42128-F1 (same Foxk1 protein as
2A3S — they should retrieve/cluster together on seq+struct, across sources).

## Next (not MVP)

- Optional pretrained sequence encoder (e.g. ESM2-t12) if retrieval quality becomes the bottleneck — separate proposal; NVIDIA La-Proteina evaluated and rejected (above)
- Scale store -> sqlite-vec / LanceDB; bulk PDB + AFDB ingest
- SIFTS UniProt/Pfam metadata joins; Prompteina text<->latent alignment
