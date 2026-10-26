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

## NVIDIA autoencoder?

Not the default — see `mmcif_vec/nvidia_encoder.py`. The La-Proteina /
RFD4-Proteina autoencoder (explicit CA + per-residue latents for sequence and
side chains, 256M params in RFD4) is a drop-in encoder swap behind the same
`chain -> vector` contract, once clustering/RAG value is proven and CUDA +
weights are available. The MVP encoder proves the pipeline today, offline.

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

- Swap encoder -> NVIDIA La-Proteina/RFD4-Proteina autoencoder latents
- Scale store -> sqlite-vec / LanceDB; bulk PDB + AFDB ingest
- SIFTS UniProt/Pfam metadata joins; Prompteina text<->latent alignment
