# DRAFT: NVIDIA autoencoder encoder (La-Proteina / RFD4-Proteina)

Status: **evaluated and rejected (2026-10-26)** — benchmark run, local
encoder stays the default. See the NVIDIA section of
[`EVALUATION.md`](../EVALUATION.md) and `results/BENCHMARK_RUN.md`.
The local encoder in `mmcif_vec/features.py` remains the default. This
document was the work order for evaluating the NVIDIA autoencoder as a
replacement encoder, nothing more.

## Why this exists

Current encoder: sequence hashed k-mers + rotation/translation-invariant
CA distance histograms (`mmcif_vec/features.py`). Measured so far:

- v1 retrieval, leave-self-out top-1 family accuracy: 21/22 on the
  bundled 13-entry sample set (`EVALUATION.md`)
- v1 forced clustering purity 0.77 (fixed in v2 by neighbour grouping)
- known construction limits (not measured against any alternative):
  CA histogram discards fold topology order and all side-chain atoms;
  k-mers degrade at low sequence identity

Hypothesis to test (unproven): NVIDIA's autoencoder latents — trained so
each residue latent jointly carries amino-acid identity and side-chain
geometry, with CA coordinates explicit — produce chain vectors that
retrieve/cluster better, especially for same-fold / low-identity pairs.

## What the NVIDIA models are

- La-Proteina (NVIDIA, arXiv:2507.09466,
  github.com/NVIDIA-Digital-Bio/la-proteina): VAE encodes sequence +
  all non-CA atom coordinates into fixed-size per-residue latents;
  decoder reconstructs sequence + all-atom structure. A flow-matching
  model generates in that partially latent space.
- Proteina-Complexa: builds on La-Proteina for binder/complex design.
- RFD4-Proteina (RosettaCommons/RFD4-Proteina, IPD + NVIDIA): latent
  flow model; ~434M params total, ~256M of them the autoencoder. The
  trunk runs in autoencoder latent space.

## The CUDA / core requirements (the actual blocker)

This is why it is not in the MVP. From the La-Proteina repo setup:

- `torch==2.7.0` from the **cu118** wheel index
  (`pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118`)
- `torch_geometric`, `torch_scatter`, `torch_sparse`, `torch_cluster`
  built for `torch-2.7.0+cu118`
- `graphein==1.7.7`, mamba/conda env (`environment.yaml` in that repo)
- Model checkpoints downloaded into `./checkpoints_laproteina`
  (NGC / Hugging Face; exact checkpoint + licence terms to confirm)

Practical consequences:

- Needs an NVIDIA GPU host. The development laptop here (Apple M3,
  no CUDA) cannot run this stack as specified. CPU/MPS ports are
  out of scope unless the reviewing agent proves one works.
- Environment is heavy and version-pinned; it must live in its own
  env/service, never inside this repo's `uv` environment.
- RFD4-Proteina also ships as an NVIDIA NIM (`ipd/rfd4-proteina`).
  A NIM/API route, if accessible, avoids the local CUDA stack but
  needs an NVIDIA API key and sends structures to a remote service.
  Local stack vs NIM is an open decision (see Questions).

## Interface contract (must match before any code)

The rest of the pipeline is encoder-agnostic. The new encoder must
fulfil the same contract as the local one, so v1/v2, clustering,
complex recipes, and tests do not change:

```
encode_chain(residues) -> np.ndarray[float32, fixed_dim], L2-normalised
```

Input problem to solve first: **our parser currently keeps only CA
atoms** (`mmcif_vec/parse.py`, `CA` representative atoms from
`_atom_site`). The autoencoder needs all-atom residues (backbone +
side chain). Required parser extension:

- Keep every `_atom_site` atom per residue (or at least backbone
  N/CA/C/O + side chain), with the existing altloc / HETATM /
  first-model / zero-occupancy rules applied per atom, not per CA.
- Preserve residue order and label/auth numbering already retained.
- Define behaviour for missing side-chain atoms, non-standard
  residues (MSE handled; others TBD), nucleic acids (exclude for now,
  as today), and chains longer than the model's residue limit
  (La-Proteina reports up to ~800; chunking policy TBD — do not
  silently truncate).

Aggregation: per-residue latents + explicit CA trace must be reduced
to ONE vector per chain. Options to evaluate, in order:

1. Mean-pool per-residue latents; append a summary of the CA trace.
2. Mean + max pool.
3. Learned pooling — out of scope for the evaluation.

State the chosen fixed dimension and normalisation in code.

## Benchmark protocol (how it earns the swap)

No swap without beating the baseline on identical code paths:

1. Same inputs: `samples/` (13 entries, 30 chains) plus a harder set
   to be added: same SCOP/CATH fold, low sequence identity (<30%),
   and same-sequence / different-state controls if available.
2. Same downstream code: swap only the encoder behind the contract
   above; v1 retrieval, v2 families, and complex grouping unchanged.
3. Metrics: leave-self-out top-1 family accuracy; v2 family purity;
   complex-group agreement on haemoglobin/MEF2 controls.
4. Controls: Foldseek (and/or an ESM-family sequence encoder) on the
   same sets, so "beats hashed k-mers" is not confused with "good".
5. Report runtime, memory, and failure cases per entry. A win only
   on the easy bundled set is not sufficient.

Acceptance: strictly better than the local encoder on the hard set,
not worse on the bundled set, and runnable by someone other than
the author from the setup doc alone.

## Suggested task breakdown for the reviewing agent

- [x] Confirm checkpoint source, size, and licence for the La-Proteina
      (or RFD4-Proteina) autoencoder weights; link them here.
      Done: `AE1_ucond_512.ckpt` / `AE2_ucond_800.ckpt` from Hugging Face
      `nvidia/NV-La-Proteina-Ucond-v1` (ungated, NVIDIA Open Model License).
- [x] Stand up the CUDA environment on a GPU host; record GPU, driver,
      CUDA, and exact package versions that worked.
      Done: RTX 4070 SUPER 12 GB (120 W cap), WSL2 Ubuntu-22.04, torch 2.7.0+cu118.
- [x] Prove encoder-only inference: load a checkpoint, encode one
      small PDB chain (e.g. 1UBQ), print latent shape. No generation.
- [x] Extend `_atom_site` parsing to all-atom residues (above).
- [x] Implement `encode_chain` aggregation; wire behind the existing
      `mmcif_vec/nvidia_encoder.py` stub (it currently raises
      `NotAvailable` on purpose — keep that behaviour until wired).
      Done: default pooling is now mean-only 8-d (validated; see EVALUATION.md).
- [x] Run the benchmark protocol; write results into `EVALUATION.md`
      or a sibling file, including the losses.
      Done: see `EVALUATION.md` and `results/BENCHMARK_RUN.md`.
- [ ] Only if accepted: document how a user enables it
      (`--encoder nvidia` or equivalent) and the GPU requirement.
      Not done — not accepted (rejected 2026-10-26), so this stays unticked.

## Questions / open decisions

- Local CUDA stack vs NVIDIA NIM API route? (key management, data
  leaving the machine, cost, offline requirement)
- La-Proteina checkpoint vs RFD4-Proteina autoencoder — which is
  actually obtainable and licence-compatible for this use?
- Residue-limit and long-chain policy (AFDB full-length models vs
  domains — the exact case the simple encoder handles by containment).
- Is generation ever in scope? (Currently no: retrieval/clustering
  only. Generation would be a separate proposal.)

## What not to do

- Do not make the NVIDIA encoder the default, and do not add torch
  to this repo's dependencies.
- Do not remove or weaken the local encoder or its tests.
- Do not claim improvement without the side-by-side benchmark above.
