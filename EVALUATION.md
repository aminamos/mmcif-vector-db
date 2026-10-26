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
