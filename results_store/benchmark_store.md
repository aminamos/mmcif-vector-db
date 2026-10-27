# sqlite-vec store benchmark

Synthetic vectors: real combined vector + Gaussian noise (sigma=0.05), renormalised; chunked generation/insertion.

| size | insert vec/s | db MB | new-search p50 ms | new-search p95 ms | brute-force p50 ms | top10 agreement |
|---|---|---|---|---|---|---|
| real-2202 | 1480.5 | 37.58 | 6.026 | 6.18 | 1.019 | 1.0 |
| 10000 | 1414.0 | 147.46 | 17.518 | 19.592 | 1.556 | 1.0 |
| 100000 | 1335.3 | 1461.39 | 176.396 | 181.509 | 14.301 | 0.998 |
| 500000 | 1296.6 | 7302.5 | 871.39 | 882.288 | 79.38 | 0.998 |
| 2000000 | 1280.1 | 29199.15 | 3538.315 | 3616.446 | 338.433 | 0.994 |

- Insert path: one db.transaction() per size, upsert_chain joining the ambient transaction (the DB.ingest pattern). Previous standalone per-row-commit run measured 296.8/286.6/229.5/237.9 vec/s at real-2202/10k/100k/500k; the transaction-wrapped re-run supersedes those numbers.
- 2M was attempted and MEASURED (not projected): 1280.1 vec/s at 2M after 1296.6 vec/s at 500k projected 1541 s (~26 min), under the ~30 min stop rule.
- Peak RSS is the process peak working set (Windows GetProcessMemoryInfo) sampled at the end of each size in the run that produced that record; 2000000 peak_rss_bytes=6324903936 (~6.32 GB). Records came from three runs of the same script, so peak values are per-run maxima, not strictly cumulative across sizes.
- Top-10 agreement below 1.0 is exclusively adjacent near-tie swaps, verified exhaustively at 500000 (1 query: rerank scores 0.398931 vs 0.398909, delta 2.2e-5) and at 2000000 (3 queries: deltas 7e-6, 4.9e-5, 7e-6 between the swapped rank-10/11 candidates; identical at search's 4-decimal reported precision). All other queries agree exactly.
- Unexpected: exact vec0 search() latency scales linearly with corpus size (6.0 ms @ 2202 -> 3538 ms @ 2000000) and is ~10x slower than the chunked numpy brute force at every large size (e.g. 3538 ms vs 338 ms @ 2M), i.e. vec0 KNN here behaves as a linear scan plus Python fetch/rerank overhead rather than an indexed lookup.
- DB size is ~14.6 KB/vector at all sizes (37.58 MB @ 2202, 29199.15 MB @ 2000000), roughly 2.4x the raw 6 KB float32 vector payload per chain across chains BLOBs + four vec0 tables; insert throughput is essentially flat (~1.28-1.48k vec/s) with no throughput cliff up to 2M.
