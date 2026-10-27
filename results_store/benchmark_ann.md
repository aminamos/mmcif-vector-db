# ANN (LanceDB) benchmark

DB: `E:\development\mmcif-vector-db\results_store\.bench_dbs\bench_500000.db` (500000 chains). Engine: lancedb 0.40.0. Queries: 50 fixed (benchmark_store seed protocol), k=10, pool=40.

- Build: built now in 16.347 s; index dir 3078.97 MB (3078971832 bytes).
- Index params: {"engine": "lancedb", "lancedb_version": "0.40.0", "dim": 768, "n": 500000, "vector": "combined", "column": "combined_vec", "created_from": "E:\\development\\mmcif-vector-db\\results_store\\.bench_dbs\\bench_500000.db", "index_type": "IVF_FLAT", "index_created": true, "num_partitions": 256, "num_sub_vectors": null, "nprobes": 128, "metric": "cosine"}
- Exact search p50 882.876 ms (p95 895.774 ms).

| nprobes | ANN cand p50 ms | ANN cand p95 ms | pool recall@40 | top10 agreement | search(ann) p50 ms | search(ann) p95 ms |
|---|---|---|---|---|---|---|
| 20 | 9.278 | 40.102 | 0.6845 | 0.658 | 17.993 | 19.456 |
| 40 | 12.892 | 18.1 | 0.8755 | 0.854 | 22.688 | 23.648 |
| 80 | 19.251 | 21.857 | 0.967 | 0.95 | 29.188 | 30.596 |
| 128 | 25.508 | 26.613 | 0.998 | 0.998 | 35.564 | 36.656 |

Chosen: nprobes=128 — pool recall 0.998 (target >=0.98: True), end-to-end speedup 24.82x vs exact (target >=5x: True).
Real 2202: ANN top-1 == exact top-1 for 20/20 sampled queries (reused E:\development\mmcif-vector-db\results_store\.bench_dbs\bench_real-2202.db).
