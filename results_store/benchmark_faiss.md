# FAISS ANN benchmark

DB: `E:\development\mmcif-vector-db\results_store\.bench_dbs\bench_2000000.db` (2000000 chains). Engine: faiss 1.15.1 (route: repo (uv sync --extra faiss); Python 3.14.2; Windows-10-10.0.19045-SP0). Queries: 50 fixed (benchmark_store seed protocol), k=10, pool=40.
Exact reference: chunked-numpy top-40 over E:\development\mmcif-vector-db\results_store\.bench_dbs\bench_2000000_combined.npy. Exact search p50 3536.691 ms (p95 3585.955 ms).
nlist justification: nlist=4096 at 2M ~= 488 vectors/IVF cell (sqrt(2M)~=1414; 4096 ~= 2.9x sqrt, a standard FAISS operating point for 1-10M corpora); 262k training cap ~= 64 samples/centroid. nprobe 8/32/128 scans ~0.2%/0.8%/3.1% of the corpus.
Peak RSS (process peak working set): 8197304320 bytes.

## ivf_flat_nlist4096 (IVF4096,Flat, size=2000000, candidate_pool=40)

- Build: built now=False in 345.399 s (train 61.899 s, add 280.71 s); index dir 6252.62 MB (6252616462 bytes); peak RSS after factory: 6471266304 bytes.
- Index meta: {"engine": "faiss", "faiss_version": "1.15.1", "factory": "IVF4096,Flat", "index_factory": "ivf_flat", "dim": 768, "n": 2000000, "vector": "combined", "column": "combined_vec", "created_from": "E:\\development\\mmcif-vector-db\\results_store\\.bench_dbs\\bench_2000000.db", "nlist": 4096, "pq_m": null, "trained": true, "train_sample": 262144, "nprobe": 32, "metric": "inner_product", "train_seconds": 61.899, "add_seconds": 280.71, "build_seconds": 345.399}

| nprobe | FAISS cand p50 ms | FAISS cand p95 ms | pool recall@40 | top10 agreement | search(faiss) p50 ms | search(faiss) p95 ms |
|---|---|---|---|---|---|---|
| 8 | 1.572 | 2.156 | 0.2075 | 0.22 | 51.219 | 55.876 |
| 32 | 4.211 | 4.883 | 0.4125 | 0.422 | 41.186 | 47.174 |
| 128 | 12.476 | 13.515 | 0.7005 | 0.702 | 47.24 | 51.733 |

## ivf_pq_nlist4096 (IVF4096,PQ96, size=2000000, candidate_pool=40)

- Build: built now=False in 453.431 s (train 149.786 s, add 303.381 s); index dir 301.4 MB (301402933 bytes); peak RSS after factory: 6471266304 bytes.
- Index meta: {"engine": "faiss", "faiss_version": "1.15.1", "factory": "IVF4096,PQ96", "index_factory": "ivf_pq", "dim": 768, "n": 2000000, "vector": "combined", "column": "combined_vec", "created_from": "E:\\development\\mmcif-vector-db\\results_store\\.bench_dbs\\bench_2000000.db", "nlist": 4096, "pq_m": 96, "trained": true, "train_sample": 262144, "nprobe": 32, "metric": "inner_product", "train_seconds": 149.786, "add_seconds": 303.381, "build_seconds": 453.431}

| nprobe | FAISS cand p50 ms | FAISS cand p95 ms | pool recall@40 | top10 agreement | search(faiss) p50 ms | search(faiss) p95 ms |
|---|---|---|---|---|---|---|
| 8 | 1.39 | 1.872 | 0.091 | 0.11 | 40.68 | 46.358 |
| 32 | 3.667 | 4.151 | 0.117 | 0.146 | 35.411 | 38.298 |
| 128 | 10.729 | 11.656 | 0.141 | 0.19 | 41.366 | 44.556 |

## ivf_flat_nlist512 (IVF512,Flat, size=2000000, candidate_pool=40)

- Build: built now=True in 310.101 s (train 41.436 s, add 265.521 s); index dir 6241.58 MB (6241577741 bytes); peak RSS after factory: 8197304320 bytes.
- Index meta: {"engine": "faiss", "faiss_version": "1.15.1", "factory": "IVF512,Flat", "index_factory": "ivf_flat", "dim": 768, "n": 2000000, "vector": "combined", "column": "combined_vec", "created_from": "E:\\development\\mmcif-vector-db\\results_store\\.bench_dbs\\bench_2000000.db", "nlist": 512, "pq_m": null, "trained": true, "train_sample": 262144, "nprobe": 32, "metric": "inner_product", "train_seconds": 41.436, "add_seconds": 265.521, "build_seconds": 309.915}

| nprobe | FAISS cand p50 ms | FAISS cand p95 ms | pool recall@40 | top10 agreement | search(faiss) p50 ms | search(faiss) p95 ms |
|---|---|---|---|---|---|---|
| 32 | 13.655 | 17.055 | 0.68 | 0.702 | 65.332 | 70.968 |
| 128 | 46.506 | 51.15 | 0.9505 | 0.938 | 81.363 | 86.908 |
| 256 | 85.972 | 92.541 | 0.9985 | 0.996 | 116.139 | 138.243 |

## ivf_flat_nlist4096_nprobe512 (IVF4096,Flat, size=2000000, candidate_pool=40)

- Build: built now=False in 345.399 s (train 61.899 s, add 280.71 s); index dir 6252.62 MB (6252616462 bytes); peak RSS after factory: 8197304320 bytes.
- Index meta: {"engine": "faiss", "faiss_version": "1.15.1", "factory": "IVF4096,Flat", "index_factory": "ivf_flat", "dim": 768, "n": 2000000, "vector": "combined", "column": "combined_vec", "created_from": "E:\\development\\mmcif-vector-db\\results_store\\.bench_dbs\\bench_2000000.db", "nlist": 4096, "pq_m": null, "trained": true, "train_sample": 262144, "nprobe": 32, "metric": "inner_product", "train_seconds": 61.899, "add_seconds": 280.71, "build_seconds": 345.399}

| nprobe | FAISS cand p50 ms | FAISS cand p95 ms | pool recall@40 | top10 agreement | search(faiss) p50 ms | search(faiss) p95 ms |
|---|---|---|---|---|---|---|
| 512 | 38.489 | 42.045 | 0.9495 | 0.932 | 79.375 | 87.665 |

## ivf_pq_nlist4096_pool400 (IVF4096,PQ96, size=2000000, candidate_pool=400)

- Build: built now=False in 453.431 s (train 149.786 s, add 303.381 s); index dir 301.4 MB (301402933 bytes); peak RSS after factory: 8197304320 bytes.
- Index meta: {"engine": "faiss", "faiss_version": "1.15.1", "factory": "IVF4096,PQ96", "index_factory": "ivf_pq", "dim": 768, "n": 2000000, "vector": "combined", "column": "combined_vec", "created_from": "E:\\development\\mmcif-vector-db\\results_store\\.bench_dbs\\bench_2000000.db", "nlist": 4096, "pq_m": 96, "trained": true, "train_sample": 262144, "nprobe": 32, "metric": "inner_product", "train_seconds": 149.786, "add_seconds": 303.381, "build_seconds": 453.431}

| nprobe | FAISS cand p50 ms | FAISS cand p95 ms | pool recall@400 | top10 agreement | search(faiss) p50 ms | search(faiss) p95 ms |
|---|---|---|---|---|---|---|
| 128 | 10.943 | 12.056 | 0.2045 | 0.436 | 171.344 | 252.611 |

## Notes

- Measured, not projected: all configs were benchmarked at the full 2M size (no 500k fallback was needed).
- Chosen FAISS operating point (best exact-ish): IVF512,Flat (ivf_flat_nlist512) at nprobe=256 -- pool recall@40 0.9985, top-10 agreement 0.996, candidate p50 85.972 ms, search(faiss) p50 116.139 ms vs exact search p50 3536.691 ms; this is the config to use when near-exact results matter, at the cost of a 6241.58 MB index and probing a large cell fraction.
- Best PQ config: IVF4096,PQ96 at nprobe=128 with candidate_pool=400 -- pool recall@400 0.2045, top-10 agreement 0.436 after exact rerank, search(faiss) p50 171.344 ms, from a compact 301.4 MB index; pool=40 starved PQ (recall@40 ~0.14 in the nlist4096 rows above); the 10x candidate pool feeding the exact rerank more than doubles top-10 agreement, but PQ still remains well short of viable as the sole candidate source on this corpus -- its trade-off is a ~20x smaller index and cheap candidate generation in exchange for that large recall/agreement loss, plus slower end-to-end search once 400 candidates must be fetched and reranked.
- Honest comparison vs LanceDB @500k (36 ms end-to-end, pool recall 0.998): LanceDB probed 128/256 = 50% of partitions, while most FAISS rows above probe far fewer of 4096 cells (nprobe=128 = 3.1%). The LanceDB-comparable FAISS point, IVF512,Flat at nprobe=256 (50% of cells) on a 4x larger 2M corpus, gives pool recall 0.9985, top-10 agreement 0.996, candidate p50 85.972 ms and search(faiss) p50 116.139 ms -- so FAISS reaches its best recall only at a comparable probed fraction, not for free at low nprobe.
- ivf_pq (PQ96) recall at pool=40 collapses (~0.14 at nprobe=128) despite the smallest index, paralleling the LanceDB IVF_PQ collapse (~0.15 at 500k) noted in benchmark_ann.py on this synthetic-jittered corpus.
- search(faiss) p50 is typically dominated by exact rerank row fetches from the 29GB SQLite DB, not FAISS candidate generation.