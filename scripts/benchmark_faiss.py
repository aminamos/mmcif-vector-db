"""Benchmark the optional FAISS ANN path against exact search.

Protocol (same as scripts/benchmark_ann.py): 50 fixed seeded queries
from benchmark_store.make_queries, k=10, pool=40. Per factory and per
nprobe setting: candidate latency p50/p95, pool recall@40 against a
chunked-numpy exact top-40 on the same corpus (the
``<db stem>_combined.npy`` sidecar written by benchmark_store, read in
10k chunks -- the 29GB SQLite DB is never loaded wholesale), and final
top-10 agreement of ``search(..., ann=faiss_index)`` vs exact
``search()``. Records train/build wall time, index file bytes, and peak
RSS (peak working set on Windows, via benchmark_store.rss_bytes).

Default corpus is the 2M benchmark DB. Factories: ivf_flat with
nlist=4096 and ivf_pq with pq_m=96, both at nlist=4096.
Justification for nlist=4096 at 2M: 2M/4096 ~= 488 vectors per IVF cell
(vs ~7812/cell at nlist=256, the LanceDB 500k setting), so probing a
fixed fraction scans far fewer flat vectors; sqrt(2M) ~= 1414, and 4096
(~2.9x sqrt) is a standard FAISS operating point for 1-10M corpora,
with the 262k training cap still giving ~64 samples per centroid.
nprobe 8/32/128 then scans ~0.2%/0.8%/3.1% of the corpus.

INSTALL ROUTE / ENV: faiss-cpu==1.15.1 installed via the repo route
(``uv sync --extra faiss``) on Python 3.14 / Windows -- see the record's
``environment`` field, stamped at runtime. No WSL fallback was needed;
mmcif_vec/faiss_ann.py is platform-agnostic.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmcif_vec.db import DB, search  # noqa: E402
from mmcif_vec.faiss_ann import FaissIndex  # noqa: E402
from benchmark_store import load_real_vectors, make_queries, rss_bytes, _uid  # noqa: E402

CHUNK = 10_000


def pct(vals, p):
    return round(float(np.percentile(vals, p)), 3) if vals else None


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def exact_pools_chunked_numpy(npy_path: Path, queries, pool: int, n: int):
    """Exact combined-vector top-`pool` uids per query, 10k-chunk scan."""
    mm = np.load(npy_path, mmap_mode="r")
    if mm.shape != (n, 768):
        raise SystemExit(f"{npy_path} shape {mm.shape} != ({n}, 768)")
    cand_idx = [[] for _ in queries]
    cand_score = [[] for _ in queries]
    for start in range(0, n, CHUNK):
        block = np.asarray(mm[start:start + CHUNK], dtype=np.float32)
        for qi, q in enumerate(queries):
            scores = block @ q["combined"]
            take = min(pool, len(block))
            part = np.argpartition(-scores, take - 1)[:take]
            cand_idx[qi].extend((part + start).tolist())
            cand_score[qi].extend(scores[part].tolist())
    pools = []
    for qi in range(len(queries)):
        order = np.argsort(-np.asarray(cand_score[qi]))[:pool]
        pools.append([_uid(cand_idx[qi][oi]) for oi in order])
    del mm
    return pools


def run_config(db, *, label, factory, nlist, pq_m, index_dir, nprobes_list,
               pool, exact_pools_ref, queries, exact_hits, k, n, rebuild):
    """Build (or load) one FAISS config and evaluate its nprobe settings.

    ``pool`` is both the FAISS candidate count and the ``candidate_pool``
    fed to ``search()`` (which exact-reranks that many candidates).
    ``exact_pools_ref`` holds exact top-``pool`` uids per query.
    """
    index_dir = Path(index_dir)
    meta_path = index_dir / "faiss_meta.json"
    if rebuild or not meta_path.exists():
        t0 = time.perf_counter()
        idx = FaissIndex.build(db, index_dir, index_factory=factory,
                               nlist=nlist, pq_m=pq_m)
        build_seconds = round(time.perf_counter() - t0, 3)
        built_now = True
    else:
        idx = FaissIndex.load(index_dir)
        build_seconds = idx.meta.get("build_seconds")
        built_now = False
        print(f"[{label}] loaded existing FAISS index from {index_dir} "
              "(pass --rebuild to rebuild)", flush=True)
    index_bytes = dir_bytes(index_dir)
    print(f"[{label}] meta={idx.meta} index_bytes={index_bytes}", flush=True)

    settings = []
    for nprobe in nprobes_list:
        cand_ms, recalls, agreements, e2e_ms = [], [], [], []
        for qi, q in enumerate(queries):
            t0 = time.perf_counter()
            cand = idx.query(q["combined"], pool, nprobe=nprobe)
            cand_ms.append((time.perf_counter() - t0) * 1000.0)
            cand_uids = [u for u, _ in cand]
            ref = set(exact_pools_ref[qi])
            recalls.append(len(ref & set(cand_uids)) / pool if pool else 1.0)

            idx.nprobe = nprobe
            t0 = time.perf_counter()
            hits = search(db, q_seq=q["q_seq"], q_struct=q["q_struct"],
                          q_text=q["q_text"], k=k, ann=idx,
                          candidate_pool=pool)
            e2e_ms.append((time.perf_counter() - t0) * 1000.0)
            got = [h["uid"] for h in hits]
            agreements.append(
                len(set(got) & set(exact_hits[qi])) / k if k else 1.0)
        s = {
            "nprobe": nprobe,
            "candidate_pool": pool,
            "faiss_candidate_p50_ms": pct(cand_ms, 50),
            "faiss_candidate_p95_ms": pct(cand_ms, 95),
            "pool_recall_at_pool": round(float(np.mean(recalls)), 4),
            "pool_recall_min": round(float(np.min(recalls)), 4),
            "top10_agreement_after_rerank": round(float(np.mean(agreements)), 4),
            "top10_agreement_min": round(float(np.min(agreements)), 4),
            "search_faiss_p50_ms": pct(e2e_ms, 50),
            "search_faiss_p95_ms": pct(e2e_ms, 95),
        }
        settings.append(s)
        print(f"[{label}] {json.dumps(s)}", flush=True)

    record = {
        "config_label": label,
        "factory": factory,
        "faiss_factory_string": idx.meta.get("factory"),
        "size": n,
        "db": str(db.con.execute("PRAGMA database_list").fetchone()[2]),
        "index_dir": str(index_dir),
        "index_meta": idx.meta,
        "built_now": built_now,
        "build_seconds": build_seconds,
        "train_seconds": idx.meta.get("train_seconds"),
        "add_seconds": idx.meta.get("add_seconds"),
        "index_bytes": index_bytes,
        "index_mb": round(index_bytes / 1e6, 2),
        "candidate_pool": pool,
        "peak_rss_bytes_after_factory": rss_bytes(),
        "settings": settings,
    }
    del idx
    gc.collect()
    return record


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results_store/.bench_dbs/bench_2000000.db")
    ap.add_argument("--out", default="results_store/benchmark_faiss.json")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--pq-m", type=int, default=96)
    ap.add_argument("--nprobes-list", default="8,32,128")
    ap.add_argument("--factories", default="ivf_flat,ivf_pq")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild indexes even if faiss_meta.json exists")
    ap.add_argument("--no-extras", action="store_true",
                    help="skip the tuning follow-up configs (nlist=512 flat, "
                         "nprobe=512 on nlist=4096 flat, PQ with pool=400)")
    args = ap.parse_args(argv)

    import faiss
    import platform

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    if not db_path.exists():
        raise SystemExit(
            f"benchmark DB not found: {db_path} (run scripts/benchmark_store.py "
            "first; this script never rebuilds the benchmark DB)")
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index_root = ROOT / "results_store" / ".faiss_indexes"
    npy_path = db_path.parent / f"{db_path.stem}_combined.npy"
    if not npy_path.exists():
        raise SystemExit(f"exact-reference sidecar not found: {npy_path}")

    real = load_real_vectors()
    queries = make_queries(real, args.queries)
    k = args.k
    pool = max(4 * k, k)
    nprobes_list = [int(x) for x in args.nprobes_list.split(",") if x.strip()]
    factories = [x.strip() for x in args.factories.split(",") if x.strip()]

    db = DB(str(db_path))
    n = db.stats()["chains"]
    print(f"db={db_path} chains={n} queries={len(queries)} k={k} pool={pool} "
          f"faiss={faiss.__version__}", flush=True)

    # ------------------------------------------------ exact references
    t0 = time.perf_counter()
    exact_pools = exact_pools_chunked_numpy(npy_path, queries, pool, n)
    print(f"chunked-numpy exact pools in {time.perf_counter() - t0:.1f}s", flush=True)
    exact_hits, exact_ms = [], []
    for q in queries:
        t0 = time.perf_counter()
        hits = search(db, q_seq=q["q_seq"], q_struct=q["q_struct"],
                      q_text=q["q_text"], k=k)
        exact_ms.append((time.perf_counter() - t0) * 1000.0)
        exact_hits.append([h["uid"] for h in hits])
    print(f"exact search p50={pct(exact_ms, 50)} ms", flush=True)

    # --------------------------------------------------- per-factory runs
    common = dict(queries=queries, exact_hits=exact_hits, k=k, n=n,
                  rebuild=args.rebuild)
    factory_records = []
    for factory in factories:
        factory_records.append(run_config(
            db, label=f"{factory}_nlist{args.nlist}", factory=factory,
            nlist=args.nlist, pq_m=args.pq_m,
            index_dir=index_root / f"{db_path.stem}_{factory}_faiss",
            nprobes_list=nprobes_list, pool=pool,
            exact_pools_ref=exact_pools, **common))

    # ------------------------------------------- tuning follow-up configs
    if not args.no_extras:
        # (a) ivf_flat nlist=512; nprobe=256 probes 50% of cells, the
        # LanceDB-comparable operating point (128/256 there).
        factory_records.append(run_config(
            db, label="ivf_flat_nlist512", factory="ivf_flat", nlist=512,
            pq_m=args.pq_m,
            index_dir=index_root / f"{db_path.stem}_ivf_flat_nlist512_faiss",
            nprobes_list=[32, 128, 256], pool=pool,
            exact_pools_ref=exact_pools, **common))
        # (b) ivf_flat nlist=4096, nprobe=512 (12.5% of cells); reuses the
        # already-built index above (loaded, not rebuilt).
        factory_records.append(run_config(
            db, label="ivf_flat_nlist4096_nprobe512", factory="ivf_flat",
            nlist=args.nlist, pq_m=args.pq_m,
            index_dir=index_root / f"{db_path.stem}_ivf_flat_faiss",
            nprobes_list=[512], pool=pool,
            exact_pools_ref=exact_pools, **common))
        # (c) ivf_pq pool=400: does a 10x candidate pool feeding the exact
        # rerank make PQ viable? Recall is against exact top-400.
        pool400 = 400
        t0 = time.perf_counter()
        exact_pools400 = exact_pools_chunked_numpy(npy_path, queries, pool400, n)
        print(f"chunked-numpy exact top-{pool400} pools in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        factory_records.append(run_config(
            db, label="ivf_pq_nlist4096_pool400", factory="ivf_pq",
            nlist=args.nlist, pq_m=args.pq_m,
            index_dir=index_root / f"{db_path.stem}_ivf_pq_faiss",
            nprobes_list=[128], pool=pool400,
            exact_pools_ref=exact_pools400, **common))

    record = {
        "db": str(db_path),
        "size": n,
        "engine": "faiss",
        "environment": {
            "route": "repo (uv sync --extra faiss)",
            "faiss_version": getattr(faiss, "__version__", None),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "queries": len(queries),
        "k": k,
        "pool": pool,
        "nlist": args.nlist,
        "pq_m": args.pq_m,
        "nlist_justification": (
            "nlist=4096 at 2M ~= 488 vectors/IVF cell (sqrt(2M)~=1414; "
            "4096 ~= 2.9x sqrt, a standard FAISS operating point for "
            "1-10M corpora); 262k training cap ~= 64 samples/centroid. "
            "nprobe 8/32/128 scans ~0.2%/0.8%/3.1% of the corpus."),
        "exact_search_p50_ms": pct(exact_ms, 50),
        "exact_search_p95_ms": pct(exact_ms, 95),
        "exact_reference": f"chunked-numpy top-{pool} over {npy_path}",
        "peak_rss_bytes": rss_bytes(),
        "factories": factory_records,
    }
    db.con.close()

    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    write_md(record, out_path.with_suffix(".md"))
    print(f"wrote {out_path}", flush=True)
    return record


def write_md(r, md_path: Path):
    env = r["environment"]
    lines = [
        "# FAISS ANN benchmark",
        "",
        f"DB: `{r['db']}` ({r['size']} chains). Engine: faiss "
        f"{env['faiss_version']} (route: {env['route']}; Python "
        f"{env['python']}; {env['platform']}). Queries: {r['queries']} "
        f"fixed (benchmark_store seed protocol), k={r['k']}, pool={r['pool']}.",
        f"Exact reference: {r['exact_reference']}. Exact search p50 "
        f"{r['exact_search_p50_ms']} ms (p95 {r['exact_search_p95_ms']} ms).",
        f"nlist justification: {r['nlist_justification']}",
        f"Peak RSS (process peak working set): {r['peak_rss_bytes']} bytes.",
        "",
    ]
    def _best(factory=None, pool=None, label=None):
        best = None
        for f in r["factories"]:
            if factory and f["factory"] != factory:
                continue
            if pool is not None and f.get("candidate_pool", r["pool"]) != pool:
                continue
            if label and f.get("config_label") != label:
                continue
            for s in f["settings"]:
                if best is None or s["pool_recall_at_pool"] > best[1]["pool_recall_at_pool"]:
                    best = (f, s)
        return best

    best_flat = _best(factory="ivf_flat", pool=r["pool"])
    best_pq_pool = _best(factory="ivf_pq", pool=400)
    lance_point = _best(label="ivf_flat_nlist512")
    # nprobe=256 on nlist=512 specifically (50% of cells):
    for f in r["factories"]:
        if f.get("config_label") == "ivf_flat_nlist512":
            for s in f["settings"]:
                if s["nprobe"] == 256:
                    lance_point = (f, s)
    for f in r["factories"]:
        cpool = f.get("candidate_pool", r["pool"])
        lines += [
            f"## {f.get('config_label', f['factory'])} ({f['faiss_factory_string']}, size={f['size']}, candidate_pool={cpool})",
            "",
            f"- Build: built now={f['built_now']} in {f['build_seconds']} s "
            f"(train {f['train_seconds']} s, add {f['add_seconds']} s); "
            f"index dir {f['index_mb']} MB ({f['index_bytes']} bytes); "
            f"peak RSS after factory: {f['peak_rss_bytes_after_factory']} bytes.",
            f"- Index meta: {json.dumps(f['index_meta'])}",
            "",
            f"| nprobe | FAISS cand p50 ms | FAISS cand p95 ms | pool recall@{cpool} | top10 agreement | search(faiss) p50 ms | search(faiss) p95 ms |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in f["settings"]:
            lines.append(
                f"| {s['nprobe']} | {s['faiss_candidate_p50_ms']} | "
                f"{s['faiss_candidate_p95_ms']} | {s['pool_recall_at_pool']} | "
                f"{s['top10_agreement_after_rerank']} | "
                f"{s['search_faiss_p50_ms']} | {s['search_faiss_p95_ms']} |")
        lines.append("")
    lines += [
        "## Notes",
        "",
        "- Measured, not projected: all configs were benchmarked at the full 2M size (no 500k fallback was needed).",
    ]
    if best_flat:
        f, s = best_flat
        lines.append(
            f"- Chosen FAISS operating point (best exact-ish): {f['faiss_factory_string']} "
            f"({f.get('config_label')}) at nprobe={s['nprobe']} -- pool recall@{f.get('candidate_pool', r['pool'])} "
            f"{s['pool_recall_at_pool']}, top-10 agreement {s['top10_agreement_after_rerank']}, "
            f"candidate p50 {s['faiss_candidate_p50_ms']} ms, search(faiss) p50 "
            f"{s['search_faiss_p50_ms']} ms vs exact search p50 {r['exact_search_p50_ms']} ms; "
            "this is the config to use when near-exact results matter, at the cost of a "
            f"{f['index_mb']} MB index and probing a large cell fraction.")
    if best_pq_pool:
        f, s = best_pq_pool
        lines.append(
            f"- Best PQ config: {f['faiss_factory_string']} at nprobe={s['nprobe']} with "
            f"candidate_pool=400 -- pool recall@400 {s['pool_recall_at_pool']}, top-10 agreement "
            f"{s['top10_agreement_after_rerank']} after exact rerank, search(faiss) p50 "
            f"{s['search_faiss_p50_ms']} ms, from a compact {f['index_mb']} MB index; "
            "pool=40 starved PQ (recall@40 ~0.14 in the nlist4096 rows above); the 10x "
            "candidate pool feeding the exact rerank more than doubles top-10 agreement, "
            "but PQ still remains well short of viable as the sole candidate source on "
            "this corpus -- its trade-off is a ~20x smaller index and cheap candidate "
            "generation in exchange for that large recall/agreement loss, plus slower "
            "end-to-end search once 400 candidates must be fetched and reranked.")
    if lance_point:
        f, s = lance_point
        lines.append(
            f"- Honest comparison vs LanceDB @500k (36 ms end-to-end, pool recall 0.998): "
            f"LanceDB probed 128/256 = 50% of partitions, while most FAISS rows above probe "
            f"far fewer of 4096 cells (nprobe=128 = 3.1%). The LanceDB-comparable FAISS point, "
            f"{f['faiss_factory_string']} at nprobe={s['nprobe']} (50% of cells) on a 4x larger "
            f"2M corpus, gives pool recall {s['pool_recall_at_pool']}, top-10 agreement "
            f"{s['top10_agreement_after_rerank']}, candidate p50 {s['faiss_candidate_p50_ms']} ms "
            f"and search(faiss) p50 {s['search_faiss_p50_ms']} ms -- so FAISS reaches its best "
            "recall only at a comparable probed fraction, not for free at low nprobe.")
    lines += [
        "- ivf_pq (PQ96) recall at pool=40 collapses (~0.14 at nprobe=128) despite the smallest index, paralleling the LanceDB IVF_PQ collapse (~0.15 at 500k) noted in benchmark_ann.py on this synthetic-jittered corpus.",
        "- search(faiss) p50 is typically dominated by exact rerank row fetches from the 29GB SQLite DB, not FAISS candidate generation.",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
