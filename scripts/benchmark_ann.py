"""Benchmark the optional LanceDB ANN path against exact sqlite-vec search.

Uses the EXISTING benchmark DB from scripts/benchmark_store.py
(default: results_store/.bench_dbs/bench_500000.db; never rebuilds it).
Builds an ANN index over its combined vectors, then for 50 fixed queries
(same seed protocol as benchmark_store.make_queries) measures:

- ANN build wall time + index directory bytes
- ANN candidate latency (p50/p95) per nprobes setting
- pool recall@pool: fraction of exact vec0 top-pool uids (pool = 4k = 40
  for k=10) recovered by ann.query
- final top-10 agreement after exact rerank: search(ann=) vs search()
- end-to-end search(ann=) p50 vs exact search() p50

Also verifies, on the real 2,202-vector corpus (reusing
bench_real-2202.db when present, else building it from
results_v3/local.npz via upsert_chain), that ANN top-1 equals exact
vec0 top-1 for 20 sampled real vectors.

Outputs results_store/benchmark_ann.json and .md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmcif_vec.ann import ANNIndex  # noqa: E402
from mmcif_vec.db import DB, _blob, search  # noqa: E402
from benchmark_store import load_real_vectors, make_queries, split_blocks  # noqa: E402


def pct(vals, p):
    return round(float(np.percentile(vals, p)), 3) if vals else None


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def exact_pool_uids(db, q_combined, pool):
    hits = db.con.execute(
        "SELECT uid FROM vec_combined WHERE embedding MATCH ? AND k = ?",
        (_blob(q_combined), pool)).fetchall()
    return [h[0] for h in hits]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results_store/.bench_dbs/bench_500000.db")
    ap.add_argument("--index-dir", default=None,
                    help="default: results_store/.ann_indexes/<db stem>_ann")
    ap.add_argument("--out", default="results_store/benchmark_ann.json")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--index-type", default="IVF_FLAT")
    ap.add_argument("--num-partitions", type=int, default=None)
    ap.add_argument("--num-sub-vectors", type=int, default=96)
    ap.add_argument("--nprobes-list", default="20,40,80,128")
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    if not db_path.exists():
        raise SystemExit(
            f"benchmark DB not found: {db_path} (run scripts/benchmark_store.py first; "
            "this script never rebuilds the benchmark DB)")
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index_dir = (Path(args.index_dir) if args.index_dir else
                 ROOT / "results_store" / ".ann_indexes" / f"{db_path.stem}_ann")
    if not index_dir.is_absolute():
        index_dir = ROOT / index_dir

    real = load_real_vectors()
    queries = make_queries(real, args.queries)
    k = args.k
    pool = max(4 * k, k)

    db = DB(str(db_path))
    n = db.stats()["chains"]
    print(f"db={db_path} chains={n} queries={len(queries)} k={k} pool={pool}", flush=True)

    # ---------------------------------------------------------- build/load
    meta_path = index_dir / "ann_meta.json"
    if args.rebuild or not meta_path.exists():
        t0 = time.perf_counter()
        ann = ANNIndex.build(db, index_dir, index_type=args.index_type,
                             num_partitions=args.num_partitions,
                             num_sub_vectors=args.num_sub_vectors)
        build_seconds = time.perf_counter() - t0
        built_now = True
    else:
        t0 = time.perf_counter()
        ann = ANNIndex.load(index_dir)
        build_seconds = None
        built_now = False
        print(f"loaded existing ANN index from {index_dir} "
              f"(pass --rebuild to rebuild)", flush=True)
    index_bytes = dir_bytes(index_dir)
    print(f"ann meta: {ann.meta}", flush=True)

    # ------------------------------------------------ exact reference
    exact_pools, exact_hits, exact_ms = [], [], []
    for q in queries:
        exact_pools.append(exact_pool_uids(db, q["combined"], pool))
        t0 = time.perf_counter()
        hits = search(db, q_seq=q["q_seq"], q_struct=q["q_struct"],
                      q_text=q["q_text"], k=k)
        exact_ms.append((time.perf_counter() - t0) * 1000.0)
        exact_hits.append([h["uid"] for h in hits])

    # ------------------------------------------------ per-nprobes settings
    nprobes_list = [int(x) for x in args.nprobes_list.split(",") if x.strip()]
    settings = []
    for nprobes in nprobes_list:
        ann.nprobes = nprobes
        cand_ms, recalls, agreements, e2e_ms = [], [], [], []
        for qi, q in enumerate(queries):
            t0 = time.perf_counter()
            cand = ann.query(q["combined"], pool)
            cand_ms.append((time.perf_counter() - t0) * 1000.0)
            cand_uids = [u for u, _ in cand]
            ref = set(exact_pools[qi])
            recalls.append(len(ref & set(cand_uids)) / pool if pool else 1.0)

            t0 = time.perf_counter()
            hits = search(db, q_seq=q["q_seq"], q_struct=q["q_struct"],
                          q_text=q["q_text"], k=k, ann=ann)
            e2e_ms.append((time.perf_counter() - t0) * 1000.0)
            got = [h["uid"] for h in hits]
            agreements.append(len(set(got) & set(exact_hits[qi])) / k if k else 1.0)
        s = {
            "nprobes": nprobes,
            "ann_candidate_p50_ms": pct(cand_ms, 50),
            "ann_candidate_p95_ms": pct(cand_ms, 95),
            "pool_recall_at_pool": round(float(np.mean(recalls)), 4),
            "pool_recall_min": round(float(np.min(recalls)), 4),
            "top10_agreement_after_rerank": round(float(np.mean(agreements)), 4),
            "top10_agreement_min": round(float(np.min(agreements)), 4),
            "search_ann_p50_ms": pct(e2e_ms, 50),
            "search_ann_p95_ms": pct(e2e_ms, 95),
        }
        settings.append(s)
        print(json.dumps(s), flush=True)

    exact_p50 = pct(exact_ms, 50)
    meeting = [s for s in settings if s["pool_recall_at_pool"] >= 0.98]
    chosen = (min(meeting, key=lambda s: s["search_ann_p50_ms"]) if meeting
              else max(settings, key=lambda s: s["pool_recall_at_pool"]))
    speedup = (round(exact_p50 / chosen["search_ann_p50_ms"], 2)
               if exact_p50 and chosen["search_ann_p50_ms"] else None)
    # Persist the chosen probe count as the index's default.
    ann.nprobes = chosen["nprobes"]
    ann.save()

    # ------------------------------------------------ real 2202 check
    real_result = verify_real(real, out_path.parent)
    print(json.dumps(real_result), flush=True)

    record = {
        "db": str(db_path),
        "size": n,
        "engine": ann.meta.get("engine"),
        "lancedb_version": ann.meta.get("lancedb_version"),
        "index_dir": str(index_dir),
        "index_meta": ann.meta,
        "built_now": built_now,
        "build_seconds": round(build_seconds, 3) if build_seconds is not None else None,
        "index_bytes": index_bytes,
        "index_mb": round(index_bytes / 1e6, 2),
        "queries": len(queries),
        "k": k,
        "pool": pool,
        "exact_search_p50_ms": exact_p50,
        "exact_search_p95_ms": pct(exact_ms, 95),
        "settings": settings,
        "chosen_setting": chosen,
        "chosen_speedup_vs_exact_p50": speedup,
        "target_pool_recall_ge_0.98": chosen["pool_recall_at_pool"] >= 0.98,
        "target_speedup_ge_5x": (speedup is not None and speedup >= 5),
        "real_2202": real_result,
    }
    db.con.close()

    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    write_md(record, out_path.with_suffix(".md"))
    print(f"wrote {out_path}")
    return record


def verify_real(real, store_dir: Path):
    """ANN top-1 == exact vec0 top-1 for 20 sampled real vectors."""
    bench_dir = store_dir / ".bench_dbs"
    real_db_path = bench_dir / "bench_real-2202.db"
    tmp_build = False
    if real_db_path.exists():
        db = DB(str(real_db_path))
        source = f"reused {real_db_path}"
    else:
        # Build from results_v3/local.npz vectors via upsert_chain
        # (split_blocks labelling, same as benchmark_store).
        real_db_path = bench_dir / "bench_real-2202_anncheck.db"
        db = DB(str(real_db_path))
        if db.stats()["chains"] != len(real):
            with db.transaction():
                for gi, vec in enumerate(real):
                    sv, stv, tv = split_blocks(vec)
                    db.upsert_chain(f"real{gi:05d}", f"E{gi:07d}", "A", "real",
                                    f"real chain {gi}", "benchmark", "A" * 120,
                                    120, "", sv, stv, tv, vec)
        source = f"built {real_db_path} from results_v3/local.npz via upsert_chain"
        tmp_build = True
    idx_dir = store_dir / ".ann_indexes" / "bench_real-2202_ann"
    ann = ANNIndex.build(db, idx_dir)
    rng = np.random.default_rng([20_261_027, 777])
    sample_idx = rng.choice(len(real), size=min(20, len(real)), replace=False)
    agree = 0
    details = []
    for gi in sample_idx:
        q = np.asarray(real[int(gi)], dtype=np.float32)
        ann_top = ann.query(q, 1)
        exact_top = exact_pool_uids(db, q, 1)
        ann_uid = ann_top[0][0] if ann_top else None
        exact_uid = exact_top[0] if exact_top else None
        ok = ann_uid == exact_uid
        agree += int(ok)
        details.append({"index": int(gi), "ann_top1": ann_uid,
                        "exact_top1": exact_uid, "match": ok})
    db.con.close()
    return {"source": source, "n_vectors": int(len(real)),
            "sampled": int(len(sample_idx)), "top1_matches": int(agree),
            "top1_agreement": round(agree / len(sample_idx), 4) if len(sample_idx) else None,
            "details": details}


def write_md(r, md_path: Path):
    lines = [
        "# ANN (LanceDB) benchmark",
        "",
        f"DB: `{r['db']}` ({r['size']} chains). Engine: {r['engine']} "
        f"{r['lancedb_version']}. Queries: {r['queries']} fixed "
        f"(benchmark_store seed protocol), k={r['k']}, pool={r['pool']}.",
        "",
        f"- Build: {'built now in ' + str(r['build_seconds']) + ' s' if r['built_now'] else 'index reused (build not timed this run)'}; "
        f"index dir {r['index_mb']} MB ({r['index_bytes']} bytes).",
        f"- Index params: {json.dumps(r['index_meta'])}",
        f"- Exact search p50 {r['exact_search_p50_ms']} ms (p95 {r['exact_search_p95_ms']} ms).",
        "",
        "| nprobes | ANN cand p50 ms | ANN cand p95 ms | pool recall@40 | top10 agreement | search(ann) p50 ms | search(ann) p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in r["settings"]:
        lines.append(
            f"| {s['nprobes']} | {s['ann_candidate_p50_ms']} | {s['ann_candidate_p95_ms']} "
            f"| {s['pool_recall_at_pool']} | {s['top10_agreement_after_rerank']} "
            f"| {s['search_ann_p50_ms']} | {s['search_ann_p95_ms']} |")
    c = r["chosen_setting"]
    lines += [
        "",
        f"Chosen: nprobes={c['nprobes']} — pool recall {c['pool_recall_at_pool']} "
        f"(target >=0.98: {r['target_pool_recall_ge_0.98']}), end-to-end speedup "
        f"{r['chosen_speedup_vs_exact_p50']}x vs exact (target >=5x: {r['target_speedup_ge_5x']}).",
        f"Real 2202: ANN top-1 == exact top-1 for {r['real_2202']['top1_matches']}/"
        f"{r['real_2202']['sampled']} sampled queries ({r['real_2202']['source']}).",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
