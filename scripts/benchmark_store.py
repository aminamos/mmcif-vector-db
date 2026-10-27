"""Benchmark the sqlite-vec store behind mmcif_vec.db at scale.

Corpus: seeded synthetic chain vectors derived from REAL data
(results_v3/local.npz combined 768-d vectors if present, else a fixed
random fallback). Each synthetic vector = a sampled real vector +
Gaussian noise (sigma 0.05), renormalised. Block vectors are the
combined vector split into 3x256 chunks, each renormalised (plausible
only -- this measures the STORE, not encoder quality).

Generation and insertion are chunked (10k vectors); generated vectors
are never held as a full N x 768 float64 array. Combined vectors are
also written chunk-wise to a float32 .npy memmap used by the in-script
brute-force reference (chunked cosine + argpartition, no N x N).

Resumable: per-size DBs live under results_store/.bench_dbs/ and a
size whose record already exists in the output JSON is skipped.

Usage:
  uv run python scripts/benchmark_store.py [--sizes 10000,100000] \
      [--queries 50] [--k 10] [--out results_store/benchmark_store.json]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmcif_vec.db import DB, search  # noqa: E402
from mmcif_vec.features import combined_vector, seq_vector, text_vector  # noqa: E402

CHUNK = 10_000
SEED = 20_261_027
SIGMA = 0.05
AA = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------- corpus
def load_real_vectors() -> np.ndarray:
    """Return (N, 768) float32 L2-normalised real combined vectors."""
    npz = ROOT / "results_v3" / "local.npz"
    if npz.exists():
        with np.load(npz) as d:
            keys = list(d.files)
            # Current format: one (768,) float32 vector per chain, keyed by uid.
            per_chain = [k for k in keys if d[k].ndim == 1 and d[k].shape[0] == 768]
            if len(per_chain) >= 10:
                vecs = np.stack([np.asarray(d[k], dtype=np.float32) for k in sorted(per_chain)])
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return (vecs / norms).astype(np.float32)
            candidates = []
            for key in keys:
                arr = d[key]
                if arr.ndim == 2 and arr.shape[1] == 768 and arr.shape[0] >= 10:
                    candidates.append((key, arr))
            if candidates:
                candidates.sort(key=lambda kv: (0 if any(s in kv[0].lower() for s in ("combined", "emb", "vec")) else 1, kv[0]))
                vecs = np.asarray(candidates[0][1], dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return (vecs / norms).astype(np.float32)
    # Fallback (no real data on this machine): fixed random unit vectors.
    rng = np.random.default_rng(SEED)
    vecs = rng.standard_normal((256, 768), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype(np.float32)


def synth_chunk(real: np.ndarray, start: int, count: int) -> np.ndarray:
    """Deterministic (per chunk start) synthetic vectors, float32 (count,768)."""
    rng = np.random.default_rng([SEED, start])
    idx = rng.integers(0, len(real), size=count)
    vecs = real[idx].astype(np.float32, copy=True)
    vecs += rng.normal(0.0, SIGMA, size=vecs.shape).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


def split_blocks(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(768,) -> three renormalised (256,) blocks."""
    out = []
    for i in range(3):
        b = np.asarray(vec[i * 256 : (i + 1) * 256], dtype=np.float32)
        n = np.linalg.norm(b)
        out.append((b / n).astype(np.float32) if n else b)
    return out[0], out[1], out[2]


def make_queries(real: np.ndarray, n_queries: int):
    """50 fixed queries: (q_seq str, q_struct (256,), q_text str, combined (768,)).

    q_struct is the struct block of a perturbed real vector; q_seq/q_text
    are fixed strings whose hashed blocks search() derives itself. The
    brute-force reference rebuilds the exact query combined vector that
    search() builds (same features helpers), so agreement is exact.
    """
    rng = np.random.default_rng([SEED, 999_983])
    queries = []
    for qi in range(n_queries):
        base = real[rng.integers(0, len(real))].astype(np.float32, copy=True)
        base += rng.normal(0.0, SIGMA, size=base.shape).astype(np.float32)
        base /= np.linalg.norm(base)
        _, struct_block, _ = split_blocks(base)
        seq = "".join(rng.choice(list(AA), size=120))
        text = f"synthetic query protein kinase {qi}"
        q_combined = combined_vector(
            seq_vector(seq), struct_block, text_vector(text)
        ).astype(np.float32)
        queries.append(
            {
                "q_seq": seq,
                "q_struct": struct_block,
                "q_text": text,
                "combined": q_combined,
                "seq_block": seq_vector(seq),
                "text_block": text_vector(text),
            }
        )
    return queries


# ------------------------------------------------------------------- RSS
def rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            fn = ctypes.windll.psapi.GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            get_proc = ctypes.windll.kernel32.GetCurrentProcess
            get_proc.restype = wintypes.HANDLE
            counters = PMC()
            counters.cb = ctypes.sizeof(PMC)
            ok = fn(get_proc(), ctypes.byref(counters), counters.cb)
            # PeakWorkingSetSize = peak RSS of this process so far.
            return int(counters.PeakWorkingSetSize) if ok else None
        except Exception:
            return None
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return None


# -------------------------------------------------------------- brute ref
def _uid(i: int) -> str:
    return f"syn{i:07d}"


# --------------------------------------------------------------- one size
def run_size(label: str, n: int, real: np.ndarray, queries, k: int,
             bench_dir: Path, use_real_corpus: bool = False) -> dict:
    db_path = bench_dir / f"bench_{label}.db"
    npy_path = bench_dir / f"bench_{label}_combined.npy"
    for p in (db_path, npy_path):
        if p.exists():
            p.unlink()
    # WAL sidecars from a crashed run
    for suffix in ("-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()

    mm = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float32, shape=(n, 768)
    )
    db = DB(str(db_path))
    insert_seconds = 0.0
    inserted = 0
    # One ambient transaction for the whole size (the DB.ingest pattern):
    # upsert_chain joins it instead of committing per row. Generation time
    # between chunks stays outside the insert_seconds accumulation.
    with db.transaction():
        for start in range(0, n, CHUNK):
            count = min(CHUNK, n - start)
            chunk = (
                real[start : start + count].astype(np.float32, copy=False)
                if use_real_corpus
                else synth_chunk(real, start, count)
            )
            mm[start : start + count] = chunk
            mm.flush()
            t0 = time.perf_counter()
            for j in range(count):
                gi = start + j
                vec = chunk[j]
                sv, stv, tv = split_blocks(vec)
                db.upsert_chain(
                    _uid(gi) if not use_real_corpus else f"real{gi:05d}",
                    f"E{gi:07d}",
                    "A",
                    "real" if use_real_corpus else "synthetic",
                    f"synthetic chain {gi}",
                    "benchmark",
                    AA * 6,
                    len(AA * 6),
                    "",
                    sv,
                    stv,
                    tv,
                    vec,
                )
            insert_seconds += time.perf_counter() - t0
            inserted += count
    del mm
    db.con.close()

    db_bytes = db_path.stat().st_size

    # Reopen (exercises migration check on an existing vec store).
    db = DB(str(db_path))
    search_ms: list[float] = []
    new_topk: list[list[str]] = []
    for q in queries:
        t0 = time.perf_counter()
        hits = search(
            db, q_seq=q["q_seq"], q_struct=q["q_struct"], q_text=q["q_text"], k=k
        )
        search_ms.append((time.perf_counter() - t0) * 1000.0)
        new_topk.append([h["uid"] for h in hits])
    db.con.close()

    # Brute force needs uids matching the corpus labelling.
    brute_ms: list[float] = []
    agreements: list[float] = []
    for qi, q in enumerate(queries):
        topk, secs = _brute_for_label(npy_path, n, q, k, use_real_corpus)
        brute_ms.append(secs * 1000.0)
        ref = set(topk)
        got = set(new_topk[qi])
        agreements.append(len(ref & got) / k if k else 1.0)

    record = {
        "label": label,
        "size": n,
        "inserted": inserted,
        "insert_mode": "single db.transaction() per size (upsert_chain joins ambient txn)",
        "insert_seconds": round(insert_seconds, 3),
        "insert_vec_per_sec": round(n / insert_seconds, 1) if insert_seconds else None,
        "db_bytes": db_bytes,
        "db_mb": round(db_bytes / 1e6, 2),
        "search_p50_ms": round(float(np.percentile(search_ms, 50)), 3),
        "search_p95_ms": round(float(np.percentile(search_ms, 95)), 3),
        "brute_p50_ms": round(float(np.percentile(brute_ms, 50)), 3),
        "brute_p95_ms": round(float(np.percentile(brute_ms, 95)), 3),
        "top10_agreement": round(float(np.mean(agreements)), 4) if agreements else None,
        "per_query_agreement_min": round(float(np.min(agreements)), 4) if agreements else None,
        "peak_rss_bytes": rss_bytes(),
        "queries": len(queries),
        "k": k,
    }
    return record


def _brute_for_label(npy_path, n, q, k, use_real_corpus):
    """brute_topk with uid labelling for the corpus in use."""
    pool_size = max(4 * k, k)
    mm = np.load(npy_path, mmap_mode="r")
    t0 = time.perf_counter()
    cand_idx: list[int] = []
    cand_score: list[float] = []
    for start in range(0, n, CHUNK):
        block = np.asarray(mm[start : start + CHUNK], dtype=np.float32)
        scores = block @ q["combined"]
        take = min(pool_size, len(block))
        part = np.argpartition(-scores, take - 1)[:take]
        cand_idx.extend((part + start).tolist())
        cand_score.extend(scores[part].tolist())
    order = np.argsort(-np.asarray(cand_score))[:pool_size]
    results = []
    for oi in order:
        gi = cand_idx[oi]
        seq_b, struct_b, text_b = split_blocks(np.asarray(mm[gi], dtype=np.float32))
        parts = [
            float(q["seq_block"] @ seq_b),
            float(q["q_struct"] @ struct_b),
            float(q["text_block"] @ text_b),
        ]
        results.append((sum(parts) / len(parts), gi))
    results.sort(key=lambda t: -t[0])
    elapsed = time.perf_counter() - t0
    fmt = (lambda i: f"real{i:05d}") if use_real_corpus else _uid
    return [fmt(gi) for _, gi in results[:k]], elapsed


# ------------------------------------------------------------------ main
def write_md(records: list[dict], md_path: Path, notes: list[str]):
    lines = [
        "# sqlite-vec store benchmark",
        "",
        "Synthetic vectors: real combined vector + Gaussian noise "
        f"(sigma={SIGMA}), renormalised; chunked generation/insertion.",
        "",
        "| size | insert vec/s | db MB | new-search p50 ms | new-search p95 ms "
        "| brute-force p50 ms | top10 agreement |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        if r.get("skipped"):
            lines.append(
                f"| {r['label']} | skipped | - | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {r['label']} | {r['insert_vec_per_sec']} | {r['db_mb']} "
            f"| {r['search_p50_ms']} | {r['search_p95_ms']} "
            f"| {r['brute_p50_ms']} | {r['top10_agreement']} |"
        )
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10000,100000,500000,2000000")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="results_store/benchmark_store.json")
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bench_dir = out_path.parent / ".bench_dbs"
    bench_dir.mkdir(parents=True, exist_ok=True)

    data = {"results": [], "notes": []}
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"results": [], "notes": []}
    records: list[dict] = data.get("results", [])
    done = {r["label"] for r in records if not r.get("skipped")}

    def save():
        data["results"] = records
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        write_md(records, out_path.with_suffix(".md"), data.get("notes", []))

    real = load_real_vectors()
    queries = make_queries(real, args.queries)
    print(f"real corpus: {len(real)} vectors; queries: {len(queries)}", flush=True)

    jobs: list[tuple[str, int, bool]] = []
    if not args.skip_real:
        jobs.append((f"real-{len(real)}", len(real), True))
    for tok in args.sizes.split(","):
        tok = tok.strip()
        if tok:
            jobs.append((tok, int(tok), False))

    skipped = {r["label"] for r in records if r.get("skipped")}
    for label, n, use_real in jobs:
        if label in done:
            print(f"skip {label} (already recorded)", flush=True)
            continue
        if label in skipped:
            print(f"skip {label} (already recorded as skipped)", flush=True)
            continue
        # Honest stop rule: project insert time from the largest completed
        # size; skip (never fabricate) if it exceeds ~30 min additional.
        prev = [r for r in records if not r.get("skipped") and r.get("insert_vec_per_sec")]
        if prev and not use_real:
            last = prev[-1]
            projected_insert = n / last["insert_vec_per_sec"]
            if projected_insert > 1800 and n > last["size"]:
                rec = {
                    "label": label,
                    "size": n,
                    "skipped": True,
                    "reason": "projected insert time exceeds 30 min at the "
                              "largest completed size's measured rate",
                    "projected_insert_seconds": round(projected_insert, 1),
                    "projected_insert_minutes": round(projected_insert / 60, 1),
                    "based_on_label": last["label"],
                    "based_on_insert_vec_per_sec": last["insert_vec_per_sec"],
                    "projected_db_mb": round(last["db_mb"] * n / last["size"], 1),
                    "projected_search_p50_ms": round(last["search_p50_ms"] * n / last["size"], 1),
                }
                records.append(rec)
                data.setdefault("notes", []).append(
                    f"{label} skipped: projected {rec['projected_insert_minutes']} min "
                    f"insert at {last['insert_vec_per_sec']} vec/s (measured at "
                    f"{last['label']}), projected DB ~{rec['projected_db_mb']} MB, "
                    f"projected search p50 ~{rec['projected_search_p50_ms']} ms "
                    "(linear extrapolation). Not measured."
                )
                save()
                print(json.dumps(rec), flush=True)
                continue
        print(f"running {label} (n={n}) ...", flush=True)
        t0 = time.perf_counter()
        rec = run_size(label, n, real, queries, args.k, bench_dir, use_real)
        rec["wall_seconds"] = round(time.perf_counter() - t0, 1)
        records.append(rec)
        save()
        print(json.dumps(rec), flush=True)
    save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
