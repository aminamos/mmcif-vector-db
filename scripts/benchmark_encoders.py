#!/usr/bin/env python
"""Benchmark encoder NPZ files on the bundled samples (numpy only).

Protocol (reconstructed from EVALUATION.md; reproduces the local
combined_vector baseline of 21/22):

- 30 chains in samples/ (13 entries). Family labels by entry:
  globin-alpha / globin-beta (1A3N + 1HHO, split by sequence),
  MEF2 (all chains of 1EGW / 1N6J / 1TQE, including the short
  co-complex peptide chains of 1N6J/1TQE), Foxk1 (2A3S / 2D2W /
  AF-P42128-F1). That is 3 + 3 + 13 + 3 = 22 family queries.
  Remaining entries (1LYZ, 1EMA, 1UBQ, 4INS, 6LU7) are singletons:
  no other entry shares their fold/family in the set, so they have
  no correct top-1 and are reported separately, never counted in
  the family accuracy. (4INS chains are insulin A/B fragments of a
  single entry, so at entry level it is a singleton too.)
- Primary metric: leave-ENTRY-out top-1 — every chain of the query's
  own entry is excluded, so exact duplicate chains deposited in the
  same entry cannot trivially answer the query. Secondary metric:
  leave-self-out (only the query uid excluded). Both give 21/22 for
  the local encoder; the only miss is AF-P42128-F1 (719-res full-length
  AFDB model vs 101-res PDB Foxk1 domains), the known v1 limitation.

Usage:
  uv run python scripts/benchmark_encoders.py --dump-local results/local.npz
  uv run python scripts/benchmark_encoders.py results/local.npz results/nvidia_ae1.npz
  uv run python scripts/benchmark_encoders.py results/a.npz --sidecar results/a.json

An NPZ maps uid -> 1-D vector (keys ending in ``__latents`` are
ignored, so per-residue latent dumps can share the file). A JSON
sidecar next to the NPZ (same stem) or given with --sidecar supplies
runtime / memory columns. Sequences for the pairwise-identity report
come from parsing --samples (and --hard manifest entries if present).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GLOBIN_ENTRIES = {"1A3N", "1HHO"}
MEF2_ENTRIES = {"1EGW", "1N6J", "1TQE"}
FOXK1_ENTRIES = {"2A3S", "2D2W", "AF-P42128-F1"}


def family_label(entry_id: str, sequence: str):
    """Family label for a samples/ chain, or None for singletons."""
    if entry_id in GLOBIN_ENTRIES:
        return "globin-alpha" if sequence.startswith("VLSP") else "globin-beta"
    if entry_id in MEF2_ENTRIES:
        return "MEF2"
    if entry_id in FOXK1_ENTRIES:
        return "Foxk1"
    return None


def load_records(samples_dirs):
    from mmcif_vec.parse import parse_cif

    records = []
    for d in samples_dirs:
        d = Path(d)
        if not d.exists():
            continue
        files = sorted(d.rglob("*.cif")) if d.is_dir() else [d]
        for f in files:
            records.extend(parse_cif(f))
    return records


def build_labels(records, hard_dir=None):
    """uid -> (family-or-None, entry_id, sequence). Manifest extends samples."""
    labels = {}
    for r in records:
        labels[r.uid] = (family_label(r.entry_id, r.sequence), r.entry_id, r.sequence)
    if hard_dir:
        manifest = Path(hard_dir) / "manifest.json"
        if manifest.exists():
            for ent in json.loads(manifest.read_text()).get("entries", []):
                fam = ent.get("family")
                for chain in ent.get("chains", []):
                    uid = f"{ent['entry']}:{chain}"
                    if uid in labels:
                        labels[uid] = (fam, ent["entry"], labels[uid][2])
    return labels


def sequence_identity(a: str, b: str) -> float:
    """Global (Needleman-Wunsch) pairwise identity: matches / aligned columns.

    Scoring match +1, mismatch -1, gap -1. Deterministic, dependency-free.
    """
    if not a or not b:
        return 0.0
    # keep the smaller sequence as columns for a smaller DP matrix
    if len(b) > len(a):
        a, b = b, a
    n, m = len(a), len(b)
    scores = np.zeros((n + 1, m + 1), dtype=np.int32)
    scores[:, 0] = -np.arange(n + 1)
    scores[0, :] = -np.arange(m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        prev, row = scores[i - 1], scores[i]
        for j in range(1, m + 1):
            row[j] = max(prev[j - 1] + (1 if ai == b[j - 1] else -1),
                         prev[j] - 1, row[j - 1] - 1)
    # traceback, counting matches over aligned columns
    i, j, matches, cols = n, m, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and scores[i, j] == scores[i - 1, j - 1] + (1 if a[i - 1] == b[j - 1] else -1):
            matches += int(a[i - 1] == b[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and scores[i, j] == scores[i - 1, j] - 1:
            i -= 1
        else:
            j -= 1
        cols += 1
    return matches / max(cols, 1)


def identity_report(labels):
    """Cross-entry same-family pairwise identities (evidence of hardness)."""
    by_family: dict[str, list] = {}
    for uid, (fam, entry, seq) in sorted(labels.items()):
        if fam:
            by_family.setdefault(fam, []).append((uid, entry, seq))
    rows = []
    for fam, members in sorted(by_family.items()):
        vals = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if members[i][1] == members[j][1]:
                    continue  # cross-entry pairs only
                vals.append(sequence_identity(members[i][2], members[j][2]))
        if vals:
            rows.append((fam, len(vals), float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))))
    return rows


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        vecs = {k: np.asarray(z[k], dtype=np.float64).ravel()
                for k in z.files if not k.endswith("__latents")}
    return vecs


def _norm_rows(vecs, uids):
    mat = np.stack([vecs[u] for u in uids])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms > 0, norms, 1.0)


def evaluate(vecs, labels, name):
    uids = [u for u in vecs if u in labels]
    cand_uids = list(vecs)
    mat = _norm_rows(vecs, cand_uids)
    index = {u: i for i, u in enumerate(cand_uids)}
    fam_total = fam_ok_entry = fam_ok_self = 0
    singleton_uids, misses = [], []
    for uid in uids:
        fam, entry, _seq = labels[uid]
        sims = mat @ mat[index[uid]]
        order = np.argsort(-sims)
        if fam is None:
            singleton_uids.append(uid)
            continue
        fam_total += 1
        for mode, exclude_entry in (("entry", True), ("self", False)):
            top = next(cand_uids[k] for k in order
                       if cand_uids[k] != uid and not (exclude_entry and labels.get(cand_uids[k], (None, None))[1] == entry))
            hit = labels.get(top, (None,))[0] == fam
            if mode == "entry":
                fam_ok_entry += int(hit)
                if not hit:
                    misses.append((uid, fam, top, labels.get(top, (None,))[0]))
            else:
                fam_ok_self += int(hit)
    n_queries = len(uids)
    overall = fam_ok_entry / n_queries if n_queries else 0.0
    return {
        "name": name, "dim": int(next(iter(vecs.values())).size) if vecs else 0,
        "n_vectors": len(vecs), "n_queries_labelled": n_queries,
        "family_queries": fam_total,
        "leave_entry_out_correct": fam_ok_entry,
        "leave_entry_out_accuracy": fam_ok_entry / fam_total if fam_total else 0.0,
        "leave_self_out_correct": fam_ok_self,
        "leave_self_out_accuracy": fam_ok_self / fam_total if fam_total else 0.0,
        "singleton_queries": len(singleton_uids),
        "singleton_uids": singleton_uids,
        # Singletons have no correct top-1 by construction; this is the
        # family accuracy diluted by counting them as misses.
        "overall_incl_singletons_accuracy": overall,
        "misses": misses,
    }


def sidecar_for(npz_path, explicit):
    if explicit:
        return explicit
    cand = Path(npz_path).with_suffix(".json")
    return str(cand) if cand.exists() else None


def read_sidecar(path):
    if not path:
        return {}
    try:
        d = json.loads(Path(path).read_text())
    except Exception:
        return {}
    out = {}
    for key in ("total_seconds", "elapsed_seconds", "runtime_seconds", "wall_seconds", "seconds"):
        if isinstance(d.get(key), (int, float)):
            out["runtime_s"] = d[key]
            break
    for key in ("peak_gpu_mem_mb", "peak_gpu_memory_mb", "peak_mem_mb", "peak_memory_mb"):
        if isinstance(d.get(key), (int, float)):
            out["peak_mem_mb"] = d[key]
            break
    if isinstance(d.get("per_chain"), dict) and d["per_chain"]:
        secs = [v.get("seconds") for v in d["per_chain"].values()
                if isinstance(v, dict) and isinstance(v.get("seconds"), (int, float))]
        if secs:
            out.setdefault("runtime_s", float(sum(secs)))
            out["mean_chain_s"] = float(np.mean(secs))
    out["sidecar"] = d
    return out


def dump_local(samples_dir, out_path):
    from mmcif_vec.features import combined_vector, seq_vector, struct_vector, text_vector

    records = load_records([samples_dir])
    vecs, per_chain = {}, {}
    t0 = time.perf_counter()
    for r in records:
        t = time.perf_counter()
        vecs[r.uid] = combined_vector(seq_vector(r.sequence), struct_vector(r.ca_coords),
                                      text_vector(r.text))
        per_chain[r.uid] = {"seconds": time.perf_counter() - t, "n_res": len(r.sequence)}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{k: np.asarray(v, dtype=np.float32) for k, v in vecs.items()})
    sidecar = {"encoder": "local combined_vector (seq+struct+text)",
               "total_seconds": time.perf_counter() - t0, "peak_mem_mb": None,
               "per_chain": per_chain, "failures": {}}
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1))
    print(f"wrote {out_path} ({len(vecs)} vectors) + {out_path.with_suffix('.json')}")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="*", help="NPZ files mapping uid -> vector")
    ap.add_argument("--sidecar", action="append", default=[],
                    help="JSON sidecar(s), in the same order as NPZ files (default: <npz stem>.json)")
    ap.add_argument("--samples", default="samples", help="samples dir for labels/sequences")
    ap.add_argument("--hard", default="samples_hard", help="hard-set dir with manifest.json (optional)")
    ap.add_argument("--dump-local", metavar="OUT.npz", help="dump local-encoder vectors from --samples and exit")
    ap.add_argument("--json", metavar="OUT.json", help="also write machine-readable results here")
    args = ap.parse_args(argv)

    if args.dump_local:
        dump_local(args.samples, args.dump_local)
        return 0
    if not args.npz:
        ap.error("give at least one NPZ file, or --dump-local")

    records = load_records([args.samples, args.hard])
    labels = build_labels(records, args.hard)

    results = []
    for i, npz_path in enumerate(args.npz):
        vecs = load_npz(npz_path)
        res = evaluate(vecs, labels, Path(npz_path).stem)
        res.update(read_sidecar(sidecar_for(npz_path, args.sidecar[i] if i < len(args.sidecar) else None)))
        res.pop("sidecar", None)
        results.append(res)

    print("| encoder | dim | vectors | family leave-entry-out | family leave-self-out | singletons (excluded) | overall incl. singletons | runtime (s) | peak mem (MB) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        runtime = f"{r['runtime_s']:.2f}" if isinstance(r.get("runtime_s"), (int, float)) else "-"
        peak = f"{r['peak_mem_mb']:.1f}" if isinstance(r.get("peak_mem_mb"), (int, float)) else "-"
        print(f"| {r['name']} | {r['dim']} | {r['n_vectors']} | "
              f"{r['leave_entry_out_correct']}/{r['family_queries']} = {r['leave_entry_out_accuracy']:.3f} | "
              f"{r['leave_self_out_correct']}/{r['family_queries']} = {r['leave_self_out_accuracy']:.3f} | "
              f"{r['singleton_queries']} | {r['overall_incl_singletons_accuracy']:.3f} | "
              f"{runtime} | {peak} |")
    for r in results:
        if r["misses"]:
            print(f"\n{r['name']} leave-entry-out misses:")
            for uid, fam, top, top_fam in r["misses"]:
                print(f"  {uid} ({fam}) -> {top} ({top_fam})")
        if r["singleton_uids"]:
            print(f"{r['name']} singleton queries (no correct top-1 exists): {', '.join(r['singleton_uids'])}")

    rows = identity_report(labels)
    if rows:
        print("\nCross-entry same-family pairwise sequence identity (Needleman-Wunsch):")
        print("| family | pairs | mean | min | max |")
        print("|---|---|---|---|---|")
        for fam, n, mean, lo, hi in rows:
            print(f"| {fam} | {n} | {mean:.3f} | {lo:.3f} | {hi:.3f} |")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"results": results,
             "identity": [{"family": f, "pairs": n, "mean": m, "min": lo, "max": hi}
                          for f, n, m, lo, hi in rows]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
