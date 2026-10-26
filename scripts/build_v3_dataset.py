"""Build the v3 SCOPe domain unit set (see v3/manifest.json metadata).

Protocol points implemented here:
 1. Unit set fixed FIRST: SCOPe domains (defined residue ranges), sampled
    deterministically (seed 42) BEFORE any encoding/gold computation.
    Filters (declared a priori, in metadata): single auth chain;
    segment-estimated (or, for whole-chain domains, parsed) length
    40-512 residues; superfamilies with >=4 eligible domains; cap 8
    sampled per superfamily; standard-AA sequence (parser alphabet,
    selenocysteine 'U' rejected).
 3. Domains failing validation are dropped and counted by reason;
    nothing is imputed or silently substituted.

Source: SCOPe parseable classification dir.cla.scope 2.01-stable
https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.01-stable.txt
(cached under v3/.cache_entries/). Slicing downloads entry mmCIFs from
https://files.rcsb.org/download/<PDB>.cif (cached under
v3/.cache_entries/), keeps residues whose (auth_chain, auth_seq_id,
ins_code) fall in the SCOPe segment ranges (segment order preserved,
discontinuous segments concatenated), writes a gemmi mmCIF per domain
under v3/domains/, and validates by re-parsing with mmcif_vec.parse:
exactly 1 chain, residue count within 2 of the expected segment length,
parsed length 40-512, standard-AA sequence.

Deterministic / resumable: sampling depends only on the cached CLA file
and the seed; existing sliced files are re-validated and reused instead
of being re-downloaded/re-sliced, so a re-run yields the same sampled
domain_id list (and, given the same cache, the same manifest).
No dependencies beyond the repo's (gemmi, numpy-unused-here, stdlib).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime
import json
import random
import re
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gemmi  # noqa: E402
from mmcif_vec.parse import parse_cif  # noqa: E402

CLA_URL = "https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.01-stable.txt"
CLA_NAME = "dir.cla.scope.2.01-stable.txt"
SOURCE_NAME = "SCOPe"
SOURCE_VERSION = "2.01-stable"
ENTRY_URL = "https://files.rcsb.org/download/{pdb}.cif"
SEED = 42
MIN_LEN, MAX_LEN = 40, 512
CAP_PER_SUPERFAMILY = 8
MIN_PER_SUPERFAMILY = 4
POOL_SIZE = 3200  # candidates attempted; expected validated yield ~2000-2500
COUNT_TOLERANCE = 2
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
POOL_SIZE = 2600  # candidates attempted; ~89% validated in pilot -> ~2300 domains
SEG_RE = re.compile(r"^(-?\d+)([A-Za-z]?)-(-?\d+)([A-Za-z]?)$")


# ---------------------------------------------------------------- CLA parse
def parse_cla(path: Path):
    """Yield dicts for parseable CLA rows; count unparseable ones."""
    rows, bad = [], 0
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6:
            bad += 1
            continue
        sid, pdb, spec, sccs = parts[0].strip(), parts[1].strip().lower(), parts[2].strip(), parts[3].strip()
        try:
            hier = dict(kv.split("=") for kv in parts[5].split(","))
            fold_id, sf_id, fa_id = int(hier["cf"]), int(hier["sf"]), int(hier["fa"])
        except Exception:
            bad += 1
            continue
        segments, chains, est, ok = [], set(), 0, True
        for part in spec.split(","):
            if ":" not in part:
                ok = False
                break
            chain, rng = part.split(":", 1)
            chain = chain.strip()
            chains.add(chain)
            if rng == "":
                segments.append({"chain": chain, "start": None, "start_ins": "",
                                 "end": None, "end_ins": ""})
            else:
                m = SEG_RE.match(rng.strip())
                if not m:
                    ok = False
                    break
                segments.append({"chain": chain, "start": int(m.group(1)),
                                 "start_ins": m.group(2), "end": int(m.group(3)),
                                 "end_ins": m.group(4)})
                est += int(m.group(3)) - int(m.group(1)) + 1
        if not ok:
            bad += 1
            continue
        bounded = all(s["start"] is not None for s in segments)
        rows.append({
            "domain_id": sid, "pdb_id": pdb, "segments": segments,
            "chains": chains, "estimated_length": est if bounded else None,
            "sccs": sccs, "fold_id": fold_id, "superfamily_id": sf_id,
            "family_id": fa_id, "scop_sid": sid,
        })
    return rows, bad


def sample_candidates(rows, seed=SEED, pool_size=POOL_SIZE):
    """Apply a-priori filters, then deterministic seeded sampling.

    Returns (candidates, prefilter_counts). Superfamilies are visited in
    seeded-shuffled order and contributed whole (up to the cap) until the
    candidate pool reaches pool_size, so smaller pools are prefixes.
    """
    counts = collections.Counter()
    eligible = []
    for r in rows:
        counts["cla_rows"] += 1
        if len(r["chains"]) != 1:
            counts["dropped_multi_chain"] += 1
            continue
        if r["estimated_length"] is not None and not (MIN_LEN <= r["estimated_length"] <= MAX_LEN):
            counts["dropped_estimated_length_out_of_range"] += 1
            continue
        eligible.append(r)
    by_sf = collections.defaultdict(list)
    for r in eligible:
        by_sf[r["superfamily_id"]].append(r)
    for sf in list(by_sf):
        if len(by_sf[sf]) < MIN_PER_SUPERFAMILY:
            counts["dropped_superfamilies_below_min_eligible"] += 1
            counts["dropped_domains_in_small_superfamilies"] += len(by_sf.pop(sf))
    rng = random.Random(seed)
    sf_ids = sorted(by_sf)
    rng.shuffle(sf_ids)
    candidates = []
    for sf in sf_ids:
        if len(candidates) >= pool_size:
            break
        doms = sorted(by_sf[sf], key=lambda r: r["domain_id"])
        rng.shuffle(doms)
        candidates.extend(doms[:CAP_PER_SUPERFAMILY])
    counts["candidate_pool"] = len(candidates)
    counts["candidate_superfamilies"] = len({r["superfamily_id"] for r in candidates})
    return candidates, counts


# ------------------------------------------------------------------ slicing
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _pdb_lock(pdb: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(pdb, threading.Lock())


def download_entry(pdb: str, cache_dir: Path) -> Path:
    dest = cache_dir / f"{pdb}.cif"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with _pdb_lock(pdb):
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        tmp = cache_dir / f"{pdb}.cif.tmp{threading.get_ident()}"
        last = None
        for _ in range(2):
            try:
                req = urllib.request.Request(ENTRY_URL.format(pdb=pdb.upper()),
                                             headers={"User-Agent": "mmcif-vector-db-v3-build/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
                    fh.write(resp.read())
                tmp.replace(dest)
                return dest
            except Exception as e:  # noqa: BLE001 - recorded as failure reason
                last = e
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(f"download_failed: {last}")


def _ins(v: str) -> str:
    return "" if v in (".", "?", "") else v


def _key(auth_seq: str, ins: str):
    try:
        return (int(auth_seq), _ins(ins))
    except (TypeError, ValueError):
        return None


def select_residues(records, segments):
    """Residues of the (single) auth chain inside segments, segment order."""
    chain = segments[0]["chain"]
    residues = []
    for rec in records:
        if rec.auth_chain_id == chain:
            residues.extend(rec.residues)
    if not residues:
        return [], 0, chain
    selected, seen = [], set()
    expected = 0
    for seg in segments:
        if seg["start"] is None:
            expected += len(residues)
            chosen = residues
        else:
            lo, hi = (seg["start"], seg["start_ins"]), (seg["end"], seg["end_ins"])
            expected += seg["end"] - seg["start"] + 1
            chosen = []
            for r in residues:
                k = _key(r.auth_seq_id, r.ins_code)
                if k is not None and lo <= k <= hi:
                    chosen.append(r)
        for r in chosen:
            if id(r) not in seen:
                seen.add(id(r))
                selected.append(r)
    return selected, expected, chain


def write_domain_cif(path: Path, domain_id: str, pdb_id: str, chain: str, residues) -> None:
    doc = gemmi.cif.Document()
    block = doc.add_new_block(domain_id)
    block.set_pair("_entry.id", domain_id)
    loop = block.init_loop("_atom_site.", [
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
        "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
        "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
        "B_iso_or_equiv", "auth_seq_id", "auth_asym_id", "pdbx_PDB_model_num",
    ])
    atom_id = 0
    for i, r in enumerate(residues, 1):
        for a in r.atoms:
            atom_id += 1
            element = (a.element or "").strip() or a.name.strip("0123456789")[0]
            loop.add_row([
                "ATOM", str(atom_id), element, a.name, ".", r.comp_id, "A", "1",
                str(i), _ins(r.ins_code) or ".",
                f"{a.coord[0]:.3f}", f"{a.coord[1]:.3f}", f"{a.coord[2]:.3f}",
                "." if a.occupancy is None else f"{a.occupancy:.2f}",
                "." if a.b_factor is None else f"{a.b_factor:.2f}",
                str(r.auth_seq_id), chain, "1",
            ])
    doc.write_file(str(path))


def validate_sliced(path: Path):
    """Re-parse a sliced file; return (sequence, n_residues) or raise."""
    records = parse_cif(path)
    if len(records) != 1:
        raise ValueError(f"reparse_chain_count={len(records)}")
    return records[0].sequence, len(records[0].residues)




def process_candidate(cand, cache_dir: Path, domains_dir: Path):
    """Return (record, None) or (None, failure_reason)."""
    domain_id = cand["domain_id"]
    out_path = domains_dir / f"{domain_id}.cif"
    try:
        if out_path.exists():
            records = parse_cif(out_path)
            if len(records) != 1:
                return None, "reparse_chain_count"
            seq, n = records[0].sequence, len(records[0].residues)
            expected = cand["estimated_length"]
            if expected is not None and abs(n - expected) > COUNT_TOLERANCE:
                raise ValueError("residue_count_mismatch")
            if not (MIN_LEN <= n <= MAX_LEN):
                return None, "parsed_length_out_of_range"
            if not set(seq) <= STANDARD_AA:
                return None, "nonstandard_aa"
            return _record(cand, seq, n, _manifest_segments(cand["segments"], records[0].residues)), None
        entry = download_entry(cand["pdb_id"], cache_dir)
        records = parse_cif(entry)
        if not records:
            return None, "parse_failed_no_chain"
        selected, expected, chain = select_residues(records, cand["segments"])
        if not selected:
            return None, "chain_or_segment_not_found"
        if abs(len(selected) - expected) > COUNT_TOLERANCE:
            return None, "residue_count_mismatch"
        if not (MIN_LEN <= len(selected) <= MAX_LEN):
            return None, "parsed_length_out_of_range"
        seq = "".join(r.aa for r in selected)
        if not set(seq) <= STANDARD_AA:
            return None, "nonstandard_aa"
        write_domain_cif(out_path, domain_id, cand["pdb_id"], chain, selected)
        seq2, n2 = validate_sliced(out_path)
        if n2 != len(selected) or seq2 != seq:
            out_path.unlink(missing_ok=True)
            return None, "reparse_mismatch"
        return _record(cand, seq2, n2, _manifest_segments(cand["segments"], selected)), None
    except RuntimeError as e:
        return None, str(e).split(":")[0]
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "residue_count_mismatch" in msg:
            return None, "residue_count_mismatch"
        if "reparse_chain_count" in msg:
            return None, "reparse_chain_count"
        return None, f"error:{type(e).__name__}:{msg}"[:200]


def _manifest_segments(segments, selected):
    out = []
    for seg in segments:
        s = {"chain": seg["chain"], "start": seg["start"], "start_ins": seg["start_ins"],
             "end": seg["end"], "end_ins": seg["end_ins"]}
        if s["start"] is None and selected:
            first = _key(selected[0].auth_seq_id, selected[0].ins_code)
            last = _key(selected[-1].auth_seq_id, selected[-1].ins_code)
            if first and last:
                s.update(start=first[0], start_ins=first[1], end=last[0], end_ins=last[1])
        out.append(s)
    return out


def _record(cand, seq, n, segments):
    return {
        "domain_id": cand["domain_id"],
        "pdb_id": cand["pdb_id"],
        "auth_chain": cand["segments"][0]["chain"],
        "segments": segments,
        "scop_sid": cand["scop_sid"],
        "family_id": cand["family_id"],
        "superfamily_id": cand["superfamily_id"],
        "fold_id": cand["fold_id"],
        "sequence": seq,
        "n_residues": n,
        "source_entry": cand["pdb_id"],
    }


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--pool-size", type=int, default=POOL_SIZE)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--v3-dir", type=Path, default=ROOT / "v3")
    args = ap.parse_args(argv)

    v3 = args.v3_dir
    cache_dir, domains_dir = v3 / ".cache_entries", v3 / "domains"
    cache_dir.mkdir(parents=True, exist_ok=True)
    domains_dir.mkdir(parents=True, exist_ok=True)

    cla_path = cache_dir / CLA_NAME
    if not cla_path.exists():
        req = urllib.request.Request(CLA_URL, headers={"User-Agent": "mmcif-vector-db-v3-build/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            cla_path.write_bytes(resp.read())
    download_date = datetime.date.fromtimestamp(cla_path.stat().st_mtime).isoformat()

    rows, bad_cla = parse_cla(cla_path)
    candidates, counts = sample_candidates(rows, seed=args.seed, pool_size=args.pool_size)
    counts["dropped_unparseable_cla_rows"] = bad_cla

    failures, records = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_candidate, c, cache_dir, domains_dir): c for c in candidates}
        for fut in concurrent.futures.as_completed(futs):
            rec, reason = fut.result()
            if rec is not None:
                records.append(rec)
            else:
                failures.append({"domain_id": futs[fut]["domain_id"], "reason": reason})

    # Re-apply the retrieval floor on validated domains (declared a priori).
    sf_counts = collections.Counter(r["superfamily_id"] for r in records)
    kept, dropped_sf = [], 0
    for r in records:
        if sf_counts[r["superfamily_id"]] >= MIN_PER_SUPERFAMILY:
            kept.append(r)
        else:
            dropped_sf += 1
            (domains_dir / f"{r['domain_id']}.cif").unlink(missing_ok=True)
    records = sorted(kept, key=lambda r: r["domain_id"])
    failures.sort(key=lambda f: f["domain_id"])
    fail_counts = collections.Counter(f["reason"].split(":")[0] for f in failures)

    metadata = {
        "source": SOURCE_NAME,
        "source_version": SOURCE_VERSION,
        "source_url": CLA_URL,
        "source_file": CLA_NAME,
        "download_date": download_date,
        "entry_url_template": ENTRY_URL,
        "seed": args.seed,
        "substitution_from_cath": False,
        "notes": ("SCOPe dir.cla.scope 2.01-stable used as specified; no CATH "
                  "substitution. SCOPe CLA provides no sequence, so sequence "
                  "validation against a source expectation is not applicable; "
                  "sequences are the parsed sliced-domain sequences."),
        "filters": {
            "unit": "SCOPe domain (defined residue segments, not whole entries)",
            "single_auth_chain_only": True,
            "min_length": MIN_LEN, "max_length": MAX_LEN,
            "length_basis": "segment-estimated length pre-filter; parsed sliced length enforced",
            "cap_per_superfamily": CAP_PER_SUPERFAMILY,
            "min_domains_per_superfamily_eligible": MIN_PER_SUPERFAMILY,
            "min_domains_per_superfamily_validated_reapplied": MIN_PER_SUPERFAMILY,
            "standard_aa_only": sorted(STANDARD_AA),
            "residue_count_tolerance_vs_expected_segment_length": COUNT_TOLERANCE,
            "candidate_pool_size": args.pool_size,
            "sampling": "seeded shuffle (random.Random(seed)) of sorted superfamily ids, then of sorted domain ids within each superfamily; superfamilies contributed whole up to cap until pool size reached",
        },
        "counts": {
            **{k: int(v) for k, v in sorted(counts.items())},
            "attempted_candidates": len(candidates),
            "validated_before_superfamily_floor": int(sum(sf_counts.values())),
            "dropped_validated_superfamily_below_min": dropped_sf,
            "final_domains": len(records),
            "final_superfamilies": len({r["superfamily_id"] for r in records}),
            "final_folds": len({r["fold_id"] for r in records}),
            "final_unique_pdb_entries": len({r["pdb_id"] for r in records}),
            "failures_by_reason": dict(sorted(fail_counts.items())),
        },
    }
    def _dir_bytes(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    metadata["resumability"] = (
        "Re-running skips cached CLA/entry CIFs and re-validates/reuses "
        "existing v3/domains/<domain_id>.cif files (existence + re-parse "
        "check) instead of re-slicing; entry cache may be deleted after a "
        "completed run and resumability still holds from sliced outputs "
        "alone, except failed candidates would need their entry again. "
        "Entries are parsed/sliced one at a time per worker; no bulk load.")
    manifest = {"metadata": metadata, "domains": records, "failures": failures}
    out = v3 / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
    for _ in range(3):  # converge: footprint block itself changes manifest size
        metadata["footprint_bytes"] = {
            "domains_dir": _dir_bytes(domains_dir),
            "cache_entries_dir": _dir_bytes(cache_dir),
            "manifest": out.stat().st_size,
        }
        metadata["footprint_mb"] = {k: round(v / 1e6, 3) for k, v in metadata["footprint_bytes"].items()}
        out.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
    print(json.dumps(metadata["counts"], indent=1))
    print(f"footprint_mb: {metadata['footprint_mb']}")
    print(f"manifest: {out} ({len(records)} domains)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
