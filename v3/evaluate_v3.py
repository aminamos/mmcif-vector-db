#!/usr/bin/env python
"""v3 encoder evaluation against exhaustive Foldseek gold (7-point protocol).

Protocol implemented here
--------------------------
* The unit set is fixed by the manifest (SCOPe domains).  Every unordered
  pair of distinct domains is a potential pair; self-pairs are always out.
* Gold comes from an exhaustive Foldseek all-vs-all TSV with columns
  ``query,target,alntmscore,qtmscore,ttmscore,bits,evalue``.  The symmetric
  gold score for a pair is ``gold_tm = mean(qtmscore, ttmscore)`` (protocol
  point 4).  Duplicate rows for an unordered pair (both search directions)
  are de-duplicated keeping the row with the largest ``bits``.
* Missing gold is NEVER imputed (point 3): pairs absent from the TSV are
  not covered, are excluded from every correlation / classification metric,
  and are only reported via coverage counts.
* Headline analysis excludes pairs whose domains share a PDB entry
  (``pdb_id``); a sensitivity analysis including them is reported second
  (point 7).
* Pearson AND Spearman are reported (point 5).  Spearman uses average
  (mid) ranks for ties and is implemented here with numpy only.
* Confidence intervals are by bootstrap over PROTEINS (domains), not
  pairs (point 6): the domain list participating in the evaluated headline
  pairs is resampled with replacement (size = number of such domains,
  seeded RNG).  For a resample with domain multiplicities ``c_i``, every
  evaluated pair ``(i, j)`` contributes weight ``c_i * c_j``; instance
  pairs mapping to the same original domain have weight contributions
  only through pairs of distinct original domains, i.e. a pair of
  resampled instances of one original domain is never formed.  Pearson
  and weighted Spearman (weights enter through weighted midranks) are
  recomputed on the weighted pair set.  The 95% CI is the 2.5/97.5
  percentile of the valid resamples.  The same resamples are used for
  Pearson and Spearman.
* Retrieval: superfamily top-1 with leave-entry-out - candidates from the
  query's own PDB entry are excluded; queries with no cross-entry
  same-superfamily candidate (among domains with vectors) are excluded
  and counted, never scored as misses.
* Classification: same-superfamily (per the manifest) is the binary label
  over evaluated headline pairs, scored by cosine similarity.  AUROC is
  the Mann-Whitney rank statistic with average ranks for ties.  AUPR is
  the step-wise area under the precision-recall curve where all pairs
  sharing a score enter together (ties handled as one threshold step).

Memory behaviour (n = domains, P = covered pairs, d = vector dim)
------------------------------------------------------------------
Peak memory is O(n*d + P + chunk), never a dense n x n matrix:

* The gold TSV is streamed row by row.  De-duplication keeps one float64
  gold array, one float32 bits array and one bool covered array indexed
  by unordered-pair rank: at n=2500 (P_universe ~= 3.1M) that is ~40 MB,
  released after the covered-pair arrays are extracted.
* All downstream pair handling uses flat arrays over covered pairs only.
* Cosine similarities for pairs are computed in chunks of at most
  ``_PAIR_CHUNK`` pairs from the n x d vector matrix.
* Top-1 retrieval scores one query at a time (an n-vector per query).
* Bootstrap reuses the precomputed evaluated-pair arrays; per resample it
  allocates only O(P_evaluated) weight/rank scratch (~8 float64 arrays,
  ~200 MB worst case at 3.1M evaluated pairs), well within 16 GB.

No file is re-read per bootstrap resample or per encoder except each
encoder's own NPZ.
"""
from __future__ import annotations

import argparse
import csv
import json
import math

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PAIR_CHUNK = 262_144



# --------------------------------------------------------------------------
# Data containers / loaders
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainInfo:
    domain_id: str
    pdb_id: str
    superfamily_id: object  # int (SCOPe) or str; only equality is used


@dataclass
class GoldPairs:
    """Covered unordered pairs (domain positions into ``domain_ids``)."""
    pair_i: np.ndarray
    pair_j: np.ndarray
    gold_tm: np.ndarray
    stats: dict


def _get(record: dict, *names, default=None):
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    # tolerate one level of nesting (e.g. {"scop": {...}})
    for value in record.values():
        if isinstance(value, dict):
            for name in names:
                if name in value and value[name] is not None:
                    return value[name]
    return default


def load_manifest(path) -> dict[str, DomainInfo]:
    """Load the v3 manifest: ``{"metadata": ..., "domains": [records]}``.

    Also tolerates a bare list of records or a ``{domain_id: record}``
    mapping.  Returns ``domain_id -> DomainInfo`` in file order.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("domains"), list):
        records = data["domains"]
    elif isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = []
        for key, rec in data.items():
            if isinstance(rec, dict):
                rec = {"domain_id": key, **rec}
                records.append(rec)
    else:
        raise ValueError(f"unrecognised manifest structure in {path}")
    domains: dict[str, DomainInfo] = {}
    for rec in records:
        domain_id = str(_get(rec, "domain_id", "id", "uid", "scop_sid", "sid"))
        pdb_id = str(_get(rec, "pdb_id", "pdb", "entry_id", "source_entry", "source_pdb"))
        sf = _get(rec, "superfamily_id", "superfamily", "sf_id", "scop_superfamily_id")
        if domain_id in domains:
            raise ValueError(f"duplicate domain_id in manifest: {domain_id}")
        domains[domain_id] = DomainInfo(domain_id=domain_id, pdb_id=pdb_id, superfamily_id=sf)
    return domains


def _normalise_gold_id(raw: str) -> str:
    name = str(raw).strip().replace("\\", "/").rsplit("/", 1)[-1]
    for suffix in (".cif", ".pdb", ".ent", ".mmcif"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _iter_gold_rows(path):
    """Stream the gold TSV, yielding (query, target, gold_tm, bits).

    ``gold_tm`` is the symmetrised mean(qtmscore, ttmscore) of that row.
    Rows that cannot be parsed are skipped (and counted by callers via
    the yielded ``None`` sentinel is avoided; malformed rows are simply
    not yielded).
    """
    with open(path, newline="", encoding="utf-8") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        has_header = "query" in sample.split("\n", 1)[0].lower() and "target" in sample.split("\n", 1)[0].lower()
        if has_header:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                try:
                    q = _normalise_gold_id(row["query"])
                    t = _normalise_gold_id(row["target"])
                    gold_tm = 0.5 * (float(row["qtmscore"]) + float(row["ttmscore"]))
                    bits = float(row["bits"]) if row.get("bits") not in (None, "") else -math.inf
                except (KeyError, TypeError, ValueError):
                    continue
                yield q, t, gold_tm, bits
        else:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                if len(row) < 7:
                    continue
                try:
                    gold_tm = 0.5 * (float(row[3]) + float(row[4]))
                    bits = float(row[5])
                except ValueError:
                    continue
                yield _normalise_gold_id(row[0]), _normalise_gold_id(row[1]), gold_tm, bits


def load_gold(path) -> dict[tuple[str, str], float]:
    """Load gold as ``{(a, b): gold_tm}`` with a < b (small inputs/tests).

    Symmetrises each row, de-duplicates by max ``bits``, drops self-pairs.
    Production evaluation should use :func:`load_gold_pairs`, which
    streams without materialising a Python dict per pair.
    """
    best: dict[tuple[str, str], tuple[float, float]] = {}
    for q, t, gold_tm, bits in _iter_gold_rows(path):
        if q == t:
            continue
        key = (q, t) if q < t else (t, q)
        prev = best.get(key)
        if prev is None or bits > prev[1]:
            best[key] = (gold_tm, bits)
    return {k: v[0] for k, v in best.items()}


def _pair_rank(i: int, j: int, n: int) -> int:
    """Rank of unordered pair (i, j), i < j, in row-major enumeration."""
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def load_gold_pairs(path, domain_ids: list[str]) -> GoldPairs:
    """Stream the gold TSV into flat arrays over covered pairs only.

    De-duplication state is three numpy arrays indexed by pair rank
    (see module docstring for the memory accounting); the returned
    arrays contain exactly the covered distinct-domain pairs.
    """
    n = len(domain_ids)
    index = {d: k for k, d in enumerate(domain_ids)}
    total_pairs = n * (n - 1) // 2
    gold_all = np.full(total_pairs, np.nan, dtype=np.float64)
    bits_all = np.full(total_pairs, -np.inf, dtype=np.float32)
    covered = np.zeros(total_pairs, dtype=bool)
    stats = {"n_rows": 0, "n_rows_self_ignored": 0, "n_rows_unknown_domain_ignored": 0,
             "n_duplicate_rows": 0}
    for q, t, gold_tm, bits in _iter_gold_rows(path):
        stats["n_rows"] += 1
        if q == t:
            stats["n_rows_self_ignored"] += 1
            continue
        i = index.get(q)
        j = index.get(t)
        if i is None or j is None:
            stats["n_rows_unknown_domain_ignored"] += 1
            continue
        if i > j:
            i, j = j, i
        rank = _pair_rank(i, j, n)
        if covered[rank]:
            stats["n_duplicate_rows"] += 1
            if bits <= bits_all[rank]:
                continue
        covered[rank] = True
        bits_all[rank] = bits
        gold_all[rank] = gold_tm
    # Extract covered pairs, scanning pair ranks row by row (bounded slices).
    i_parts, j_parts, g_parts = [], [], []
    for i in range(n):
        start = _pair_rank(i, i + 1, n) if i + 1 < n else total_pairs
        stop = start + (n - i - 1)
        if stop <= start:
            continue
        cov = covered[start:stop]
        if not cov.any():
            continue
        js = np.nonzero(cov)[0] + (i + 1)
        i_parts.append(np.full(js.shape, i, dtype=np.int64))
        j_parts.append(js.astype(np.int64))
        g_parts.append(gold_all[start:stop][cov])
    del gold_all, bits_all, covered
    if i_parts:
        pair_i = np.concatenate(i_parts)
        pair_j = np.concatenate(j_parts)
        gold_tm = np.concatenate(g_parts)
    else:
        pair_i = np.empty(0, dtype=np.int64)
        pair_j = np.empty(0, dtype=np.int64)
        gold_tm = np.empty(0, dtype=np.float64)
    stats["n_pairs_covered_distinct"] = int(pair_i.size)
    return GoldPairs(pair_i=pair_i, pair_j=pair_j, gold_tm=gold_tm, stats=stats)


_STRUCTURED_UID_KEYS = ("uids", "uid", "ids", "domain_ids", "domain_id")
_STRUCTURED_VEC_KEYS = ("vectors", "embeddings", "matrix", "X", "vecs")


def load_vectors(path) -> dict[str, np.ndarray]:
    """Load an encoder NPZ: ``uid -> 1-D float64 vector``.

    Supports per-uid arrays (repo convention; ``*__latents`` keys are
    skipped) and a structured layout with a uid array plus a 2-D vector
    matrix under the keys listed in ``_STRUCTURED_*_KEYS``.
    """
    with np.load(path, allow_pickle=False) as z:
        files = list(z.files)
        uid_key = next((k for k in files if k in _STRUCTURED_UID_KEYS), None)
        vec_key = next((k for k in files if k in _STRUCTURED_VEC_KEYS), None)
        if uid_key is not None and vec_key is not None:
            uids = [u.decode() if isinstance(u, bytes) else str(u) for u in np.asarray(z[uid_key]).ravel()]
            mat = np.asarray(z[vec_key], dtype=np.float64)
            if mat.ndim != 2 or mat.shape[0] != len(uids):
                raise ValueError(f"{path}: vector matrix shape {mat.shape} does not match {len(uids)} uids")
            return {u: mat[k].copy() for k, u in enumerate(uids)}
        out = {}
        for k in files:
            if k.endswith("__latents") or k in _STRUCTURED_UID_KEYS:
                continue
            out[k] = np.asarray(z[k], dtype=np.float64).ravel().copy()
        return out


# --------------------------------------------------------------------------
# Metrics (numpy only)
# --------------------------------------------------------------------------

def rank_average(x) -> np.ndarray:
    """1-based ranks with average (mid) ranks for ties."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)  # mean of ranks i+1 .. j
        i = j
    return ranks


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    dx = x - x.mean()
    dy = y - y.mean()
    denom = math.sqrt(float(dx @ dx) * float(dy @ dy))
    if denom == 0.0:
        return float("nan")
    return float(dx @ dy / denom)


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    return pearson(rank_average(x), rank_average(y))


def auroc(labels, scores) -> float:
    """Mann-Whitney AUROC; average ranks handle ties. NaN if one class."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rank_average(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def aupr(labels, scores) -> float:
    """Step-wise AUPR; all pairs sharing a score enter as one step."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    if n_pos == 0 or labels.size == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = fp = 0
    prev_recall = 0.0
    area = 0.0
    i = 0
    while i < sorted_scores.size:
        j = i + 1
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        group = sorted_labels[i:j]
        tp += int(group.sum())
        fp += int(group.size - group.sum())
        recall = tp / n_pos
        area += (recall - prev_recall) * (tp / (tp + fp))
        prev_recall = recall
        i = j
    return float(area)


# --------------------------------------------------------------------------
# Protein (domain) bootstrap
# --------------------------------------------------------------------------

def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    total = float(w.sum())
    if total <= 0.0:
        return float("nan")
    mx = float(w @ x) / total
    my = float(w @ y) / total
    dx = x - mx
    dy = y - my
    sxx = float(w @ (dx * dx))
    syy = float(w @ (dy * dy))
    if sxx <= 0.0 or syy <= 0.0:
        return float("nan")
    return float(w @ (dx * dy) / math.sqrt(sxx * syy))


def _weighted_midrank_groups(values: np.ndarray):
    """Group indices (np.unique inverse + group count) for weighted ranks."""
    _uniq, inverse = np.unique(values, return_inverse=True)
    return inverse, int(inverse.max()) + 1 if inverse.size else 0


def _weighted_spearman(x, y, w, x_inv, x_groups, y_inv, y_groups) -> float:
    def midranks(inv, groups):
        gw = np.bincount(inv, weights=w, minlength=groups)
        before = np.cumsum(gw) - gw
        group_rank = before + (gw + 1.0) / 2.0
        return group_rank[inv]
    return _weighted_pearson(midranks(x_inv, x_groups), midranks(y_inv, y_groups), w)


def bootstrap_correlations(pair_i, pair_j, n_domains_resampled: int, x, y,
                           n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """Protein-bootstrap CIs for Pearson and Spearman (see module docstring).

    ``pair_i``/``pair_j`` are positions into the resampled domain list
    (0..n_domains_resampled-1).  Arrays are reused across resamples.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pair_i = np.asarray(pair_i)
    pair_j = np.asarray(pair_j)
    rng = np.random.default_rng(seed)
    x_inv, x_groups = _weighted_midrank_groups(x)
    y_inv, y_groups = _weighted_midrank_groups(y)
    pearson_vals, spearman_vals = [], []
    for _ in range(int(n_bootstrap)):
        sampled = rng.integers(0, n_domains_resampled, size=n_domains_resampled)
        counts = np.bincount(sampled, minlength=n_domains_resampled).astype(np.float64)
        w = counts[pair_i] * counts[pair_j]
        if w.sum() <= 0.0:
            continue
        r_p = _weighted_pearson(x, y, w)
        r_s = _weighted_spearman(x, y, w, x_inv, x_groups, y_inv, y_groups)
        if math.isfinite(r_p):
            pearson_vals.append(r_p)
        if math.isfinite(r_s):
            spearman_vals.append(r_s)

    def ci(vals):
        if not vals:
            return [None, None]
        lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
        return [float(lo), float(hi)]

    return {
        "pearson_ci95": ci(pearson_vals),
        "spearman_ci95": ci(spearman_vals),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid_pearson": len(pearson_vals),
        "n_bootstrap_valid_spearman": len(spearman_vals),
        "seed": int(seed),
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _pair_counts_by_group(keys: np.ndarray) -> int:
    """Sum of C(count, 2) over groups (pairs inside the same group)."""
    if keys.size == 0:
        return 0
    _u, counts = np.unique(keys, return_counts=True)
    return int((counts * (counts - 1) // 2).sum())


def _cosine_for_pairs(mat: np.ndarray, pair_i, pair_j) -> np.ndarray:
    """Cosine similarity for pairs whose endpoints are present, in chunks.

    Rows of ``mat`` for absent domains are zero; callers must mask pairs
    with absent endpoints before using the result (zero rows would give
    a spurious 0.0, which must never enter a metric).
    """
    out = np.empty(pair_i.size, dtype=np.float64)
    for start in range(0, pair_i.size, _PAIR_CHUNK):
        stop = min(start + _PAIR_CHUNK, pair_i.size)
        a = mat[pair_i[start:stop]]
        b = mat[pair_j[start:stop]]
        out[start:stop] = np.einsum("ij,ij->i", a, b)
    return out


def _stratum_metrics(cos: np.ndarray, gold_tm: np.ndarray) -> dict:
    return {
        "n_pairs": int(cos.size),
        "pearson": _clean(pearson(cos, gold_tm)),
        "spearman": _clean(spearman(cos, gold_tm)),
    }




def evaluate_encoder(name, vectors, domain_ids, domains: dict[str, DomainInfo],
                     gold: GoldPairs, headline_mask: np.ndarray,
                     n_universe_headline: int, n_universe_all: int,
                     gold_coverage_headline: dict, *,
                     seed: int = 42, n_bootstrap: int = 1000) -> dict:
    n = len(domain_ids)
    index = {d: k for k, d in enumerate(domain_ids)}
    pdb_arr = np.array([domains[d].pdb_id for d in domain_ids], dtype=object)
    sf_arr = np.array([domains[d].superfamily_id for d in domain_ids], dtype=object)

    # Vector matrix aligned to domain order; missing/invalid -> absent.
    present = np.zeros(n, dtype=bool)
    dim = 0
    mat = np.zeros((0, 0), dtype=np.float64)
    vec_dim = None
    for d, v in vectors.items():
        v = np.asarray(v, dtype=np.float64).ravel()
        if vec_dim is None:
            vec_dim = int(v.size)
        elif v.size != vec_dim:
            raise ValueError(f"encoder {name}: inconsistent vector dims ({v.size} != {vec_dim})")
    if vec_dim is not None:
        dim = vec_dim
        mat = np.zeros((n, vec_dim), dtype=np.float64)
        for d, v in vectors.items():
            k = index.get(d)
            if k is None:
                continue
            v = np.asarray(v, dtype=np.float64).ravel()
            norm = float(np.linalg.norm(v))
            if norm > 0.0 and np.isfinite(v).all():
                mat[k] = v / norm
                present[k] = True
    n_in_manifest = int(sum(1 for d in vectors if d in index))

    result: dict = {
        "name": name,
        "dim": dim,
        "n_vectors_total": int(len(vectors)),
        "n_vectors_in_manifest": n_in_manifest,
        "vector_coverage": {
            "n_domains_with_vector": int(present.sum()),
            "n_domains": n,
            "fraction": float(present.mean()) if n else None,
        },
        # Gold coverage is encoder-independent: computed once over the
        # full headline universe and asserted identical for every encoder.
        "gold_coverage_headline": dict(gold_coverage_headline),
    }
    assert result["gold_coverage_headline"] == gold_coverage_headline

    if gold.pair_i.size == 0 or dim == 0:
        result.update({
            "n_pairs_evaluated_headline": 0, "pearson": None, "spearman": None,
            "bootstrap": None, "sensitivity": None, "retrieval_top1": None,
            "aupr": None, "auroc": None,
            "strata": {
                "candidate_high": _stratum_metrics(np.zeros(0), np.zeros(0)),
                "candidate_mid": _stratum_metrics(np.zeros(0), np.zeros(0)),
                "same_superfamily": _stratum_metrics(np.zeros(0), np.zeros(0)),
            },
        })
        return result

    both = present[gold.pair_i] & present[gold.pair_j]
    cos_all = _cosine_for_pairs(mat, gold.pair_i, gold.pair_j)

    # ---- headline (cross-entry) evaluated pairs -------------------------
    eval_mask = headline_mask & both
    ei, ej = gold.pair_i[eval_mask], gold.pair_j[eval_mask]
    ex, ey = cos_all[eval_mask], gold.gold_tm[eval_mask]
    labels = (sf_arr[ei] == sf_arr[ej]) if ei.size else np.zeros(0, dtype=bool)

    boot = None
    if ei.size >= 2:
        # Resample over domains in evaluated pairs; map both endpoints
        # into one shared resample index space.
        all_eval = np.unique(np.concatenate([ei, ej]))
        pos = {int(d): k for k, d in enumerate(all_eval)}
        bi = np.array([pos[int(v)] for v in ei], dtype=np.int64)
        bj = np.array([pos[int(v)] for v in ej], dtype=np.int64)
        boot = bootstrap_correlations(bi, bj, int(all_eval.size), ex, ey,
                                      n_bootstrap=n_bootstrap, seed=seed)
    result.update({
        "n_pairs_evaluated_headline": int(ei.size),
        "pearson": _clean(pearson(ex, ey)),
        "spearman": _clean(spearman(ex, ey)),
        "bootstrap": boot,
        "aupr": _clean(aupr(labels, ex)),
        "auroc": _clean(auroc(labels, ex)),
        "n_pairs_same_superfamily_headline": int(labels.sum()) if ei.size else 0,
        "n_pairs_diff_superfamily_headline": int(labels.size - labels.sum()) if ei.size else 0,
    })
    # ---- strata over headline evaluated pairs (no bootstrap) -----------
    # Subsets of the exact headline pair arrays above; computing them does
    # not mutate ex/ey, so headline numbers are unchanged.
    result["strata"] = {
        "candidate_high": _stratum_metrics(ex[ey >= 0.5], ey[ey >= 0.5]),
        "candidate_mid": _stratum_metrics(ex[ey >= 0.3], ey[ey >= 0.3]),
        "same_superfamily": _stratum_metrics(ex[labels], ey[labels]),
    }

    # ---- sensitivity (including same-entry pairs) -----------------------
    sens_mask = both
    sx, sy = cos_all[sens_mask], gold.gold_tm[sens_mask]
    result["sensitivity"] = {
        "n_pairs_evaluated": int(sx.size),
        "n_pairs_universe": int(n_universe_all),
        "n_pairs_gold_covered": int(gold.pair_i.size),
        "pearson": _clean(pearson(sx, sy)),
        "spearman": _clean(spearman(sx, sy)),
    }

    # ---- superfamily top-1, leave-entry-out retrieval --------------------
    # One query at a time: scores is an n-vector, never an n x n matrix.
    n_correct = n_queries = 0
    excl_no_family = excl_no_vector = excl_no_candidate = 0
    if dim:

        for q in range(n):
            if not present[q]:
                excl_no_vector += 1
                continue
            scores = mat @ mat[q]
            cand = present.copy()
            cand[q] = False
            cand &= pdb_arr != pdb_arr[q]
            if not cand.any():
                excl_no_candidate += 1
                continue
            if not (sf_arr[cand] == sf_arr[q]).any():
                excl_no_family += 1
                continue
            masked = np.where(cand, scores, -np.inf)
            top = int(np.argmax(masked))
            n_queries += 1
            n_correct += int(sf_arr[top] == sf_arr[q])
    result["retrieval_top1"] = {
        "accuracy": (n_correct / n_queries) if n_queries else None,
        "n_correct": int(n_correct),
        "n_queries": int(n_queries),
        "n_excluded_no_cross_entry_family_member": int(excl_no_family),
        "n_excluded_no_vector": int(excl_no_vector),
        "n_excluded_no_cross_entry_candidate": int(excl_no_candidate),
    }
    return result


def _clean(value):
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _gold_summary(gold_tm: np.ndarray, n_covered: int, n_universe: int) -> dict:
    if gold_tm.size:
        pct = np.percentile(gold_tm, [5, 25, 50, 75, 95])
        frac07 = float((gold_tm >= 0.7).mean())
        mean = float(gold_tm.mean())
        pct_d = {str(k): float(v) for k, v in zip((5, 25, 50, 75, 95), pct)}
    else:
        frac07 = mean = None
        pct_d = {str(k): None for k in (5, 25, 50, 75, 95)}
    return {
        "n_pairs_total": int(n_universe),
        "n_pairs_covered": int(n_covered),
        "coverage_fraction": (n_covered / n_universe) if n_universe else None,
        "coverage_percent": (100.0 * n_covered / n_universe) if n_universe else None,
        "fraction_tm_ge_0.7": frac07,
        "mean_tm": mean,
        "tm_percentiles": pct_d,
    }


def evaluate(domains: dict[str, DomainInfo], gold, encoders: dict[str, dict],
             *, seed: int = 42, n_bootstrap: int = 1000) -> dict:
    """Full evaluation. ``gold`` is a GoldPairs or a {(a, b): gold_tm} dict."""
    domain_ids = sorted(domains)
    n = len(domain_ids)
    if isinstance(gold, dict):
        index = {d: k for k, d in enumerate(domain_ids)}
        items = [(index[a], index[b], v) for (a, b), v in gold.items()
                 if a in index and b in index and a != b]
        items.sort()
        gold = GoldPairs(
            pair_i=np.array([t[0] for t in items], dtype=np.int64),
            pair_j=np.array([t[1] for t in items], dtype=np.int64),
            gold_tm=np.array([t[2] for t in items], dtype=np.float64),
            stats={"n_pairs_covered_distinct": len(items)},
        )
    pdb_arr = np.array([domains[d].pdb_id for d in domain_ids], dtype=object)
    headline_mask = pdb_arr[gold.pair_i] != pdb_arr[gold.pair_j]
    n_universe_all = n * (n - 1) // 2
    n_same_entry = _pair_counts_by_group(pdb_arr)
    n_universe_headline = n_universe_all - n_same_entry
    n_covered_headline = int(headline_mask.sum())

    gold_cov_headline = {
        "n_pairs_covered": n_covered_headline,
        "n_pairs_universe": int(n_universe_headline),
        "fraction": (n_covered_headline / n_universe_headline) if n_universe_headline else None,
    }
    gold_block = {
        "headline_excluding_same_entry": _gold_summary(
            gold.gold_tm[headline_mask], n_covered_headline, n_universe_headline),
        "including_same_entry": _gold_summary(gold.gold_tm, int(gold.pair_i.size), n_universe_all),
        "n_pairs_same_entry_universe": int(n_same_entry),
        "loader_stats": gold.stats,
    }

    encoder_results = []
    coverages = []
    for name, vectors in encoders.items():
        res = evaluate_encoder(name, vectors, domain_ids, domains, gold, headline_mask,
                               n_universe_headline, n_universe_all, gold_cov_headline,
                               seed=seed, n_bootstrap=n_bootstrap)
        coverages.append(res["gold_coverage_headline"])
        encoder_results.append(res)
    # Gold coverage is encoder-independent by construction (same gold,
    # same universe); assert it rather than letting vector missingness
    # leak into it.  Per-encoder differences live in vector_coverage and
    # n_pairs_evaluated_headline only.
    for cov in coverages:
        assert cov == gold_cov_headline, "gold coverage differs across encoders"

    return {
        "n_domains": n,
        "protocol": {
            "gold_symmetrization": "mean(qtmscore, ttmscore), dedupe by max bits",
            "missing_gold": "excluded, never imputed; reported as coverage",
            "headline": "cross-PDB-entry pairs only; sensitivity including same-entry pairs reported per encoder",
            "bootstrap": "over domains (proteins), 95% percentile CI, see module docstring",
            "seed": int(seed),
            "n_bootstrap": int(n_bootstrap),
        },
        "gold": gold_block,
        "encoders": encoder_results,
    }


# --------------------------------------------------------------------------
# Reporting / CLI
# --------------------------------------------------------------------------

def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _fmt_ci(ci):
    if not ci or ci[0] is None:
        return "[n/a, n/a]"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def render_markdown(results: dict) -> str:
    gold_h = results["gold"]["headline_excluding_same_entry"]
    lines = [
        "# v3 encoder evaluation",
        "",
        f"Domains: {results['n_domains']}  ",
        f"Gold (headline, cross-entry): {gold_h['n_pairs_covered']}/{gold_h['n_pairs_total']} pairs covered "
        f"({(gold_h['coverage_percent'] or 0.0):.1f}%), "
        f"fraction TM>=0.7: {_fmt(gold_h['fraction_tm_ge_0.7'])}",
        "",
        "| encoder | dim | coverage | Pearson [CI] | Spearman [CI] | top-1 | AUPR | AUROC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for enc in results["encoders"]:
        boot = enc.get("bootstrap") or {}
        p_ci = _fmt_ci(boot.get("pearson_ci95"))
        s_ci = _fmt_ci(boot.get("spearman_ci95"))
        cov = enc["gold_coverage_headline"]["fraction"]
        retr = enc.get("retrieval_top1") or {}
        lines.append(
            f"| {enc['name']} | {enc['dim']} | {(100.0 * cov if cov is not None else float('nan')):.1f}% "
            f"| {_fmt(enc.get('pearson'))} {p_ci} | {_fmt(enc.get('spearman'))} {s_ci} "
            f"| {_fmt(retr.get('accuracy'))} | {_fmt(enc.get('aupr'))} | {_fmt(enc.get('auroc'))} |"
        )
    lines += [
        "",
        "Strata (headline pairs only, no bootstrap):",
        "",
        "| encoder | stratum | n_pairs | Pearson | Spearman |",
        "|---|---|---|---|---|",
    ]
    for enc in results["encoders"]:
        strata = enc.get("strata") or {}
        for stratum in ("candidate_high", "candidate_mid", "same_superfamily"):
            s = strata.get(stratum) or {}
            lines.append(
                f"| {enc['name']} | {stratum} | {s.get('n_pairs', 0)} "
                f"| {_fmt(s.get('pearson'))} | {_fmt(s.get('spearman'))} |"
            )
    lines.append("")
    return "\n".join(lines)


def _parse_vector_args(values) -> dict[str, str]:
    out = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--vectors entry must be name=path, got: {value}")
        name, path = value.split("=", 1)
        out[name] = path
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--vectors", nargs="+", required=True, help="name=path.npz entries")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)

    domains = load_manifest(args.manifest)
    domain_ids = sorted(domains)
    gold = load_gold_pairs(args.gold, domain_ids)
    encoders = {name: load_vectors(path) for name, path in _parse_vector_args(args.vectors).items()}
    results = evaluate(domains, gold, encoders, seed=args.seed, n_bootstrap=args.n_bootstrap)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path = out.with_suffix(".md")
    md_path.write_text(render_markdown(results), encoding="utf-8")
    print(f"wrote {out} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
