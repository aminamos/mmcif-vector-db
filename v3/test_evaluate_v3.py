"""Synthetic tests for evaluate_v3 (no dataset / gold files required)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_v3 as ev


# ---------------------------------------------------------------- helpers

def make_domains(specs):
    """specs: list of (domain_id, pdb_id, superfamily_id)."""
    return {d: ev.DomainInfo(d, p, s) for d, p, s in specs}


def unit(angle_deg):
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)])


def write_gold(tmp_path, rows, name="gold.tsv"):
    path = tmp_path / name
    lines = ["query\ttarget\talntmscore\tqtmscore\tttmscore\tbits\tevalue"]
    for q, t, tm, bits in rows:
        lines.append(f"{q}\t{t}\t{tm}\t{tm}\t{tm}\t{bits}\t1e-5")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_manifest(tmp_path, specs):
    path = tmp_path / "manifest.json"
    data = {"metadata": {}, "domains": [
        {"domain_id": d, "pdb_id": p, "auth_chain": "A", "segments": [],
         "scop_sid": d, "family_id": s, "superfamily_id": s, "fold_id": 1,
         "sequence": "ACDE", "n_residues": 4, "source_entry": p}
        for d, p, s in specs]}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------- metrics

def test_pearson_spearman_hand_computed():
    x, y = [1.0, 2.0, 3.0], [1.0, 3.0, 2.0]
    assert ev.pearson(x, y) == pytest.approx(0.5)
    assert ev.spearman(x, y) == pytest.approx(0.5)
    assert ev.rank_average([10.0, 20.0, 20.0, 40.0]).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_perfect_monotone():
    x = np.array([0.1, 0.4, 0.9, 2.0, 5.0])
    y = x ** 3  # strictly monotone, not linear
    assert ev.spearman(x, y) == pytest.approx(1.0)
    assert ev.pearson(x, y) < 1.0
    assert ev.pearson(x, x) == pytest.approx(1.0)
    assert ev.spearman(x, -x) == pytest.approx(-1.0)


def test_gold_symmetrization_and_dedupe_by_max_bits(tmp_path):
    path = tmp_path / "g.tsv"
    path.write_text(
        "query\ttarget\talntmscore\tqtmscore\tttmscore\tbits\tevalue\n"
        "a.cif\tb.cif\t0.9\t0.8\t0.6\t50\t1e-5\n"   # mean = 0.7, bits 50
        "b\ta\t0.9\t1.0\t1.0\t10\t1e-5\n"           # mean = 1.0 but lower bits -> dropped
        "a\ta\t1\t1\t1\t99\t0\n",                    # self-pair dropped
        encoding="utf-8")
    gold = ev.load_gold(path)
    assert gold == {("a", "b"): pytest.approx(0.7)}


# ---------------------------------------------------------------- bootstrap

def _all_pairs(n):
    i, j = [], []
    for a in range(n):
        for b in range(a + 1, n):
            i.append(a)
            j.append(b)
    return np.array(i), np.array(j)


def test_bootstrap_ci_contains_point_and_seeded_reproducible():
    n = 8
    pi, pj = _all_pairs(n)
    rng = np.random.default_rng(0)
    gold = rng.random(pi.size)
    cos = gold.copy()  # identical scores -> every valid resample has r = 1
    out1 = ev.bootstrap_correlations(pi, pj, n, cos, gold, n_bootstrap=60, seed=42)
    out2 = ev.bootstrap_correlations(pi, pj, n, cos, gold, n_bootstrap=60, seed=42)
    assert out1 == out2  # seeded reproducibility
    assert out1["pearson_ci95"] == pytest.approx([1.0, 1.0])
    assert out1["spearman_ci95"] == pytest.approx([1.0, 1.0])
    point = ev.pearson(cos, gold)
    assert out1["pearson_ci95"][0] <= point <= out1["pearson_ci95"][1]


def test_bootstrap_noisy_ci_contains_point_estimate():
    n = 10
    pi, pj = _all_pairs(n)
    rng = np.random.default_rng(7)
    gold = rng.random(pi.size)
    cos = gold + rng.normal(0, 0.05, pi.size)
    out = ev.bootstrap_correlations(pi, pj, n, cos, gold, n_bootstrap=200, seed=3)
    point = ev.pearson(cos, gold)
    lo, hi = out["pearson_ci95"]
    assert lo <= point <= hi
    assert out["n_bootstrap_valid_pearson"] > 150


# ---------------------------------------------------------------- coverage

def test_missing_gold_pair_not_imputed_and_coverage_drops(tmp_path):
    # a,b,c on distinct entries; only a-b and a-c in gold, b-c missing.
    specs = [("a", "p1", 1), ("b", "p2", 1), ("c", "p3", 2)]
    domains = make_domains(specs)
    gold_path = write_gold(tmp_path, [("a", "b", 0.9, 50), ("a", "c", 0.2, 50)])
    gold = ev.load_gold_pairs(gold_path, sorted(domains))
    vecs = {"a": unit(0), "b": unit(10), "c": unit(80)}
    res = ev.evaluate(domains, gold, {"enc": vecs}, n_bootstrap=20)
    enc = res["encoders"][0]
    assert enc["gold_coverage_headline"] == {
        "n_pairs_covered": 2, "n_pairs_universe": 3, "fraction": pytest.approx(2 / 3)}
    assert enc["n_pairs_evaluated_headline"] == 2
    # Metric must equal the covered-subset-only computation.  Imputing the
    # missing b-c pair as gold=0 would change r (checked: it does here).
    covered_cos = [float(unit(0) @ unit(10)), float(unit(0) @ unit(80))]
    expected = ev.pearson(covered_cos, [0.9, 0.2])
    assert enc["pearson"] == pytest.approx(expected)
    imputed = ev.pearson(covered_cos + [float(unit(10) @ unit(80))], [0.9, 0.2, 0.0])
    assert enc["pearson"] != pytest.approx(imputed)


def test_vector_coverage_reported_separately_from_gold_coverage():
    specs = [("a", "p1", 1), ("b", "p2", 1), ("c", "p3", 2)]
    domains = make_domains(specs)
    gold = {("a", "b"): 0.9, ("a", "c"): 0.2, ("b", "c"): 0.1}
    full = {"a": unit(0), "b": unit(10), "c": unit(80)}
    partial = {"a": unit(0), "b": unit(10)}  # c missing
    res = ev.evaluate(domains, gold, {"full": full, "partial": partial}, n_bootstrap=10)
    by_name = {e["name"]: e for e in res["encoders"]}
    # Gold coverage identical across encoders (asserted in evaluate too).
    assert by_name["full"]["gold_coverage_headline"] == by_name["partial"]["gold_coverage_headline"]
    assert by_name["partial"]["vector_coverage"]["fraction"] == pytest.approx(2 / 3)
    assert by_name["partial"]["n_pairs_evaluated_headline"] == 1
    assert by_name["full"]["n_pairs_evaluated_headline"] == 3


def test_headline_excludes_same_entry_sensitivity_includes():
    specs = [("a", "p1", 1), ("b", "p1", 1), ("c", "p2", 2)]
    domains = make_domains(specs)
    gold = {("a", "b"): 0.95, ("a", "c"): 0.2, ("b", "c"): 0.1}
    vecs = {"a": unit(0), "b": unit(5), "c": unit(80)}
    res = ev.evaluate(domains, gold, {"enc": vecs}, n_bootstrap=10)
    enc = res["encoders"][0]
    assert enc["gold_coverage_headline"]["n_pairs_universe"] == 2
    assert enc["n_pairs_evaluated_headline"] == 2
    assert enc["sensitivity"]["n_pairs_evaluated"] == 3
    assert res["gold"]["n_pairs_same_entry_universe"] == 1


# ---------------------------------------------------------------- AUPR/AUROC

def test_aupr_auroc_perfect_separation():
    labels = [True, True, False, False]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert ev.auroc(labels, scores) == pytest.approx(1.0)
    assert ev.aupr(labels, scores) == pytest.approx(1.0)
    assert ev.auroc(labels, [-s for s in scores]) == pytest.approx(0.0)


def test_auroc_half_on_random_scores_seeded():
    rng = np.random.default_rng(42)
    labels = rng.random(2000) < 0.5
    scores = rng.random(2000)
    assert ev.auroc(labels, scores) == pytest.approx(0.5, abs=0.05)
    assert 0.3 < ev.aupr(labels, scores) < 0.7


def test_aupr_auroc_ties():
    # All scores tied: AUROC must be exactly 0.5, AUPR = prevalence.
    labels = [True, False, True, False]
    assert ev.auroc(labels, [0.5] * 4) == pytest.approx(0.5)
    assert ev.aupr(labels, [0.5] * 4) == pytest.approx(0.5)


# ---------------------------------------------------------------- retrieval

def test_top1_leave_entry_out_excludes_same_entry_only_mate():
    # q's only same-superfamily mate shares its PDB entry -> excluded.
    specs = [("q", "p1", 1), ("mate", "p1", 1), ("other", "p2", 2)]
    domains = make_domains(specs)
    gold = {("q", "mate"): 0.9, ("q", "other"): 0.1, ("mate", "other"): 0.1}
    vecs = {"q": unit(0), "mate": unit(1), "other": unit(90)}
    res = ev.evaluate(domains, gold, {"enc": vecs}, n_bootstrap=5)
    retr = res["encoders"][0]["retrieval_top1"]
    assert retr["n_queries"] == 0
    assert retr["accuracy"] is None
    assert retr["n_excluded_no_cross_entry_family_member"] == 3


def test_top1_hit_cross_entry():
    specs = [("q", "p1", 1), ("mate", "p2", 1), ("other", "p3", 2)]
    domains = make_domains(specs)
    gold = {("q", "mate"): 0.9, ("q", "other"): 0.1, ("mate", "other"): 0.1}
    vecs = {"q": unit(0), "mate": unit(3), "other": unit(90)}
    res = ev.evaluate(domains, gold, {"enc": vecs}, n_bootstrap=5)
    retr = res["encoders"][0]["retrieval_top1"]
    # q and mate both retrieve each other; "other" is a singleton family.
    assert retr["n_queries"] == 2
    assert retr["accuracy"] == pytest.approx(1.0)
    assert retr["n_excluded_no_cross_entry_family_member"] == 1


# ---------------------------------------------------------------- end-to-end CLI-ish

def test_evaluate_and_markdown_and_vector_npz(tmp_path):
    specs = [("a", "p1", 1), ("b", "p2", 1), ("c", "p3", 2), ("d", "p4", 2)]
    manifest = write_manifest(tmp_path, specs)
    gold_path = write_gold(tmp_path, [
        ("a", "b", 0.9, 50), ("a", "c", 0.2, 50), ("a", "d", 0.15, 50),
        ("b", "c", 0.25, 50), ("b", "d", 0.1, 50), ("c", "d", 0.85, 50)])
    npz = tmp_path / "enc.npz"
    np.savez(npz, a=unit(0), b=unit(8), c=unit(70), d=unit(78))
    rc = ev.main(["--manifest", str(manifest), "--gold", str(gold_path),
                  "--vectors", f"enc={npz}", "--out", str(tmp_path / "evaluation.json"),
                  "--n-bootstrap", "30"])
    assert rc == 0
    results = json.loads((tmp_path / "evaluation.json").read_text())
    enc = results["encoders"][0]
    assert enc["dim"] == 2
    assert enc["gold_coverage_headline"]["fraction"] == pytest.approx(1.0)
    assert enc["pearson"] > 0.9
    assert enc["retrieval_top1"]["accuracy"] == pytest.approx(1.0)
    md = (tmp_path / "evaluation.md").read_text()
    assert "| enc | 2 | 100.0% |" in md
    assert "Pearson [CI]" in md


def test_load_vectors_structured_layout(tmp_path):
    npz = tmp_path / "s.npz"
    np.savez(npz, uids=np.array(["a", "b"]), vectors=np.array([[1.0, 0.0], [0.0, 1.0]]))
    vecs = ev.load_vectors(npz)
    assert set(vecs) == {"a", "b"}
    assert vecs["a"].tolist() == [1.0, 0.0]
