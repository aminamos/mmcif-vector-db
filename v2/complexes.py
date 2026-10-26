"""Complex layer for v2: chain families become the vocabulary.

A complex (here: all protein chains in one deposited mmCIF model) is
described by which families it contains and in how many copies, e.g.
haemoglobin = {globin-alpha x2, globin-beta x2}. Complexes with similar
recipes are grouped. Symmetry-generated assembly copies are not expanded;
this describes the chains present in the file.

Robustness choices:
- copy numbers come from un-deduped chains, so stoichiometry survives
- a chain with no family gets a token from its own sequence, so identical
  lone proteins still match across different complexes
- two complexes group only if both their family sets overlap and their
  copy-number profiles agree (not either one alone)
"""
from __future__ import annotations
import hashlib
from collections import Counter, defaultdict
from .core import load_chains, seq_related

def _solo_token(seq: str) -> str:
    return "solo:" + hashlib.md5(seq.encode()).hexdigest()[:10]

def family_tokens(chains, families) -> dict:
    """Map every raw chain uid+sequence occurrence to a family token."""
    rep_of = {}
    for n, fam in enumerate(families, 1):
        for c in fam:
            rep_of[c.sequence] = f"Family {n}" if len(fam) > 1 else _solo_token(c.sequence)
    tokens = []
    for c in chains:
        if c.sequence in rep_of:
            tokens.append(rep_of[c.sequence])
            continue
        # near-match to a family representative, else its own token
        best, best_score = None, 0.5
        for n, fam in enumerate(families, 1):
            if len(fam) < 2:
                continue
            s = max(seq_related(c.sequence, m.sequence) for m in fam)
            if s > best_score:
                best, best_score = f"Family {n}", s
        tokens.append(best or _solo_token(c.sequence))
    return tokens

def compositions(raw_chains, families) -> list[dict]:
    """One recipe per deposited file with >= 2 protein chains."""
    tokens = family_tokens(raw_chains, families)
    by_file: dict[str, dict] = {}
    for c, tok in zip(raw_chains, tokens):
        key = c.path or c.entry_id
        d = by_file.setdefault(key, {"entry_id": c.entry_id, "path": c.path,
                                     "title": c.title, "recipe": Counter(), "chains": []})
        d["recipe"][tok] += 1
        d["chains"].append(c.uid)
    return sorted((d for d in by_file.values() if len(d["chains"]) >= 2),
                  key=lambda d: d["entry_id"])

def _similar(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    cosine = dot / (na * nb) if na and nb else 0.0
    jaccard = len(set(a) & set(b)) / max(len(keys), 1)
    return cosine if jaccard >= 0.5 else 0.0

def find_complex_groups(comps: list[dict]) -> list[list[dict]]:
    """Group complexes whose recipes agree (cosine >= 0.8 after the
    family-overlap check in _similar). Loners stay alone."""
    neighbours = defaultdict(set)
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if _similar(comps[i]["recipe"], comps[j]["recipe"]) >= 0.8:
                neighbours[i].add(j)
                neighbours[j].add(i)
    seen, groups = set(), []
    for i in range(len(comps)):
        if i in seen:
            continue
        stack, group = [i], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            group.append(comps[x])
            stack.extend(neighbours[x] - seen)
        groups.append(group)
    return sorted(groups, key=len, reverse=True)

def describe(comp: dict) -> str:
    parts = [f"{tok} x{n}" if n > 1 else tok for tok, n in sorted(comp["recipe"].items())]
    return ", ".join(parts)

def build_complexes(paths, families):
    raw = load_chains(paths, dedupe=False, min_len=5)
    comps = compositions(raw, families)
    return comps, find_complex_groups(comps)
