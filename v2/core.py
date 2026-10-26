"""v2 core: no knobs for the user.

Fixes over v1, all automatic:
- exact-duplicate chains inside one entry are counted once
- relatedness is decided by neighbours (a similarity graph), not by
  forcing every protein into one of k round groups
- sequence comparison handles a short domain inside a long protein
  (e.g. a 101-residue PDB domain vs the 719-residue AFDB model)
- loners stay alone instead of being assigned to a family
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from mmcif_vec.parse import parse_cif
from mmcif_vec.features import struct_vector
from mmcif_vec.db import cosine

def iter_files(paths) -> list:
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.rglob("*.cif")) if p.is_dir() else [p]
    return files

def load_chains(paths, dedupe: bool = True, min_len: int = 20) -> list:
    """Parse files/dirs. dedupe=True drops duplicate chains within each
    entry (for family finding); dedupe=False keeps every copy (needed to
    count how many copies of a family a complex contains)."""
    seen, out = set(), []
    for f in iter_files(paths):
        for r in parse_cif(f):
            if len(r.sequence) < min_len:
                continue
            if dedupe:
                key = (r.entry_id, r.sequence)
                if key in seen:
                    continue
                seen.add(key)
            out.append(r)
    return out

def _kmers(s, k=3):
    return {s[i:i + k] for i in range(len(s) - k + 1)}

def seq_related(a: str, b: str) -> float:
    """0..1. Uses the better of shared-fraction and containment, so a
    domain counts as related to the full-length protein containing it."""
    A, B = _kmers(a), _kmers(b)
    if not A or not B:
        return 0.0
    shared = len(A & B)
    return max(shared / len(A | B), 0.85 * shared / min(len(A), len(B)))

def find_families(chains) -> list[list]:
    """Group chains that are related to at least one other chain.
    Returns families (largest first); every chain appears exactly once."""
    structs = [struct_vector(c.ca_coords) for c in chains]
    neighbours = defaultdict(set)
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            seq = seq_related(chains[i].sequence, chains[j].sequence)
            struct = cosine(structs[i], structs[j])
            strong_sequence = seq >= 0.50
            seq_and_shape = seq >= 0.10 and struct >= 0.90
            if strong_sequence or seq_and_shape:
                neighbours[i].add(j)
                neighbours[j].add(i)
    seen, families = set(), []
    for i in range(len(chains)):
        if i in seen:
            continue
        stack, group = [i], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            group.append(chains[x])
            stack.extend(neighbours[x] - seen)
        families.append(group)
    return sorted(families, key=len, reverse=True)

def representative(family):
    """The chain most related to the rest of its family (for viewing)."""
    if len(family) == 1:
        return family[0]
    best, best_score = family[0], -1.0
    for a in family:
        score = sum(seq_related(a.sequence, b.sequence) for b in family if b is not a)
        if score > best_score:
            best, best_score = a, score
    return best
