"""CLI: ingest / similar / search / cluster / rag  (RAG-for-PDB+AFDB MVP)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .db import DB, search
from .parse import parse_cif
from .features import struct_vector

def _files(p):
    p = Path(p)
    return sorted(p.rglob("*.cif")) if p.is_dir() else [p]

def rag_context(hits):
    """LLM-ready context blocks from retrieved chains."""
    blocks = []
    for h in hits:
        blocks.append(
          f"[{h['uid']} | {h['source']} | cluster={h['cluster_id']} | score={h['score']} "
          f"seq={h['seq_score']} struct={h['struct_score']} text={h['text_score']}]\n"
          f"Title: {h['title'] or '(AFDB model)'}\nResidues: {h['n_res']}\nSequence: {h['sequence'][:200]}")
    return "\n\n".join(blocks)

def _load_ann(path):
    from .ann import ANNIndex
    try:
        return ANNIndex.load(path)
    except RuntimeError as e:
        raise SystemExit(str(e)) from e

def main(argv=None):
    ap = argparse.ArgumentParser(prog="mmcif-vec")
    ap.add_argument("--db", default="mmcif.db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("paths", nargs="+")
    b = sub.add_parser("build-ann"); b.add_argument("--out", required=True)
    b.add_argument("--num-partitions", type=int, default=None)
    b.add_argument("--num-sub-vectors", type=int, default=96)
    b.add_argument("--index-type", default="IVF_FLAT")
    b.add_argument("--nprobes", type=int, default=128)
    s = sub.add_parser("similar"); s.add_argument("cif"); s.add_argument("--k", type=int, default=5); s.add_argument("--ann")
    s = sub.add_parser("search"); s.add_argument("--text"); s.add_argument("--seq"); s.add_argument("--k", type=int, default=5); s.add_argument("--ann")
    s = sub.add_parser("cluster"); s.add_argument("--k", type=int, default=3)
    s = sub.add_parser("rag"); s.add_argument("text"); s.add_argument("--k", type=int, default=5)
    sub.add_parser("stats")
    a = ap.parse_args(argv)
    db = DB(a.db)
    if a.cmd == "ingest":
        total = sum(db.ingest(f) for p in a.paths for f in _files(p))
        print(f"ingested {total} chains"); print(json.dumps(db.stats()))
    elif a.cmd == "build-ann":
        from .ann import ANNIndex
        try:
            idx = ANNIndex.build(db, a.out, num_partitions=a.num_partitions,
                                 num_sub_vectors=a.num_sub_vectors,
                                 index_type=a.index_type, nprobes=a.nprobes)
        except RuntimeError as e:
            raise SystemExit(str(e)) from e
        print(json.dumps(idx.meta, indent=2))
    elif a.cmd == "similar":
        recs = parse_cif(a.cif)
        if not recs: raise SystemExit("no chains parsed")
        r = recs[0]
        ann = _load_ann(a.ann) if a.ann else None
        hits = search(db, q_seq=r.sequence, q_struct=struct_vector(r.ca_coords), k=a.k, ann=ann)
        print(json.dumps(hits, indent=2))
    elif a.cmd == "search":
        ann = _load_ann(a.ann) if a.ann else None
        print(json.dumps(search(db, q_seq=a.seq, q_text=a.text, k=a.k, ann=ann), indent=2))
    elif a.cmd == "cluster":
        from .cluster import cluster
        print(json.dumps(cluster(db, k=a.k), indent=2))
    elif a.cmd == "rag":
        hits = search(db, q_text=a.text, k=a.k)
        print(f"Question: {a.text}\n\nRetrieved context:\n{rag_context(hits)}")
    elif a.cmd == "stats":
        print(json.dumps(db.stats(), indent=2))

if __name__ == "__main__":
    main()
