"""v2 user interface: one command, plain-English families out.

  uv run python -m v2 build samples/
  uv run python -m v2 build samples/ --out v2_out

No numbers to choose. Duplicate copies are removed, related proteins are
grouped, loners are left alone, and each family gets one representative
protein to look at. Writes families.txt, families.json, view_families.pml
(the .pml is only for the optional PyMOL viewer; PyMOL is NOT required).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .core import load_chains, find_families, representative
from .complexes import build_complexes, describe

def build(paths, out_dir) -> dict:
    chains = load_chains(paths)
    families = find_families(chains)
    comps, complex_groups = build_complexes(paths, families)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    data, lines, pml = [], [], ["# Optional: open in PyMOL to view one representative per family"]
    for n, fam in enumerate(families, 1):
        rep = representative(fam)
        label = f"Family {n}" if len(fam) > 1 else f"On its own {n}"
        names = [c.uid for c in fam]
        data.append({"family": n, "size": len(fam), "representative": rep.uid,
                     "title": rep.title, "source": rep.source, "members": names})
        lines.append(f"{label} ({len(fam)}): {', '.join(names)}")
        lines.append(f"  Look at: {rep.uid} — {rep.title or rep.source}")
        if rep.path:
            pml.append(f"load {rep.path}, fam{n}")
            pml.append(f"color marine, fam{n}")
    result = {"chains_after_dedupe": len(chains), "groups": len(data), "families": data}
    (out / "families.json").write_text(json.dumps(result, indent=2))
    (out / "families.txt").write_text("\n".join(lines) + "\n")
    (out / "view_families.pml").write_text("\n".join(pml) + "\nshow cartoon\n")
    print("\n".join(lines))
    # complex layer: recipes from families, grouped by recipe agreement
    cdata = [{"complex_group": n, "size": len(g),
              "members": [{"entry_id": c["entry_id"], "recipe": describe(c),
                           "chains": c["chains"]} for c in g]}
             for n, g in enumerate(complex_groups, 1)]
    (out / "complexes.json").write_text(json.dumps(cdata, indent=2))
    clines = []
    for n, g in enumerate(complex_groups, 1):
        head = f"Complex group {n}" if len(g) > 1 else f"Complex on its own {n}"
        clines.append(head + ": " + "; ".join(
            f"{c['entry_id']} ({describe(c)})" for c in g))
    (out / "complexes.txt").write_text("\n".join(clines) + "\n")
    print("\n" + ("\n".join(clines) if clines else "No multi-chain complexes found."))
    print(f"\nWrote {out}/families.txt, families.json, view_families.pml, complexes.txt, complexes.json")
    result["complex_groups"] = cdata
    return result

def main(argv=None):
    ap = argparse.ArgumentParser(prog="v2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("paths", nargs="+"); b.add_argument("--out", default="v2_out")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.paths, a.out)

if __name__ == "__main__":
    main()
