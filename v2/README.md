# v2 — reliable families, no maths required

v1 proved retrieval works (right relative ranked first, 21/22) but its
forced group-count clustering merged unrelated proteins. v2 fixes that
with automatic steps only — there is nothing to tune:

1. Duplicate copies of the same chain inside one entry are counted once.
2. Proteins are grouped by who their neighbours are, not by forcing
   everything into a preset number of round groups.
3. A short domain is recognised inside a longer protein, so a PDB domain
   groups with the full-length AlphaFold model of the same protein.
4. Proteins with no relatives stay on their own instead of being
   assigned to a family they do not belong to.

## Use it

```bash
uv run python -m v2 build samples/
uv run --group dev pytest -q   # includes the v2 family test
```

That prints plain-English families and writes, into `v2_out/`:

- `families.txt` — the families, in words
- `families.json` — the same, for code
- `view_families.pml` — optional viewer script (see below)

## Do we need PyMOL?

No. PyMOL (pymol.org) is a molecular *viewer* by Schrödinger — it draws
3D structures. This pipeline reads CIF files directly and never opens a
viewer. `view_families.pml` is only there if you already have PyMOL and
want to eyeball one representative protein per family:

```bash
pymol v2_out/view_families.pml
```

## Would a paid embedding model help?

Not an OpenAI-style text embedding — that would only improve searching
titles/descriptions, not protein grouping. The grouping here is limited
by the simple local sequence+shape fingerprint. The upgrade that would
actually help is a protein-specific encoder (free/local ones exist;
NVIDIA's La-Proteina / RFD4-Proteina latents are the other candidate).
v2 is the baseline those have to beat, so no API key is needed yet.

## Result on the bundled samples

20 chains after dedupe. Foxk1 group includes the AFDB full-length model;
MEF2 chains group together; globin alpha chains group with alpha and
beta with beta; lysozyme, GFP, ubiquitin and the viral protease are
correctly left on their own. See `test_v2.py`.
