"""Parse mmCIF (PDB or AFDB) into per-chain records.

The structure comes from the `_atom_site` category — the only place with
coordinates. Handling, all deliberate:
- first model only (NMR ensembles stack many models in `_atom_site`)
- representative atom = `CA`, from `ATOM` rows with a standard/modified
  amino-acid `label_comp_id` (excludes calcium ions whose atom is also
  named CA, waters, and ligands — all HETATM)
- every atom of such a residue is also retained (`Residue.atoms`, row
  order, first-seen per atom name), with the same first-model /
  zero-occupancy / altloc rules applied per atom, not only per CA
- alternate locations and duplicate residues kept once (first seen)
- zero-occupancy atoms skipped
- residue numbering is retained in both schemes: `label_seq_id` and
  `auth_seq_id` (+ insertion code), because they often differ and SIFTS
  mapping is defined between them

Note on MSE: selenomethionine is in `AA3` (as M), but that only helps
rows labelled `ATOM`. In real PDB files MSE is usually a `HETATM` row,
so it is excluded by the ATOM-only rule exactly like a ligand; the
AA3 entry matters for files (e.g. some AFDB/custom CIFs) that emit
MSE as ATOM.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import gemmi

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
"HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T",
"TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"U"}

@dataclass
class Atom:
    name: str
    element: str
    coord: tuple
    b_factor: float | None = None
    occupancy: float | None = None

@dataclass
class Residue:
    label_seq_id: int | None
    auth_seq_id: str
    ins_code: str
    comp_id: str
    aa: str
    # CA coordinate; if a residue has no CA atom, the first retained
    # atom's coordinate (so `coord`/`ca_coords` always exist per residue).
    coord: tuple
    b_factor: float | None = None
    occupancy: float | None = None
    # All retained atoms of this residue, in _atom_site row order.
    atoms: list = field(default_factory=list)

@dataclass
class ChainRecord:
    entry_id: str
    chain_id: str          # label_asym_id
    source: str            # "PDB" | "AFDB"
    title: str
    method: str
    residues: list = field(default_factory=list)  # Residue, from _atom_site CA rows
    fallback_seq: str = ""
    auth_chain_id: str = ""  # auth_asym_id, often what papers use
    path: str = ""
    @property
    def sequence(self) -> str:
        return "".join(r.aa for r in self.residues) or self.fallback_seq
    @property
    def ca_coords(self) -> list:
        return [r.coord for r in self.residues]
    @property
    def label_range(self):
        ids = [r.label_seq_id for r in self.residues if r.label_seq_id is not None]
        return (ids[0], ids[-1]) if ids else None
    @property
    def auth_range(self):
        return (self.residues[0].auth_seq_id, self.residues[-1].auth_seq_id) if self.residues else None
    @property
    def uid(self): return f"{self.entry_id}:{self.chain_id}"
    @property
    def text(self):
        return (f"{self.entry_id} chain {self.chain_id} auth {self.auth_chain_id} "
                f"{self.title} {self.method} {self.source} seq:{self.sequence}")

def _source(entry_id: str, path: Path) -> str:
    return "AFDB" if entry_id.startswith("AF-") or path.name.startswith("AF-") else "PDB"

def _col(block, tag):
    c = block.find_values(tag)
    return list(c) if c else None

def _get(col, i, default=""):
    return col[i] if col is not None and i < len(col) else default

def _int_or_none(v):
    try: return int(v)
    except (TypeError, ValueError): return None

def _float_or_none(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def parse_cif(path: str | Path) -> list[ChainRecord]:
    path = Path(path)
    doc = gemmi.cif.read_file(str(path))
    b = doc.sole_block()
    entry_id = (b.find_value("_entry.id") or path.stem).strip().strip("'\"")
    title = (b.find_value("_struct.title") or b.find_value("_struct.pdbx_descriptor") or "")
    method = (b.find_value("_exptl.method") or ("AlphaFold" if _source(entry_id, path) == "AFDB" else ""))
    seq_by_entity = {}
    try:
        for row in b.find(["_entity_poly.entity_id", "_entity_poly.pdbx_seq_one_letter_code_can"]):
            seq_by_entity[row[0]] = re.sub(r"[^A-Z]", "", row[1].upper())
    except Exception:
        pass
    # --- _atom_site ---
    atom_id = _col(b, "_atom_site.label_atom_id")
    if atom_id is None:
        return []
    n = len(atom_id)
    cols = {tag: _col(b, f"_atom_site.{tag}") for tag in (
        "group_PDB", "label_asym_id", "auth_asym_id", "label_comp_id", "label_alt_id",
        "label_seq_id", "auth_seq_id", "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y",
        "Cartn_z", "occupancy", "B_iso_or_equiv", "pdbx_PDB_model_num", "label_entity_id",
        "type_symbol")}
    chains: dict[str, dict] = {}
    first_model = None
    for i in range(n):
        atom_name = atom_id[i].strip().strip('"')
        if not atom_name:
            continue
        if cols["group_PDB"] is not None and _get(cols["group_PDB"], i) != "ATOM":
            continue
        comp = _get(cols["label_comp_id"], i).strip().upper()
        aa = AA3.get(comp)
        if aa is None:
            continue
        model = _get(cols["pdbx_PDB_model_num"], i, "1")
        if first_model is None:
            first_model = model
        if model != first_model:
            continue
        occ = _float_or_none(_get(cols["occupancy"], i))
        if occ == 0.0:
            continue
        alt = _get(cols["label_alt_id"], i, ".")
        if alt not in (".", "?", "A"):
            continue
        try:
            coord = (float(_get(cols["Cartn_x"], i)), float(_get(cols["Cartn_y"], i)),
                     float(_get(cols["Cartn_z"], i)))
        except (TypeError, ValueError):
            continue
        cid = _get(cols["label_asym_id"], i, "A")
        label_seq = _int_or_none(_get(cols["label_seq_id"], i))
        auth_seq = _get(cols["auth_seq_id"], i)
        ins = _get(cols["pdbx_PDB_ins_code"], i, "?")
        ch = chains.setdefault(cid, {"res": [], "seen": {}, "entity": _get(cols["label_entity_id"], i),
                                     "auth": _get(cols["auth_asym_id"], i, cid)})
        # Residue identity: as before, (label_seq_id, ins_code). When
        # label_seq_id is missing ('.' in many files), that pair would
        # collapse distinct residues, so fall back to auth numbering +
        # comp as a safety discriminator.
        if label_seq is not None:
            key = (label_seq, ins)
        else:
            key = (None, auth_seq, ins, comp)
        b_factor = _float_or_none(_get(cols["B_iso_or_equiv"], i))
        res = ch["seen"].get(key)
        if res is None:
            res = Residue(label_seq, auth_seq, ins, comp, aa, coord, b_factor, occ, [])
            ch["seen"][key] = res
            ch["res"].append(res)
        if any(a.name == atom_name for a in res.atoms):
            continue  # dedup per (residue, atom_name): first seen wins
        res.atoms.append(Atom(atom_name,
                              _get(cols["type_symbol"], i, "").strip().strip('"'),
                              coord, b_factor, occ))
        if atom_name == "CA":
            # Representative coord / B / occupancy stay CA-derived, as
            # before all-atom retention (byte-identical CA behaviour).
            res.coord, res.b_factor, res.occupancy = coord, b_factor, occ
    out = []
    for cid, ch in chains.items():
        rec = ChainRecord(entry_id, cid, _source(entry_id, path), title, method,
                          ch["res"], seq_by_entity.get(ch["entity"], ""), ch["auth"], str(path))
        if len(rec.sequence) >= 5:
            out.append(rec)
    return out
