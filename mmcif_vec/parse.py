"""Parse mmCIF (PDB or AFDB) into per-chain records with CA representative atoms."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import gemmi

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
"HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T",
"TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"U"}

@dataclass
class ChainRecord:
    entry_id: str
    chain_id: str
    source: str            # "PDB" | "AFDB"
    title: str
    method: str
    sequence: str          # 1-letter, from CA residues (fallback: entity_poly)
    ca_coords: list = field(default_factory=list)  # representative atoms: CA, first model
    path: str = ""
    @property
    def uid(self): return f"{self.entry_id}:{self.chain_id}"
    @property
    def text(self):
        return f"{self.entry_id} chain {self.chain_id} {self.title} {self.method} {self.source} seq:{self.sequence}"

def _source(entry_id: str, path: Path) -> str:
    return "AFDB" if entry_id.startswith("AF-") or path.name.startswith("AF-") else "PDB"

def parse_cif(path: str | Path) -> list[ChainRecord]:
    path = Path(path)
    doc = gemmi.cif.read_file(str(path))
    b = doc.sole_block()
    entry_id = (b.find_value("_entry.id") or path.stem).strip()
    title = (b.find_value("_struct.title") or b.find_value("_struct.pdbx_descriptor") or "")
    method = (b.find_value("_exptl.method") or ("AlphaFold" if _source(entry_id, path)=="AFDB" else ""))
    # entity_poly fallback sequences by entity
    seq_by_entity = {}
    try:
        for row in b.find(["_entity_poly.entity_id","_entity_poly.pdbx_seq_one_letter_code_can"]):
            seq_by_entity[row[0]] = re.sub(r"[^A-Z]", "", row[1].upper())
    except Exception: pass
    # _atom_site: first model only, CA = representative atom
    chains: dict[str, dict] = {}
    cols = ["_atom_site.label_asym_id","_atom_site.label_atom_id","_atom_site.label_comp_id",
            "_atom_site.Cartn_x","_atom_site.Cartn_y","_atom_site.Cartn_z",
            "_atom_site.pdbx_PDB_model_num","_atom_site.label_entity_id"]
    try:
        table = b.find(cols)
    except Exception:
        return []
    first_model = None
    for row in table:
        model = row[6]
        if first_model is None: first_model = model
        if model != first_model: continue
        if row[1].strip().strip('"') != "CA": continue
        ch = chains.setdefault(row[0], {"seq": [], "ca": [], "entity": row[7]})
        aa = AA3.get(row[2].strip().upper())
        if aa: ch["seq"].append(aa)
        ch["ca"].append((float(row[3]), float(row[4]), float(row[5])))
    out = []
    for cid, ch in chains.items():
        seq = "".join(ch["seq"]) or seq_by_entity.get(ch["entity"], "")
        if len(seq) < 5: continue
        out.append(ChainRecord(entry_id, cid, _source(entry_id, path), title, method,
                               seq, ch["ca"], str(path)))
    return out
