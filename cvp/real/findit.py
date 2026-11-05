"""Find it again! (Tornes et al., "Receipt Dataset for Document Forgery Detection", ICDAR 2023):
988 scanned receipts from SROIE (Malaysian shops), 163 with real forgeries made by people with
image editors (copy-paste inside the image "CPI", imitation, cut, pixel edits), the edited
regions annotated. The transcriptions give the text as printed, forgeries included.

Data: data/real/findit2/findit2/{train,val,test}/<id>.png and .txt, labels in <split>.txt
(downloaded from the University of La Rochelle, L3i).
"""
import ast
import csv
import os
import re
from dataclasses import dataclass, field

from .. import paths

ROOT = os.path.join(paths.real_dir("findit2"), "findit2")
SPLITS = ("train", "val", "test")


@dataclass
class Scan:
    id: str
    split: str
    forged: bool
    edited: list = field(default_factory=list)   # edited regions: dict(x0, y0, x1, y1, kind, entity)
    sources: list = field(default_factory=list)  # regions copied from (unmodified): same keys
    lines: list = field(default_factory=list)    # the transcription, one printed segment per line

    @property
    def image(self):
        return os.path.join(ROOT, self.split, self.id + ".png")


def _regions(text):
    try:
        ann = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [], []
    if not isinstance(ann, dict):
        return [], []
    edited, sources = [], []
    for r in ann.get("regions", []):
        s, a = r["shape_attributes"], r["region_attributes"]
        m = a.get("Modified area", {})
        kinds = [k for k, v in m.items() if v] if isinstance(m, dict) else [m]
        box = dict(x0=s["x"], y0=s["y"], x1=s["x"] + s["width"], y1=s["y"] + s["height"],
                   kind="/".join(kinds), entity=a.get("Entity type", ""))
        (sources if a.get("Original area") == "yes" else edited).append(box)
    return edited, sources


MISSING = []  # receipts listed in a label file whose image or transcription is not in the archive


def scans(split, root=ROOT):
    with open(os.path.join(root, f"{split}.txt"), newline="") as fh:
        for row in csv.DictReader(fh):
            sid = row["image"].rsplit(".", 1)[0]
            txt = os.path.join(root, split, sid + ".txt")
            if not (os.path.exists(txt) and os.path.exists(os.path.join(root, split, sid + ".png"))):
                if (split, sid) not in MISSING:
                    MISSING.append((split, sid))
                continue
            edited, sources = _regions(row["forgery annotations"]) if row["forged"] == "1" else ([], [])
            with open(txt, errors="replace") as tf:
                lines = [ln.strip() for ln in tf if ln.strip()]
            yield Scan(sid, split, row["forged"] == "1", edited, sources, lines)


# ------------------------------------------------------------------ amounts from the transcription
_AMOUNT = re.compile(r"(?<![\d.])-?\s?(?:RM\s?)?-?\d{1,3}(?:,\d{3})*\.\d{2}(?![\d])|(?<![\d.])-?\d+\.\d{2}(?![\d])")
# Keys in priority order; a key line's amount is on the line itself or one of the next two.
KEYS = {
    "total": [r"TOTAL SALES \(INCL", r"TOTAL \(?INCL", r"GRAND TOTAL", r"NETT? TOTAL", r"TOTAL AMOUNT", r"TOTAL PAYABLE",
              r"AMOUNT DUE", r"^TOTAL\s*(RM)?\s*:?\s*$", r"^TOTAL\s*(RM)?\s*:?\s*-?\d"],
    "subtotal": [r"TOTAL SALES \(EXCL", r"SUB\s?-?\s?TOTAL", r"TOTAL \(?EXCL", r"TOTAL SALES\s*:?$"],
    "tax": [r"TOTAL GST", r"^GST\s*(\d+%)?\s*:?\s*$", r"^GST @", r"GST \(?6%", r"^TAX\s*:?$", r"SERVICE TAX"],
    "discount": [r"^DISCOUNT", r"TOTAL DISCOUNT"],
    "rounding": [r"ROUND"],
    "cash": [r"^CASH\s*(RM)?\s*:?\s*$", r"^CASH\s*(RM)?\s*:?\s*-?\d", r"^TENDER", r"CASH TENDERED", r"^PAYMENT"],
    "change": [r"^CHANGE", r"CHANGE DUE"],
}


def money(text):
    m = _AMOUNT.search(text)
    if not m:
        return None
    s = m.group().replace("RM", "").replace(" ", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def amounts(scan):
    """{key: value} for the receipt's summary amounts: for each key, the first line matching its
    patterns (in priority order), with the amount on that line or one of the next two."""
    up = [ln.upper() for ln in scan.lines]
    out = {}
    for key, pats in KEYS.items():
        for pat in pats:
            rx = re.compile(pat)
            for i, ln in enumerate(up):
                if not rx.search(ln):
                    continue
                for j in range(i, min(i + 3, len(up))):
                    tail = up[j][rx.search(ln).end():] if j == i else up[j]
                    v = money(tail)
                    if v is not None:
                        out[key] = v
                        break
                if key in out:
                    break
            if key in out:
                break
    return out


TOL = 0.011  # cents: printed amounts are exact to the cent


def check(v):
    """{rule: None (does not apply) or whether it holds} on the summary amounts `v`."""
    out = {}
    if {"total", "subtotal", "tax"} <= set(v):  # without the tax line found, a gap could be the tax: no verdict
        calc = v["subtotal"] - abs(v.get("discount", 0.0)) + v["tax"] + v.get("rounding", 0.0)
        alt = v["subtotal"] - abs(v.get("discount", 0.0)) + v.get("rounding", 0.0)  # tax already in the subtotal
        out["total"] = abs(calc - v["total"]) <= TOL or abs(alt - v["total"]) <= TOL
    else:
        out["total"] = None
    out["change"] = (abs(v["cash"] - v["total"] - v["change"]) <= TOL) if {"cash", "total", "change"} <= set(v) else None
    out["tender"] = (v["cash"] >= v["total"] - TOL) if {"cash", "total"} <= set(v) else None
    out["rounding"] = (abs(v["rounding"]) <= 0.02 + 1e-9) if "rounding" in v else None
    return out
