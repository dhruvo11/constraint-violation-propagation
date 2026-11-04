"""Transcript schema: fields F, constraint system C, grade bands and equivalence classes.

A structured document is D = (F, v, b). The constraint system below is the one used
throughout the project (Table I of the report). Range constraints are kept per field
(c4_m1 ... c4_m5) so that a unary violation points at a single field.
"""
import datetime as dt
import re

import numpy as np

MARK_FIELDS = [f"m{i}" for i in range(1, 6)]
FIELDS = MARK_FIELDS + ["total", "pct", "grade", "date", "roll"]
FIELD_INDEX = {f: i for i, f in enumerate(FIELDS)}

MAX_PER_SUBJECT = 100
MAX_TOTAL = 500
TODAY = dt.date(2026, 11, 4)  # fixed reference date so experiments are reproducible
EARLIEST = dt.date(2015, 1, 1)  # institution's first examination

# beta: percentage -> grade band. [50,60) -> C and [70,80) -> A as in Appendix A.
BANDS = [(90, "O"), (80, "A+"), (70, "A"), (60, "B"), (50, "C"), (40, "D"), (0, "F")]
GRADES = [g for _, g in BANDS]
DEPTS = ["CSE", "ECE", "MEC", "CIV", "EEE", "AIM"]
ROLL_RE = re.compile(r"^1RUA\d{2}(CSE|ECE|MEC|CIV|EEE|AIM)\d{4}$")

CONSTRAINTS = ["c1", "c2", "c3"] + [f"c4_{m}" for m in MARK_FIELDS] + ["c5", "c6"]
C_INDEX = {c: i for i, c in enumerate(CONSTRAINTS)}
SCOPES = {
    "c1": MARK_FIELDS + ["total"],
    "c2": ["total", "pct"],
    "c3": ["pct", "grade"],
    **{f"c4_{m}": [m] for m in MARK_FIELDS},
    "c5": ["date"],
    "c6": ["roll"],
}
CONSTRAINT_TEXT = {
    "c1": ("Arithmetic", r"$\sum_i m_i = \mathrm{Total}$"),
    "c2": ("Arithmetic", r"$\mathrm{Pct} = 100\cdot\mathrm{Total}/\mathrm{Max}$"),
    "c3": ("Logical", r"$\mathrm{Grade} = \beta(\mathrm{Pct})$"),
    "c4": ("Range", r"$0 \le m_i \le 100$ (one per mark)"),
    "c5": ("Temporal", r"$\mathrm{Earliest} \le \mathrm{IssueDate} \le \mathrm{today}$"),
    "c6": ("Format", r"RollNo matches institutional pattern"),
}

# Constraint-equivalence classes (Proposition 1). "coherent" = no violation.
CLASSES = ["coherent", "mark", "total", "pct", "grade", "date", "roll"]
CLASS_INDEX = {k: i for i, k in enumerate(CLASSES)}
FIELD_CLASS = {**{m: "mark" for m in MARK_FIELDS}, "total": "total", "pct": "pct",
               "grade": "grade", "date": "date", "roll": "roll"}
CLASS_FIELDS = {"coherent": FIELDS, "mark": MARK_FIELDS, "total": ["total"], "pct": ["pct"],
                "grade": ["grade"], "date": ["date"], "roll": ["roll"]}


def beta(pct):
    for lo, g in BANDS:
        if pct >= lo - 1e-9:
            return g
    return "F"


def fmt(field, value):
    """Printed representation of a field value."""
    if field == "pct":
        return f"{value:.2f}"
    if field == "date":
        return value.strftime("%d/%m/%Y")
    return str(value)


def coherent_values(marks, roll, date):
    total = int(sum(marks))
    pct = round(100.0 * total / MAX_TOTAL, 2)
    vals = {m: int(v) for m, v in zip(MARK_FIELDS, marks)}
    vals.update(total=total, pct=pct, grade=beta(pct), date=date, roll=roll)
    return vals


def evaluate(values):
    """Symbolic evaluation of C. Returns (w, r): binary violation and residual vectors.

    Fields whose OCR could not be parsed are None; every constraint touching such a
    field is reported as violated with a zero residual (the parse failure itself is
    exposed to the learned layer separately).
    """
    w = np.zeros(len(CONSTRAINTS), np.float32)
    r = np.zeros(len(CONSTRAINTS), np.float32)
    v = values
    marks = [v.get(m) for m in MARK_FIELDS]

    if None in marks or v.get("total") is None:
        w[C_INDEX["c1"]] = 1
    else:
        d = sum(marks) - v["total"]
        w[C_INDEX["c1"]] = float(d != 0)
        r[C_INDEX["c1"]] = d / 100.0

    if v.get("total") is None or v.get("pct") is None:
        w[C_INDEX["c2"]] = 1
    else:
        d = v["pct"] - 100.0 * v["total"] / MAX_TOTAL
        w[C_INDEX["c2"]] = float(abs(d) > 0.006)
        r[C_INDEX["c2"]] = d / 10.0

    if v.get("pct") is None or v.get("grade") not in GRADES:
        w[C_INDEX["c3"]] = 1
    else:
        d = GRADES.index(v["grade"]) - GRADES.index(beta(v["pct"]))
        w[C_INDEX["c3"]] = float(d != 0)
        r[C_INDEX["c3"]] = d / 3.0

    for m, x in zip(MARK_FIELDS, marks):
        c = C_INDEX[f"c4_{m}"]
        if x is None:
            w[c] = 1
        else:
            d = max(0, x - MAX_PER_SUBJECT) + max(0, -x)
            w[c] = float(d > 0)
            r[c] = d / 100.0

    date = v.get("date")
    if date is None:
        w[C_INDEX["c5"]] = 1
    else:
        d = max(0, (date - TODAY).days) - max(0, (EARLIEST - date).days)
        w[C_INDEX["c5"]] = float(d != 0)
        r[C_INDEX["c5"]] = d / 365.0

    roll = v.get("roll")
    ok = roll is not None and bool(ROLL_RE.match(roll))
    w[C_INDEX["c6"]] = float(not ok)
    r[C_INDEX["c6"]] = float(not ok)
    return w, r


def signature_class(w):
    """Deterministic signature -> equivalence class lookup (the symbolic rule of Table II)."""
    on = {c for c in CONSTRAINTS if w[C_INDEX[c]] > 0}
    if not on:
        return "coherent"
    if any(c.startswith("c4_") for c in on) or on == {"c1"}:
        return "mark"
    if {"c1", "c2"} <= on:
        return "total"
    if "c1" in on:
        return "mark"
    if "c2" in on:
        return "pct"
    if "c3" in on:
        return "grade"
    if "c5" in on:
        return "date"
    return "roll"


def candidate_fields(cls, w):
    """Fields that can explain the violations: fields of class K intersected with the
    scopes of violated unary constraints (when any)."""
    fields = CLASS_FIELDS[cls]
    if cls == "mark":
        hit = [m for m in MARK_FIELDS if w[C_INDEX[f"c4_{m}"]] > 0]
        if hit:
            return hit
    return fields


def reduction_factor(class_sizes, n_fields):
    """rho = 1 - E_f[|K(f)|] / |F| for a uniform choice of tampered field (Eq. 1)."""
    expected = sum(s * s for s in class_sizes) / n_fields
    return 1.0 - expected / n_fields, expected
