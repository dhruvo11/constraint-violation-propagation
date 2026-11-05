"""Student identity card of the (fictional) Institute of Advanced Technology, the
institute that issues the transcripts.

No arithmetic at all, only cross-field logic: the USN encodes the year of admission and
the branch, the batch spans the four years of the B.Tech, the card is valid until the
July the batch graduates, the holder was 16-25 at admission and the card was issued in
the first year. The holder's name is checked by no rule. The batch takes part in five
rules and shares a signature with the USN, the date of birth and the validity date in
turn, so the commonest forgery, a stretched validity, is ambiguous to the rules alone.
"""
import datetime as dt
import re
import string

import numpy as np

from ..core import rules as R
from ..core.render import FONT_BOLD, FONT_SANS, font
from ..core.schema import Constraint, Field, Schema
from . import _draw as D
from .base import DocType, Layout
from .certificate import random_usn, sample_name
from .invoice import flag
from .transcript import DEPTS, ROLL_RE

UPPER = string.ascii_uppercase
BATCH_RE = re.compile(r"^(\d{4})-(\d{4})$")
YEARS = 4  # B.Tech


def batch(s):
    m = BATCH_RE.match(s)
    return (int(m.group(1)), int(m.group(2))) if m else None


FIELDS = [
    Field("name", "Holder's name", UPPER + " .", str, lambda s: s or None),
    Field("usn", "USN", string.digits + UPPER, str, lambda s: s or None),
    Field("dept", "Branch", UPPER, str, lambda s: s or None),
    Field("batch", "Batch", string.digits + "-", str, lambda s: s if BATCH_RE.match(s) else None),
    Field("dob", "Date of birth", string.digits + "/", R.fmt_dmy, R.parse_dmy),
    Field("valid", "Valid upto", string.digits + "/", R.fmt_dmy, R.parse_dmy),
    Field("issue", "Date of issue", string.digits + "/", R.fmt_dmy, R.parse_dmy),
]


def _branch(v):
    return flag(v["usn"][6:9] == v["dept"])


def _joined(v):
    if not ROLL_RE.match(v["usn"]):
        return True, 0.0
    d = 2000 + int(v["usn"][4:6]) - batch(v["batch"])[0]
    return d != 0, d / 2.0


def _span(v):
    s, e = batch(v["batch"])
    d = e - s - YEARS
    return d != 0, d / 2.0


def _validity(v):
    d = (v["valid"] - dt.date(batch(v["batch"])[1], 7, 31)).days
    return d != 0, d / 365.0


def _age(v):
    a = R.age_on(v["dob"], dt.date(batch(v["batch"])[0], 8, 1))
    d = max(0, 16 - a) + max(0, a - 25)
    return d > 0, d / 5.0


def _issued(v):
    s = batch(v["batch"])[0]
    lo, hi = dt.date(s, 7, 1), dt.date(s + 1, 6, 30)
    d = max(0, (lo - v["issue"]).days) + max(0, (v["issue"] - hi).days)
    return d > 0, d / 365.0


def _not_future(v):
    d = max(0, (v["issue"] - R.TODAY).days)
    return d > 0, d / 30.0


CONSTRAINTS = [
    Constraint("c_usn", "format", ("usn",), lambda v: flag(bool(ROLL_RE.match(v["usn"]))),
               "USN matches 1RUA##DEPT####", on_missing=(1.0, 1.0)),
    Constraint("c_brn", "equality", ("usn", "dept"), _branch, "Branch = branch code in the USN", sensitive=("dept",)),
    Constraint("c_jyr", "logical", ("usn", "batch"), _joined, "USN year = first year of the batch"),
    Constraint("c_spn", "logical", ("batch",), _span, "Batch spans four years"),
    Constraint("c_vld", "temporal", ("batch", "valid"), _validity, "Valid upto 31 July of the final year",
               sensitive=("valid",)),
    Constraint("c_age", "temporal", ("dob", "batch"), _age, "Aged 16-25 at admission"),
    Constraint("c_iss", "temporal", ("batch", "issue"), _issued, "Issued during the first year"),
    Constraint("c_fut", "temporal", ("issue",), _not_future, "Issue date <= today"),
]

SCHEMA = Schema("idcard", FIELDS, CONSTRAINTS)


# ---------------------------------------------------------------- template
W, H = 864, 544
HEADER = (30, 78)
B16 = (FONT_BOLD, 16)
ROWS = {"name": 100, "usn": 138, "dept": 176, "batch": 214, "dob": 252, "valid": 290}
LAYOUT = Layout({
    "name": (420, ROWS["name"], "l", FONT_BOLD, 18, 400),
    "usn": (420, ROWS["usn"], "l", *B16, 180),
    "dept": (420, ROWS["dept"], "l", *B16, 60),
    "batch": (420, ROWS["batch"], "l", *B16, 110),
    "dob": (420, ROWS["dob"], "l", *B16, 110),
    "valid": (420, ROWS["valid"], "l", *B16, 110),
    "issue": (132, 401, "l", FONT_SANS, 13, 100),
})
BLOOD = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
WEIGHTS = {"usn": 1.0, "dept": 1.0, "batch": 1.0, "dob": 1.5, "valid": 2.5, "issue": 1.0}


class IDCard(DocType):
    name = "idcard"
    title = "Student ID card"
    schema = SCHEMA
    page = (W, H)
    slots = LAYOUT.slots()
    wide = LAYOUT.wide()
    header_rule = HEADER
    weights = WEIGHTS
    date_fields = ("dob", "valid", "issue")
    class_labels = {f.name: f.label for f in FIELDS}
    unary_labels = ("", "")

    def sample_spec(self, rng):
        start = int(rng.integers(2019, 2027))
        lo = dt.date(start, 8, 1)
        issue = R.random_date(rng, lo, min(dt.date(start, 10, 31), R.TODAY))
        age = int(rng.choice([17, 18, 18, 18, 19, 19, 20, 21]))
        dob = R.add_years(lo, -age) - dt.timedelta(days=int(rng.integers(1, 365)))
        dept = DEPTS[rng.integers(len(DEPTS))]
        v = dict(name=sample_name(rng), usn=f"1RUA{start % 100:02d}{dept}{int(rng.integers(1, 10000)):04d}",
                 dept=dept, batch=f"{start}-{start + YEARS}", dob=dob, valid=dt.date(start + YEARS, 7, 31), issue=issue)
        content = dict(blood=BLOOD[rng.integers(len(BLOOD))],
                       phone=f"+91 {int(rng.integers(70000, 99999))} {int(rng.integers(10000, 99999))}")
        base = int(rng.integers(244, 253))
        style = D.sample_style(rng, SCHEMA.field_names, tint=(base, base, base - 4), ink=(15, 45))
        return v, content, style

    def anchor(self, field, style):
        return LAYOUT.origin(field, "", LAYOUT.font(field), style)

    def place(self, field, text, fnt, style):
        return LAYOUT.origin(field, text, fnt, style)

    def font_scale(self, field):
        return LAYOUT.scale(field)

    def render_clean(self, values, content, style):
        img = D.paper((W, H), style)
        D.waves(img, style, shift=(10, 6, 2), n=40, amp=(8, 22), period=(160, 320), area=(232, 84, W, 440))
        p = D.Pen(img, style)
        band = tuple(max(0, c - s) for c, s in zip(style.tint, (36, 18, 2)))
        p.rect(-style.ox, -style.oy, W, 70, fill=band)
        p.rect(-style.ox, 446, W, H, fill=band)
        p.text(0, 12, "INSTITUTE OF ADVANCED TECHNOLOGY", font(FONT_BOLD, 24), align="page")
        p.text(0, 46, "Kengeri, Bengaluru 560 060  -  STUDENT IDENTITY CARD", font(FONT_BOLD, 12), align="page")
        p.rule(HEADER[0], HEADER[1], 834, t=2)
        p.portrait(40, 100, 220, 330, shade=tuple(c - 30 for c in style.tint))
        lab = font(FONT_SANS, 14)
        for f, s in [("name", "Name"), ("usn", "USN"), ("dept", "Branch"), ("batch", "Batch"),
                     ("dob", "Date of Birth"), ("valid", "Valid Upto")]:
            p.text(250, ROWS[f] + 2, s, lab)
            p.text(402, ROWS[f] + 2, ":", lab)
        p.text(250, 330, "Programme", lab)
        p.text(402, 330, ":", lab)
        p.text(420, 328, "B.TECH", font(*B16))
        p.text(250, 366, "Blood Group", lab)
        p.text(402, 366, ":", lab)
        p.text(420, 364, content["blood"], font(*B16))
        p.text(40, 402, "Issued on:", font(FONT_SANS, 13))
        p.squiggle(640, 384, n=22, amp=5, step=6)
        p.text(706, 408, "Principal", font(FONT_SANS, 13), align="c")
        p.text(0, 462, "Institute of Advanced Technology, Kengeri, Bengaluru 560 060  -  Ph: 080-4000 1234",
               font(FONT_SANS, 11), align="page")
        p.text(0, 480, f"Emergency contact: {content['phone']}", font(FONT_SANS, 11), align="page")
        p.text(0, 506, "This card is the property of the Institute. If found, please return it to the address above.",
               font(FONT_SANS, 10), align="page")
        return img, D.draw_fields(img, LAYOUT, SCHEMA, values, style)

    # ------------------------------------------------ forgery value models
    def violating_value(self, f, v, rng):
        x = v[f]
        if f == "usn":
            u = rng.random()
            if u < 0.45:
                return x[:6] + rng.choice([d for d in DEPTS if d != x[6:9]]) + x[9:]
            if u < 0.8 and x[4:6].isdigit():
                return x[:4] + f"{(int(x[4:6]) + int(rng.choice([-2, -1, 1, 2]))) % 100:02d}" + x[6:]
            return x[:6] + ["CSX", "ECX", "MEE", "AIX"][rng.integers(4)] + x[9:] if rng.random() < 0.5 else x[:-1]
        if f == "dept":
            return rng.choice([d for d in DEPTS if d != x])
        if f == "batch":
            s, e = batch(x) or (2022, 2026)
            if rng.random() < 0.6:
                k = int(rng.choice([-2, -1, 1, 2]))
                return f"{s + k}-{e + k}"
            return f"{s}-{e + int(rng.choice([-1, 1, 2]))}"
        if f == "dob":
            s = (batch(v["batch"]) or (2022, 2026))[0]
            age = int(rng.choice([10, 12, 13, 14, 27, 30, 35]))
            return R.add_years(dt.date(s, 8, 1), -age) - dt.timedelta(days=int(rng.integers(1, 365)))
        if f == "valid":
            return R.add_years(x, int(rng.choice([1, 1, 2, 3])))
        if f == "issue":
            if rng.random() < 0.3:
                return R.TODAY + dt.timedelta(days=int(rng.integers(10, 300)))
            return R.add_years(x, int(rng.choice([-2, -1, 1, 2, 3])))
        raise KeyError(f)

    def coherent_edit(self, values, rng):
        u = rng.random()
        new = dict(values)
        if u < 0.4:  # someone else's card with the name changed
            for _ in range(100):
                n = sample_name(rng)
                if n != values["name"]:
                    break
            new["name"] = n
            return new, ["name"], "name"
        b = batch(values["batch"])
        if u < 0.6 or b is None:  # another birthday that keeps the holder's age plausible
            for _ in range(100):
                d = values["dob"] + dt.timedelta(days=int(rng.integers(-700, 700)))
                if d != values["dob"] and (b is None or 16 <= R.age_on(d, dt.date(b[0], 8, 1)) <= 25):
                    new["dob"] = d
                    return new, ["dob"], "dob"
            new["dob"] = values["dob"] + dt.timedelta(days=1)
            return new, ["dob"], "dob"
        if u < 0.85 and ROLL_RE.match(values["usn"]):  # a later batch: card valid for longer
            ks = [k for k in (1, 2) if dt.date(b[0] + k, 7, 1) <= R.TODAY] or [-1]
            k = ks[rng.integers(len(ks))]
            new["batch"] = f"{b[0] + k}-{b[1] + k}"
            new["valid"] = dt.date(b[1] + k, 7, 31)
            new["usn"] = values["usn"][:4] + f"{(b[0] + k) % 100:02d}" + values["usn"][6:]
            issue = R.add_years(values["issue"], k)
            new["issue"] = issue if issue <= R.TODAY else dt.date(b[0] + k, 7, 1)
            edited = [f.name for f in SCHEMA.fields if f.fmt(new[f.name]) != f.fmt(values[f.name])]
            return new, edited, "valid"
        dept = rng.choice([d for d in DEPTS if d != values["dept"]])  # another branch, USN to match
        new["dept"] = dept
        new["usn"] = values["usn"][:6] + dept + values["usn"][9:]
        return new, ["usn", "dept"], "dept"

    def random_values(self, rng, v):
        v = dict(v)
        if rng.random() < 0.5:
            v["name"] = " ".join("".join(rng.choice(list(UPPER), int(rng.integers(1, 10))))
                                 for _ in range(int(rng.integers(1, 4))))
        if rng.random() < 0.4:
            v["usn"] = random_usn(rng)
        if rng.random() < 0.4:
            v["dept"] = "".join(rng.choice(list(UPPER), 3))
        if rng.random() < 0.5:
            v["batch"] = f"{int(rng.integers(1990, 2040))}-{int(rng.integers(1990, 2040))}"
        for f in ("dob", "valid", "issue"):
            if rng.random() < 0.5:
                v[f] = dt.date(int(rng.integers(1970, 2040)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
        return v
