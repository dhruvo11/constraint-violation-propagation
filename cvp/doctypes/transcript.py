"""Academic transcript: ten fields, ten constraints (Table I of the report)."""
import datetime as dt
import re
import string

import numpy as np
from PIL import Image, ImageDraw

from ..core.render import FONT_BOLD, FONT_SANS, Style, font
from ..core.schema import Constraint, Field, Schema
from .base import DocType

MARK_FIELDS = [f"m{i}" for i in range(1, 6)]
MAX_PER_SUBJECT = 100
MAX_TOTAL = 500
TODAY = dt.date(2026, 11, 4)  # fixed reference date so experiments are reproducible
EARLIEST = dt.date(2015, 1, 1)  # institution's first examination
# beta: percentage -> grade band. [50,60) -> C and [70,80) -> A as in Appendix A.
BANDS = [(90, "O"), (80, "A+"), (70, "A"), (60, "B"), (50, "C"), (40, "D"), (0, "F")]
GRADES = [g for _, g in BANDS]
DEPTS = ["CSE", "ECE", "MEC", "CIV", "EEE", "AIM"]
ROLL_RE = re.compile(r"^1RUA\d{2}(CSE|ECE|MEC|CIV|EEE|AIM)\d{4}$")


def beta(pct):
    for lo, g in BANDS:
        if pct >= lo - 1e-9:
            return g
    return "F"


def coherent_values(marks, roll, date):
    total = int(sum(marks))
    pct = round(100.0 * total / MAX_TOTAL, 2)
    vals = {m: int(v) for m, v in zip(MARK_FIELDS, marks)}
    vals.update(total=total, pct=pct, grade=beta(pct), date=date, roll=roll)
    return vals


# ---------------------------------------------------------------- fields
def _int(s):
    return int(s) if s.isdigit() else None


def _pct(s):
    return float(s) if re.fullmatch(r"\d{1,3}\.\d{2}", s) else None


def _grade(s):
    return s if s in GRADES else None


def _date(s):
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", s)
    if not m:
        return None
    try:
        return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def _roll(s):
    return s or None


FIELDS = ([Field(m, f"Subject mark {m[1]}", string.digits, str, _int) for m in MARK_FIELDS] + [
    Field("total", "Total", string.digits, str, _int),
    Field("pct", "Percentage", string.digits + ".", lambda v: f"{v:.2f}", _pct),
    Field("grade", "Grade", "OABCDF+", str, _grade),
    Field("date", "Date of issue", string.digits + "/", lambda v: v.strftime("%d/%m/%Y"), _date),
    Field("roll", "Roll number", string.digits + string.ascii_uppercase, str, _roll),
])


# ---------------------------------------------------------------- constraints
def _sum(v):
    d = sum(v[m] for m in MARK_FIELDS) - v["total"]
    return d != 0, d / 100.0


def _percentage(v):
    d = v["pct"] - 100.0 * v["total"] / MAX_TOTAL
    return abs(d) > 0.006, d / 10.0


def _band(v):
    if v["grade"] not in GRADES:
        return True, 0.0
    d = GRADES.index(v["grade"]) - GRADES.index(beta(v["pct"]))
    return d != 0, d / 3.0


def _range(m):
    def check(v):
        x = v[m]
        d = max(0, x - MAX_PER_SUBJECT) + max(0, -x)
        return d > 0, d / 100.0
    return check


def _issue_date(v):
    date = v["date"]
    d = max(0, (date - TODAY).days) - max(0, (EARLIEST - date).days)
    return d != 0, d / 365.0


def _roll_format(v):
    ok = bool(ROLL_RE.match(v["roll"]))
    return not ok, float(not ok)


CONSTRAINTS = [
    Constraint("c1", "arithmetic", tuple(MARK_FIELDS) + ("total",), _sum, "sum of marks = Total",
               sensitive=tuple(MARK_FIELDS) + ("total",), exchangeable=(tuple(MARK_FIELDS),)),
    Constraint("c2", "arithmetic", ("total", "pct"), _percentage, "Pct = 100 x Total / 500",
               sensitive=("total", "pct")),
    Constraint("c3", "logical", ("pct", "grade"), _band, "Grade = beta(Pct)", sensitive=("grade",)),
    *[Constraint(f"c4_{m}", "range", (m,), _range(m), f"0 <= mark {m[1]} <= 100", family="c4")
      for m in MARK_FIELDS],
    Constraint("c5", "temporal", ("date",), _issue_date, "2015 <= Issue date <= today"),
    Constraint("c6", "format", ("roll",), _roll_format, "Roll No matches 1RUA##DEPT####",
               on_missing=(1.0, 1.0)),
]

SCHEMA = Schema("transcript", FIELDS, CONSTRAINTS, groups={"mark": MARK_FIELDS})


# ---------------------------------------------------------------- template
VALUE_FONT = (FONT_SANS, 16)
ROLL_FONT = (FONT_SANS, 15)
ROW_Y = [312 + 38 * k for k in range(5)]
# Template anchors (x, y) of each field before the global page offset.
ANCHORS = {**{m: (530, y) for m, y in zip(MARK_FIELDS, ROW_Y)},
           "total": (530, 512), "pct": (180, 572), "grade": (440, 572),
           "date": (180, 612), "roll": (210, 178)}
SLOT_WIDTH = {**{m: 70 for m in MARK_FIELDS}, "total": 70, "pct": 90, "grade": 60,
              "date": 120, "roll": 170}
W, H = 640, 832

COURSES = [("Data Structures", "CS"), ("Operating Systems", "CS"), ("Database Systems", "CS"),
           ("Computer Networks", "CS"), ("Theory of Computation", "CS"),
           ("Discrete Mathematics", "MA"), ("Linear Algebra", "MA"), ("Probability and Statistics", "MA"),
           ("Natural Language Processing", "CS"), ("Machine Learning", "CS"),
           ("Computer Architecture", "CS"), ("Software Engineering", "CS"),
           ("Digital Logic Design", "EC"), ("Signals and Systems", "EC"), ("Compiler Design", "CS")]
FIRST = ["Aarav", "Diya", "Ishaan", "Meera", "Rohan", "Ananya", "Kabir", "Sneha", "Vikram", "Priya",
         "Arjun", "Kavya", "Nikhil", "Riya", "Siddharth", "Tara", "Aditya", "Neha", "Rahul", "Pooja"]
LAST = ["Sharma", "Iyer", "Reddy", "Nair", "Rao", "Gupta", "Menon", "Kulkarni", "Patil", "Joshi",
        "Hegde", "Shetty", "Kumar", "Bhat", "Verma", "Das", "Pillai", "Gowda", "Naidu", "Mehta"]
SESSIONS = [(2022, 5), (2022, 12), (2023, 5), (2023, 12), (2024, 5), (2024, 12),
            (2025, 5), (2025, 12), (2026, 5)]
MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def sample_roll(rng, year=None):
    year = year if year is not None else int(rng.integers(20, 26))
    return f"1RUA{year:02d}{DEPTS[rng.integers(len(DEPTS))]}{int(rng.integers(1, 10000)):04d}"


class Transcript(DocType):
    name = "transcript"
    title = "Academic transcript"
    schema = SCHEMA
    page = (W, H)
    slots = {f: (*ANCHORS[f], SLOT_WIDTH[f]) for f in SCHEMA.field_names}
    header_rule = (40, 84)
    field_probs = np.array([0.1] * 5 + [0.1] * 5)  # half of single-field edits hit a subject mark
    date_fields = ("date",)
    class_labels = {"mark": "Subject mark", "total": "Total", "pct": "Percentage", "grade": "Grade",
                    "date": "Issue date", "roll": "Roll number"}
    unary_labels = ("in range", "out of range")

    def sample_spec(self, rng):
        marks = np.clip(np.round(rng.normal(66, 14, 5)), 18, 100).astype(int)
        sy, sm = SESSIONS[rng.integers(len(SESSIONS))]
        start = dt.date(sy, sm, 28)
        latest = min(TODAY, start + dt.timedelta(days=150))
        date = start + dt.timedelta(days=int(rng.integers(10, max(11, (latest - start).days))))
        roll = sample_roll(rng, year=int(rng.integers(max(20, sy % 100 - 4), sy % 100 + 1)))
        values = coherent_values(marks.tolist(), roll, date)
        idx = rng.choice(len(COURSES), 5, replace=False)
        courses = [(f"{COURSES[i][1]}{int(rng.integers(21, 49))}{int(rng.integers(0, 10))}", COURSES[i][0])
                   for i in idx]
        content = dict(name=f"{FIRST[rng.integers(len(FIRST))]} {LAST[rng.integers(len(LAST))]}",
                       courses=courses, session=f"{MONTHS[sm]} {sy}",
                       semester=["II", "IV", "VI", "VIII"][rng.integers(4)])
        base = int(rng.integers(244, 253))
        tint = (base + int(rng.integers(0, 3)), base + int(rng.integers(-2, 2)), base - int(rng.integers(2, 12)))
        g = int(rng.integers(15, 55))
        style = Style(ox=int(rng.integers(-8, 9)), oy=int(rng.integers(-8, 9)),
                      jitter={f: [int(rng.integers(-2, 3)), int(rng.integers(-2, 3))] for f in SCHEMA.field_names},
                      tint=tint, ink=(g, g, g + int(rng.integers(0, 15))),
                      sigma=float(rng.uniform(2.0, 5.0)), blur=float(rng.uniform(0.3, 0.7)),
                      q1=int(rng.integers(82, 96)), wm_angle=float(rng.uniform(25, 40)),
                      wm_shade=int(rng.integers(10, 18)), noise_seed=int(rng.integers(1 << 30)))
        return values, content, style

    def anchor(self, field, style):
        ax, ay = ANCHORS[field]
        dx, dy = style.jitter[field]
        return ax + style.ox + dx, ay + style.oy + dy

    @staticmethod
    def field_font(field):
        return font(*(ROLL_FONT if field == "roll" else VALUE_FONT))

    def render_clean(self, values, content, style):
        img = Image.new("RGB", (W, H), tuple(style.tint))
        # Faint diagonal watermark: gives the background texture that a flat fill destroys.
        wm = Image.new("L", (W, H), 0)
        wd = ImageDraw.Draw(wm)
        for k in range(-2, 6):
            wd.text((-60 + 40 * k, 180 + 150 * k), "INSTITUTE OF ADVANCED TECHNOLOGY", fill=255,
                    font=font(FONT_BOLD, 22))
        wm = wm.rotate(style.wm_angle, resample=Image.BILINEAR, center=(W // 2, H // 2))
        shade = Image.new("RGB", (W, H), tuple(max(0, c - style.wm_shade) for c in style.tint))
        img = Image.composite(shade, img, wm)

        d = ImageDraw.Draw(img)
        ox, oy, ink = style.ox, style.oy, tuple(style.ink)

        def text(xy, s, f, center=False):
            x, y = xy
            if center:
                x = (W - d.textlength(s, font=f)) / 2
            d.text((x + ox, y + oy), s, fill=ink, font=f)

        text((0, 28), "INSTITUTE OF ADVANCED TECHNOLOGY", font(FONT_BOLD, 22), center=True)
        text((0, 60), "Office of the Controller of Examinations", font(FONT_SANS, 13), center=True)
        hx, hy = self.header_rule
        d.rectangle([hx + ox, hy + oy, 600 + ox, hy + 1 + oy], fill=ink)
        text((0, 96), "STATEMENT OF MARKS", font(FONT_BOLD, 16), center=True)
        text((0, 118), f"Semester End Examination - Semester {content['semester']}", font(FONT_SANS, 12),
             center=True)

        lab = font(FONT_SANS, 13)
        for y, label in [(150, "Name of Candidate"), (178, "Roll Number"), (206, "Programme"), (234, "Examination")]:
            text((48, y + 1), label, lab)
            text((190, y + 1), ":", lab)
        text((210, 150), content["name"], font(FONT_SANS, 14))
        text((210, 206), "Bachelor of Technology", font(FONT_SANS, 14))
        text((210, 234), content["session"], font(FONT_SANS, 14))

        def rule(y):
            d.rectangle([40 + ox, y + oy, 600 + ox, y + oy], fill=ink)

        rule(270)
        hb = font(FONT_BOLD, 13)
        for x, s in [(48, "Code"), (130, "Course Title"), (430, "Max"), (520, "Marks")]:
            text((x, 280), s, hb)
        rule(300)
        for (code, title), y in zip(content["courses"], ROW_Y):
            text((48, y + 2), code, lab)
            text((130, y + 2), title, lab)
            text((430, y + 1), "100", font(FONT_SANS, 14))
        rule(500)
        text((130, 514), "Total", hb)
        text((430, 513), "500", font(FONT_SANS, 14))
        rule(544)
        text((48, 574), "Percentage", lab)
        text((360, 574), "Grade", lab)
        text((48, 614), "Date of Issue", lab)

        d.ellipse([85 + ox, 680 + oy, 175 + ox, 770 + oy], outline=ink, width=2)
        text((108, 717), "SEAL", font(FONT_BOLD, 14))
        pts = [(420 + ox + 6 * i, 725 + oy + (4 if i % 2 else -4)) for i in range(25)]
        d.line(pts, fill=ink, width=2)
        text((420, 742), "Controller of Examinations", font(FONT_SANS, 12))
        text((0, 800), "Computer-generated statement. Any alteration renders this document invalid.",
             font(FONT_SANS, 10), center=True)

        boxes = {}
        for f in SCHEMA.fields:
            xy = self.anchor(f.name, style)
            s = f.fmt(values[f.name])
            ff = self.field_font(f.name)
            d.text(xy, s, fill=ink, font=ff)
            boxes[f.name] = list(d.textbbox(xy, s, font=ff))
        return img, boxes

    # ------------------------------------------------ forgery value models
    def violating_value(self, f, values, rng):
        v = values[f]
        if f in MARK_FIELDS:
            if rng.random() < 0.15:  # out of range -> {c1, c4_i}
                return int(rng.integers(101, 200))
            for _ in range(100):
                step = int(rng.integers(5, 31)) * (1 if rng.random() < 0.8 else -1)
                new = int(np.clip(v + step, 0, 100))
                if new != v:
                    return new
            return (v + 7) % 100
        if f == "total":
            new = v + int(rng.integers(10, 91)) * (1 if rng.random() < 0.8 else -1)
            return int(np.clip(new, 100, 499)) if new != v else v + 11
        if f == "pct":
            for _ in range(100):
                new = round(v + float(rng.uniform(2, 15)) * (1 if rng.random() < 0.8 else -1), 2)
                if 0 <= new <= 100 and abs(new - v) > 0.5:
                    return new
            return round(max(0.0, v - 5.0), 2)
        if f == "grade":
            i = GRADES.index(v)
            options = [g for g in GRADES[max(0, i - 2):i] or GRADES[i + 1:i + 2]]
            return options[rng.integers(len(options))]
        if f == "date":
            return TODAY + dt.timedelta(days=int(rng.integers(20, 700)))
        if f == "roll":
            op = rng.integers(3)
            if op == 0:
                bad = ["CSX", "CSA", "ECX", "MEE", "CIX", "EEC", "AIX"]
                return v[:6] + bad[rng.integers(len(bad))] + v[9:]
            if op == 1:
                return ["1RUB", "1RVA", "1RUC", "2RUA"][rng.integers(4)] + v[4:]
            return v[:-1] if rng.random() < 0.5 else v + str(int(rng.integers(0, 10)))
        raise KeyError(f)

    def coherent_edit(self, values, rng):
        u = rng.random()
        new = dict(values)
        if u < 0.6:
            m = MARK_FIELDS[rng.integers(5)]
            for _ in range(100):
                x = int(np.clip(values[m] + int(rng.integers(5, 31)), 0, 100))
                if x != values[m]:
                    break
            marks = [x if k == m else values[k] for k in MARK_FIELDS]
            new = coherent_values(marks, values["roll"], values["date"])
            edited = [f.name for f in SCHEMA.fields if f.fmt(new[f.name]) != f.fmt(values[f.name])]
            return new, edited, m
        if u < 0.8:
            for _ in range(100):
                d = values["date"] + dt.timedelta(days=int(rng.integers(-400, 400)))
                if d <= TODAY and d != values["date"] and d.year >= 2016:
                    break
            new["date"] = d
            return new, ["date"], "date"
        for _ in range(100):
            r = values["roll"][:9] + f"{int(rng.integers(1, 10000)):04d}"
            if r != values["roll"]:
                break
        new["roll"] = r
        return new, ["roll"], "roll"

    def random_values(self, rng, v):
        v = dict(v)
        for m in MARK_FIELDS:
            if rng.random() < 0.5:
                v[m] = int(rng.integers(0, 200))
        if rng.random() < 0.5:
            v["total"] = int(rng.integers(10, 1000))
        if rng.random() < 0.5:
            v["pct"] = round(float(rng.uniform(0, 100)), 2)
        if rng.random() < 0.5:
            v["grade"] = GRADES[rng.integers(len(GRADES))]
        if rng.random() < 0.5:
            v["date"] = dt.date(int(rng.integers(2005, 2031)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
        if rng.random() < 0.4:
            n = int(rng.integers(9, 15))
            v["roll"] = "".join(rng.choice(list(string.digits + string.ascii_uppercase), n))
        return v
