"""Certificate of completion (an online programme or internship), modelled on the
course, internship and college certificates students actually hold.

Certificates carry few rules: a duration that must match the dates, an issue date
shortly after the end, a grade that must match the score, a certificate number with a
check letter that also appears in the verification address, and the holder's USN.
The holder's name is checked by no rule at all: swapping it is the forgery CVP can only
catch visually, which makes this the sparse end of the constraint spectrum.
"""
import datetime as dt
import os
import re
import string

import numpy as np

from ..core import rules as R
from ..core.render import FONT_SANS, font
from ..core.schema import Constraint, Field, Schema
from . import _draw as D
from .base import DocType, Layout
from .invoice import flag, typo
from .transcript import DEPTS, FIRST, LAST, ROLL_RE

_DIR = os.path.dirname(FONT_SANS)
STIX = os.path.join(_DIR, "STIXGeneral.ttf")
STIX_BOLD = os.path.join(_DIR, "STIXGeneralBol.ttf")
STIX_ITALIC = os.path.join(_DIR, "STIXGeneralItalic.ttf")
STIX_BOLD_ITALIC = os.path.join(_DIR, "STIXGeneralBolIta.ttf")

BANDS = [(90, "A+"), (75, "A"), (60, "B"), (50, "C"), (40, "D")]
GRADES = [g for _, g in BANDS]
CHECK_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"  # mod-23 check letter table
CERT_RE = re.compile(r"^SN(\d{2})-(\d{6})-([A-Z])$")
LAUNCH = dt.date(2021, 1, 1)  # the platform's first programme
ISSUE_WINDOW = 45  # days after the programme ends


def band(score):
    for lo, g in BANDS:
        if score >= lo:
            return g
    return None


def cert_no(issue_year, serial):
    yy = issue_year % 100
    return f"SN{yy:02d}-{serial:06d}-{CHECK_LETTERS[int(f'{yy:02d}{serial:06d}') % 23]}"


def cert_ok(s):
    m = CERT_RE.match(s)
    return bool(m) and CHECK_LETTERS[int(m.group(1) + m.group(2)) % 23] == m.group(3)


def _int(s):
    return int(s) if s.isdigit() else None


def _score(s):
    m = re.fullmatch(r"(\d{1,3})%", s)
    return int(m.group(1)) if m else None


UPPER = string.ascii_uppercase
FIELDS = [
    Field("name", "Holder's name", UPPER + " .", str, lambda s: s or None),
    Field("weeks", "Duration (weeks)", string.digits, str, _int),
    Field("start", "Start date", string.digits + UPPER + " ", R.fmt_dmony, R.parse_dmony),
    Field("end", "End date", string.digits + UPPER + " ", R.fmt_dmony, R.parse_dmony),
    Field("score", "Final score", string.digits + "%", lambda v: f"{v}%", _score),
    Field("grade", "Grade", "ABCD+", str, lambda s: s if s in GRADES else None),
    Field("issue", "Date of issue", string.digits + UPPER + " ", R.fmt_dmony, R.parse_dmony),
    Field("cert_no", "Certificate number", string.digits + UPPER + "-", str, lambda s: s or None),
    Field("usn", "USN", string.digits + UPPER, str, lambda s: s or None),
    Field("url_id", "Verification code", string.digits + UPPER + "-", str, lambda s: s or None),
]


def _duration(v):
    d = (v["end"] - v["start"]).days + 1 - 7 * v["weeks"]
    return d != 0, d / 7.0


def _issued(v):
    d = (v["issue"] - v["end"]).days
    off = max(0, -d) + max(0, d - ISSUE_WINDOW)
    return off > 0, off / 30.0


def _not_future(v):
    d = max(0, (v["issue"] - R.TODAY).days)
    return d > 0, d / 30.0


def _score_range(v):
    d = max(0, 40 - v["score"]) + max(0, v["score"] - 100)
    return d > 0, d / 10.0


def _band(v):
    if v["grade"] not in GRADES or band(v["score"]) is None:
        return True, 0.0
    d = GRADES.index(v["grade"]) - GRADES.index(band(v["score"]))
    return d != 0, d / 2.0


def _cert_year(v):
    m = CERT_RE.match(v["cert_no"])
    return flag(bool(m) and int(m.group(1)) == v["issue"].year % 100)


def _usn_year(v):
    if not ROLL_RE.match(v["usn"]):
        return True, 0.0
    joined = 2000 + int(v["usn"][4:6])
    d = max(0, joined - v["start"].year) + max(0, v["start"].year - joined - 5)
    return d > 0, d / 2.0


CONSTRAINTS = [
    Constraint("c_dur", "temporal", ("weeks", "start", "end"), _duration, "End - start + 1 day = 7 x weeks",
               sensitive=("weeks", "start", "end")),
    Constraint("c_iss", "temporal", ("end", "issue"), _issued, "Issued within 45 days after the end"),
    Constraint("c_fut", "temporal", ("issue",), _not_future, "Issue date <= today"),
    Constraint("c_rng", "range", ("score",), _score_range, "40 <= score <= 100"),
    Constraint("c_bnd", "logical", ("score", "grade"), _band, "Grade = band(score)", sensitive=("grade",)),
    Constraint("c_chk", "checksum", ("cert_no",), lambda v: flag(cert_ok(v["cert_no"])),
               "Certificate no. is SNyy-nnnnnn-L with a valid check letter", on_missing=(1.0, 1.0)),
    Constraint("c_cyr", "logical", ("cert_no", "issue"), _cert_year, "Certificate no. carries the issue year"),
    Constraint("c_url", "logical", ("cert_no", "url_id"), lambda v: flag(v["cert_no"] == v["url_id"]),
               "Verification code = certificate no.", sensitive=("cert_no", "url_id")),
    Constraint("c_usn", "format", ("usn",), lambda v: flag(bool(ROLL_RE.match(v["usn"]))),
               "USN matches 1RUA##DEPT####", on_missing=(1.0, 1.0)),
    Constraint("c_uyr", "temporal", ("usn", "start"), _usn_year, "Joined at most 5 years before the programme"),
]

SCHEMA = Schema("certificate", FIELDS, CONSTRAINTS)


# ---------------------------------------------------------------- template
W, H = 960, 672
HEADER = (180, 126)
LAYOUT = Layout({
    "name": (480, 222, "c", STIX_BOLD_ITALIC, 32, 640),
    "weeks": (306, 362, "l", STIX_BOLD, 17, 34),
    "start": (473, 362, "l", STIX_BOLD, 17, 126),
    "end": (630, 362, "l", STIX_BOLD, 17, 126),
    "score": (490, 398, "l", STIX_BOLD, 17, 56),
    "grade": (610, 398, "l", STIX_BOLD, 17, 40),
    "issue": (212, 503, "l", STIX, 15, 116),
    "cert_no": (212, 529, "l", STIX, 15, 150),
    "usn": (212, 555, "l", STIX, 15, 150),
    "url_id": (288, 587, "l", FONT_SANS, 12, 130),
})
COURSES = ["Machine Learning Foundations", "Full Stack Web Development", "Data Analytics with Python",
           "Cloud Computing Essentials", "Cyber Security Fundamentals", "Internet of Things Lab",
           "Deep Learning for Computer Vision", "UI/UX Design Studio", "Embedded Systems Internship",
           "Natural Language Processing", "DevOps and Containers", "Android App Development"]
WEIGHTS = {"weeks": 1.0, "start": 1.0, "end": 1.0, "score": 2.0, "grade": 2.0, "issue": 1.5, "cert_no": 1.0,
           "usn": 1.0, "url_id": 0.5}


def random_usn(rng):
    """A USN-shaped string with random parts (and sometimes a digit too many or too few)."""
    s = "".join(rng.choice(list(string.digits + UPPER), 4)) + f"{int(rng.integers(0, 100)):02d}" + \
        "".join(rng.choice(list(UPPER), 3)) + f"{int(rng.integers(0, 10 ** 4)):04d}"
    u = rng.random()
    return s[:-1] if u < 0.15 else s + str(int(rng.integers(10))) if u < 0.3 else s


def sample_name(rng):
    first, last = FIRST[rng.integers(len(FIRST))].upper(), LAST[rng.integers(len(LAST))].upper()
    u = rng.random()
    return f"{first} {last}" if u < 0.6 else f"{first} {last[0]}" if u < 0.8 else f"{first} {last[0]} {chr(65 + int(rng.integers(26)))}"


def rosette(img, style, cx, cy, r0, amp, petals, n=9):
    """Guilloche rosette: polar sine curves around the centre, very light."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    color = tuple(max(0, c - style.wm_shade + 2) for c in style.tint)
    t = np.linspace(0, 2 * np.pi, 720)
    for i in range(n):
        rr = r0 + amp * np.sin(petals * t + i * np.pi / n)
        pts = list(zip((cx + style.ox + rr * np.cos(t)).tolist(), (cy + style.oy + rr * np.sin(t)).tolist()))
        d.line(pts + pts[:1], fill=color, width=1)


class Certificate(DocType):
    name = "certificate"
    title = "Certificate of completion"
    schema = SCHEMA
    page = (W, H)
    slots = LAYOUT.slots()
    wide = LAYOUT.wide()
    header_rule = HEADER
    weights = WEIGHTS
    date_fields = ("start", "end", "issue")
    class_labels = {f.name: f.label for f in FIELDS}
    unary_labels = ("", "")

    def sample_spec(self, rng):
        weeks = int(rng.choice([4, 6, 8, 8, 10, 12, 12, 16]))
        start = R.random_date(rng, LAUNCH, R.TODAY - dt.timedelta(days=7 * weeks + 2))
        end = start + dt.timedelta(days=7 * weeks - 1)
        issue = min(R.TODAY, end + dt.timedelta(days=int(rng.integers(0, 31))))
        score = int(np.clip(round(rng.normal(78, 10)), 40, 100))
        lo = max(20, start.year - 2005)
        yy = int(rng.integers(lo, max(lo, start.year - 2000) + 1))
        usn = f"1RUA{yy:02d}{DEPTS[rng.integers(len(DEPTS))]}{int(rng.integers(1, 10000)):04d}"
        cno = cert_no(issue.year, int(rng.integers(0, 10 ** 6)))
        v = dict(name=sample_name(rng), weeks=weeks, start=start, end=end, score=score, grade=band(score),
                 issue=issue, cert_no=cno, usn=usn, url_id=cno)
        content = dict(course=COURSES[rng.integers(len(COURSES))])
        base = int(rng.integers(246, 253))
        style = D.sample_style(rng, SCHEMA.field_names, tint=(base, base - 3, base - 14), ink=(20, 50))
        return v, content, style

    def anchor(self, field, style):
        return LAYOUT.origin(field, "", LAYOUT.font(field), style)

    def place(self, field, text, fnt, style):
        return LAYOUT.origin(field, text, fnt, style)

    def font_scale(self, field):
        return LAYOUT.scale(field)

    def render_clean(self, values, content, style):
        img = D.paper((W, H), style)
        rosette(img, style, 480, 380, 150, 26, 9)
        rosette(img, style, 480, 380, 96, 14, 14, n=6)
        p = D.Pen(img, style)
        gold = (176, 141, 70)
        p.rect(20, 20, 940, 652, outline=gold, width=6)
        p.rect(34, 34, 926, 638, outline=gold, width=1)
        for cx, cy in [(34, 34), (926, 34), (34, 638), (926, 638)]:
            p.d.polygon([(cx + style.ox, cy - 9 + style.oy), (cx + 9 + style.ox, cy + style.oy),
                         (cx + style.ox, cy + 9 + style.oy), (cx - 9 + style.ox, cy + style.oy)], fill=gold)
        p.text(0, 56, "SKILLNEST ACADEMY", font(STIX_BOLD, 30), align="page", spacing=2)
        p.text(0, 96, "Online programmes of the Institute of Advanced Technology, Bengaluru",
               font(STIX_ITALIC, 14), align="page")
        p.rule(HEADER[0], HEADER[1], 780, t=2)
        p.text(0, 144, "CERTIFICATE OF COMPLETION", font(STIX_BOLD, 24), align="page", spacing=3)
        p.text(0, 192, "This is to certify that", font(STIX_ITALIC, 17), align="page")
        p.rule(300, 268, 660)
        p.text(0, 290, "has successfully completed the online programme", font(STIX_ITALIC, 17), align="page")
        p.text(0, 318, content["course"], font(STIX_BOLD, 22), align="page")
        f17 = font(STIX, 17)
        p.text(298, 363, "of", f17, align="r")
        p.text(346, 363, "weeks, held from", f17)
        p.text(606, 363, "to", f17)
        p.text(482, 399, "with a final score of", f17, align="r")
        p.text(532, 399, "and grade", f17)
        p.text(0, 438, "conducted online by Skillnest Academy for the Institute of Advanced Technology.",
               font(STIX_ITALIC, 14), align="page")
        f15 = font(STIX, 15)
        for y, s in [(504, "Date of Issue"), (530, "Certificate No."), (556, "USN")]:
            p.text(80, y, s, f15)
            p.text(198, y, ":", f15)
        p.text(80, 588, "Verify at: skillnest.example/verify/", font(FONT_SANS, 12))
        p.ellipse(440, 494, 524, 578, outline=gold, width=3)
        p.ellipse(450, 504, 514, 568, outline=gold, width=1)
        p.text(482, 528, "SKILLNEST", font(STIX_BOLD, 11), align="c", fill=gold)
        p.text(482, 541, "ACADEMY", font(STIX_BOLD, 9), align="c", fill=gold)
        p.squiggle(650, 528, n=22, amp=5, step=6)
        p.rule(630, 548, 820)
        p.text(725, 554, "Programme Director", f15, align="c")
        p.text(0, 618, "Issued digitally. Verify its authenticity at the address above.", font(FONT_SANS, 10),
               align="page")
        return img, D.draw_fields(img, LAYOUT, SCHEMA, values, style)

    # ------------------------------------------------ forgery value models
    def violating_value(self, f, v, rng):
        x = v[f]
        if f == "weeks":
            return int(rng.choice([w for w in range(2, 25) if w != x]))
        if f in ("start", "end"):
            for _ in range(100):
                d = x + dt.timedelta(days=int(rng.integers(1, 60)) * (1 if rng.random() < 0.5 else -1))
                if d != x:
                    return d
        if f == "score":
            if rng.random() < 0.15:
                return int(rng.integers(101, 140))
            for _ in range(100):  # a better score than the grade printed beside it
                s = int(np.clip(x + int(rng.integers(5, 30)) * (1 if rng.random() < 0.8 else -1), 40, 100))
                if band(s) != band(x):
                    return s
            return 100 if band(x) != "A+" else 45
        if f == "grade":
            i = GRADES.index(x) if x in GRADES else len(GRADES) - 1
            better = GRADES[max(0, i - 2):i] or GRADES[i + 1:i + 2]
            return better[rng.integers(len(better))]
        if f == "issue":
            u = rng.random()
            if u < 0.3:
                return R.TODAY + dt.timedelta(days=int(rng.integers(10, 400)))
            if u < 0.6:
                return v["end"] - dt.timedelta(days=int(rng.integers(5, 200)))
            return v["end"] + dt.timedelta(days=int(rng.integers(ISSUE_WINDOW + 10, 400)))
        if f in ("cert_no", "url_id"):
            if rng.random() < 0.6 or not CERT_RE.match(x):
                return typo(x, rng, [i for i, c in enumerate(x) if c.isdigit()] or [0])
            m = CERT_RE.match(x)  # a fabricated but valid number, issued in another year
            return cert_no(2000 + (int(m.group(1)) + int(rng.choice([-1, 1]))) % 100, int(rng.integers(0, 10 ** 6)))
        if f == "usn":
            u = rng.random()
            if u < 0.4 and ROLL_RE.match(x):  # joined after the programme started
                return x[:4] + f"{(v['start'].year % 100 + int(rng.integers(1, 4))) % 100:02d}" + x[6:]
            if u < 0.7:
                return x[:6] + ["CSX", "ECX", "MEE", "AIX"][rng.integers(4)] + x[9:]
            return x[:-1] if rng.random() < 0.5 else x + str(int(rng.integers(10)))
        raise KeyError(f)

    def coherent_edit(self, values, rng):
        u = rng.random()
        new = dict(values)
        if u < 0.4:  # someone else's certificate with the name changed
            for _ in range(100):
                n = sample_name(rng)
                if n != values["name"]:
                    break
            new["name"] = n
            return new, ["name"], "name"
        if u < 0.65:  # a higher score with the matching grade
            s = int(np.clip(values["score"] + int(rng.integers(3, 20)), 40, 100))
            if s == values["score"]:
                s = max(40, s - 7)
            new.update(score=s, grade=band(s) or values["grade"])
            return new, [f for f in ("score", "grade") if new[f] != values[f]], "score"
        if u < 0.85:  # a longer programme, every date moved along with it
            k = int(rng.integers(1, 5))
            new["weeks"] = values["weeks"] + k
            new["end"] = values["end"] + dt.timedelta(days=7 * k)
            new["issue"] = values["issue"] + dt.timedelta(days=7 * k)
            if new["issue"] > R.TODAY:
                new["issue"] = max(new["end"], R.TODAY) if new["end"] <= R.TODAY else new["end"]
            m = CERT_RE.match(values["cert_no"])
            if m and new["issue"].year % 100 != int(m.group(1)):
                new["cert_no"] = new["url_id"] = cert_no(new["issue"].year, int(m.group(2)))
            edited = [f.name for f in SCHEMA.fields if f.fmt(new[f.name]) != f.fmt(values[f.name])]
            return new, edited, "weeks"
        m = CERT_RE.match(values["cert_no"])  # another valid certificate number
        year = 2000 + int(m.group(1)) if m else values["issue"].year
        new["cert_no"] = new["url_id"] = cert_no(year, int(rng.integers(0, 10 ** 6)))
        return new, ["cert_no", "url_id"], "cert_no"

    def random_values(self, rng, v):
        v = dict(v)
        if rng.random() < 0.5:
            v["name"] = " ".join("".join(rng.choice(list(UPPER), int(rng.integers(1, 10))))
                                 for _ in range(int(rng.integers(1, 4))))
        if rng.random() < 0.5:
            v["weeks"] = int(rng.integers(0, 100))
        for f in ("start", "end", "issue"):
            if rng.random() < 0.5:
                v[f] = dt.date(int(rng.integers(2005, 2035)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
        if rng.random() < 0.5:
            v["score"] = int(rng.integers(0, 200))
        if rng.random() < 0.4:
            v["grade"] = GRADES[rng.integers(len(GRADES))]
        if rng.random() < 0.4:
            v["cert_no"] = f"SN{int(rng.integers(0, 100)):02d}-{int(rng.integers(0, 10 ** 6)):06d}-{chr(65 + int(rng.integers(26)))}"
        if rng.random() < 0.4:
            v["url_id"] = f"SN{int(rng.integers(0, 100)):02d}-{int(rng.integers(0, 10 ** 6)):06d}-{chr(65 + int(rng.integers(26)))}"
        if rng.random() < 0.4:
            v["usn"] = random_usn(rng)
        return v
