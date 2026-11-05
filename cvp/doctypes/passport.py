"""Specimen passport data page of 'Utopia', the fictional state used for examples in the
ICAO 9303 standard for machine-readable travel documents (TD3 format).

The visual zone prints the holder's details; the machine-readable zone (MRZ) repeats
them in two 44-character lines with ICAO check digits over the document number, the
dates of birth and expiry, the personal number, and a composite over all of them.
Rules: every printed field must equal its MRZ copy, every check digit must hold, the
expiry follows from the issue date (10 years for adults, 5 for minors) and the holder
is born before the passport was issued.

Each field is its own class, but an equality between a printed field and its MRZ copy
is symmetric: breaking it from either side gives the same signature unless the forger
forgets a check digit, and names carry no check digit at all. Which side was edited is
exactly what the visual branch must decide.
"""
import datetime as dt
import re
import string

import numpy as np

from ..core import rules as R
from ..core.render import FONT_BOLD, FONT_MONO, FONT_SANS, font
from ..core.schema import Constraint, Field, Schema
from . import _draw as D
from .base import DocType, Layout
from .invoice import flag

UPPER = string.ascii_uppercase
ALNUM = string.digits + UPPER
EARLIEST_ISSUE = dt.date(2010, 1, 1)
NUM_RE = re.compile(r"^[A-Z]{2}\d{7}$")
NAME_RE = re.compile(r"^[A-Z]+(<[A-Z]+)*<<[A-Z]+(<[A-Z]+)*<*$")


# ---------------------------------------------------------------- MRZ
def yymmdd(d):
    return d.strftime("%y%m%d")


def mrz_name(surname, given):
    return (surname.replace(" ", "<") + "<<" + given.replace(" ", "<"))[:39].ljust(39, "<")


def sex_code(s):
    return s if s in ("M", "F") else "<"


def build_mrz(v):
    """Both MRZ lines from the visual-zone values, with every check digit computed."""
    mrz1 = "P<UTO" + mrz_name(v["viz_surname"], v["viz_given"])
    return mrz1, line2(v["viz_num"], yymmdd(v["viz_dob"]), sex_code(v["viz_sex"]), yymmdd(v["viz_exp"]), v["viz_pno"])


def line2(num, dob, sex, exp, pno, fix=True, old=None):
    """Assemble MRZ line 2; with fix=False the check digits are copied from `old`
    (a careless forger), otherwise recomputed."""
    num, pno = num[:9].ljust(9, "<"), pno[:14].ljust(14, "<")
    cd = [R.icao_check(num), R.icao_check(dob), R.icao_check(exp), R.icao_check(pno)]
    if not fix and old is not None:
        cd = [old[9], old[19], old[27], old[42]]
    body = num + cd[0] + "UTO" + dob + cd[1] + sex + exp + cd[2] + pno + cd[3]
    comp = R.icao_check(body[0:10] + body[13:20] + body[21:43]) if fix or old is None else old[43]
    return body + comp


def split_name(mrz1):
    name = mrz1[5:]
    if "<<" not in name:
        return name.rstrip("<").replace("<", " "), ""
    surname, given = name.split("<<", 1)
    return surname.replace("<", " "), given.rstrip("<").replace("<", " ")


def _line(chars):
    return lambda s: s if len(s) == 44 and set(s) <= set(chars) else None


FIELDS = [
    Field("viz_num", "Passport number", ALNUM, str, lambda s: s or None),
    Field("viz_surname", "Surname", UPPER + " ", str, lambda s: s or None),
    Field("viz_given", "Given names", UPPER + " ", str, lambda s: s or None),
    Field("viz_dob", "Date of birth", ALNUM + " ", R.fmt_dmony, R.parse_dmony),
    Field("viz_sex", "Sex", "MFX", str, lambda s: s if s in ("M", "F", "X") else None),
    Field("viz_issue", "Date of issue", ALNUM + " ", R.fmt_dmony, R.parse_dmony),
    Field("viz_exp", "Date of expiry", ALNUM + " ", R.fmt_dmony, R.parse_dmony),
    Field("viz_pno", "Personal number", ALNUM, str, lambda s: s or None),
    Field("mrz1", "MRZ line 1", UPPER + "<", str, _line(UPPER + "<")),
    Field("mrz2", "MRZ line 2", ALNUM + "<", str, _line(ALNUM + "<")),
]


def _cd(a, b, c):
    return lambda v: flag(R.icao_check(v["mrz2"][a:b]) == v["mrz2"][c])


def _composite(v):
    m = v["mrz2"]
    return flag(R.icao_check(m[0:10] + m[13:20] + m[21:43]) == m[43])


def _valid_yymmdd(s):
    try:
        dt.datetime.strptime(s, "%y%m%d")
        return True
    except ValueError:
        return False


def _mrz2_format(v):
    m = v["mrz2"]
    ok = m[10:13] == "UTO" and m[20] in "MF<" and all(m[i].isdigit() for i in (9, 19, 27, 42, 43)) \
        and _valid_yymmdd(m[13:19]) and _valid_yymmdd(m[21:27])
    return flag(ok)


def _mrz1_format(v):
    m = v["mrz1"]
    return flag(m.startswith("P<UTO") and bool(NAME_RE.match(m[5:])))


def _validity(v):
    years = 10 if R.age_on(v["viz_dob"], v["viz_issue"]) >= 18 else 5
    due = R.add_years(v["viz_issue"], years) - dt.timedelta(days=1)
    d = (v["viz_exp"] - due).days
    return d != 0, d / 365.0


def _issued(v):
    d = v["viz_issue"]
    days = max(0, (d - R.TODAY).days) + max(0, (EARLIEST_ISSUE - d).days)
    return days > 0, days / 365.0


def _born(v):
    age = R.age_on(v["viz_dob"], v["viz_issue"])
    d = max(0, -age) + max(0, age - 100)
    return d > 0, d / 10.0


CONSTRAINTS = [
    Constraint("c_nf", "format", ("viz_num",), lambda v: flag(bool(NUM_RE.match(v["viz_num"]))),
               "Passport no. is two letters and seven digits", on_missing=(1.0, 1.0)),
    Constraint("c_num", "equality", ("viz_num", "mrz2"), lambda v: flag(v["viz_num"] == v["mrz2"][0:9].rstrip("<")),
               "Printed passport no. = MRZ", sensitive=("viz_num",)),
    Constraint("c_sur", "equality", ("viz_surname", "mrz1"), lambda v: flag(v["viz_surname"] == split_name(v["mrz1"])[0]),
               "Printed surname = MRZ", sensitive=("viz_surname",)),
    Constraint("c_giv", "equality", ("viz_given", "mrz1"), lambda v: flag(v["viz_given"] == split_name(v["mrz1"])[1]),
               "Printed given names = MRZ", sensitive=("viz_given",)),
    Constraint("c_dob", "equality", ("viz_dob", "mrz2"), lambda v: flag(yymmdd(v["viz_dob"]) == v["mrz2"][13:19]),
               "Printed date of birth = MRZ", sensitive=("viz_dob",)),
    Constraint("c_sex", "equality", ("viz_sex", "mrz2"), lambda v: flag(sex_code(v["viz_sex"]) == v["mrz2"][20]),
               "Printed sex = MRZ", sensitive=("viz_sex",)),
    Constraint("c_exp", "equality", ("viz_exp", "mrz2"), lambda v: flag(yymmdd(v["viz_exp"]) == v["mrz2"][21:27]),
               "Printed expiry date = MRZ", sensitive=("viz_exp",)),
    Constraint("c_pno", "equality", ("viz_pno", "mrz2"), lambda v: flag(v["viz_pno"] == v["mrz2"][28:42].rstrip("<")),
               "Printed personal no. = MRZ", sensitive=("viz_pno",)),
    Constraint("c_cd1", "checksum", ("mrz2",), _cd(0, 9, 9), "MRZ check digit: passport no.", family="c_cd"),
    Constraint("c_cd2", "checksum", ("mrz2",), _cd(13, 19, 19), "MRZ check digit: date of birth", family="c_cd"),
    Constraint("c_cd3", "checksum", ("mrz2",), _cd(21, 27, 27), "MRZ check digit: expiry date", family="c_cd"),
    Constraint("c_cd4", "checksum", ("mrz2",), _cd(28, 42, 42), "MRZ check digit: personal no.", family="c_cd"),
    Constraint("c_cmp", "checksum", ("mrz2",), _composite, "MRZ composite check digit"),
    Constraint("c_m2f", "format", ("mrz2",), _mrz2_format, "MRZ line 2 layout (state, sex, dates)"),
    Constraint("c_m1f", "format", ("mrz1",), _mrz1_format, "MRZ line 1 is P<UTO SURNAME<<GIVEN"),
    Constraint("c_val", "temporal", ("viz_dob", "viz_issue", "viz_exp"), _validity,
               "Expiry = issue + 10 years (5 for minors) - 1 day", sensitive=("viz_issue", "viz_exp")),
    Constraint("c_iss", "temporal", ("viz_issue",), _issued, "2010 <= issue date <= today"),
    Constraint("c_brn", "temporal", ("viz_dob", "viz_issue"), _born, "Born before issue (age 0-100)"),
]

SCHEMA = Schema("passport", FIELDS, CONSTRAINTS)


# ---------------------------------------------------------------- template
W, H = 864, 608
HEADER = (40, 52)
S16 = (FONT_SANS, 16)
LAYOUT = Layout({
    "viz_num": (520, 76, "l", FONT_BOLD, 16, 140),
    "viz_surname": (300, 122, "l", *S16, 300),
    "viz_given": (300, 166, "l", *S16, 400),
    "viz_dob": (300, 254, "l", *S16, 130),
    "viz_sex": (470, 254, "l", *S16, 30),
    "viz_issue": (300, 298, "l", *S16, 130),
    "viz_exp": (300, 342, "l", *S16, 130),
    "viz_pno": (470, 342, "l", *S16, 140),
    "mrz1": (146, 456, "l", FONT_MONO, 22, 580, 3, 29),
    "mrz2": (146, 500, "l", FONT_MONO, 22, 580, 3, 29),
})
SURNAMES = ["ERIKSSON", "MARLOWE", "OKAFOR", "KOVACS", "TANAKA", "SILVA", "NAIR", "HASSAN", "PETROV",
            "LINDQVIST", "MORENO", "ADEYEMI", "CHEN", "DUBOIS", "ROSSI", "NOVAK", "IYER", "BERG", "LARSEN",
            "FISCHER", "HOLM", "VARGA", "SANTOS", "KHAN", "REYES", "QUINN", "ABARA", "MENON", "SATO", "COSTA"]
GIVEN = ["ANNA MARIA", "JOHN", "PRIYA", "KENJI", "LUCIA", "OMAR", "ELENA", "TOMAS", "AMARA", "LEO", "SOFIA",
         "ARJUN", "MEI LING", "PAUL ANDRE", "NADIA", "VIKTOR", "ISABEL", "RAVI KUMAR", "HANNA", "SAMUEL",
         "AYESHA", "MATEO", "KAVYA", "OLIVER JAMES", "YUKI", "FATIMA", "DANIEL", "ROSA", "IVAN", "ZARA"]
PLACES = ["ZENITH", "AURORA", "MERIDIAN", "HAVEN", "CASCADE", "SOLACE", "ARCADIA", "LUMEN"]
MRZ_ALT = [(FONT_MONO.replace("Mono.ttf", "Mono-Bold.ttf"), 16), (FONT_MONO, 15),
           (FONT_MONO.replace("Mono.ttf", "Mono-Oblique.ttf"), 16)]  # x 22/16: 22, 21, 22 px
WEIGHTS = {"viz_num": 1.0, "viz_surname": 1.5, "viz_given": 1.5, "viz_dob": 2.0, "viz_sex": 0.5,
           "viz_issue": 1.0, "viz_exp": 2.0, "viz_pno": 0.5, "mrz1": 1.0, "mrz2": 2.0}


def expiry_for(dob, issue):
    return R.add_years(issue, 10 if R.age_on(dob, issue) >= 18 else 5) - dt.timedelta(days=1)


def sample_number(rng):
    return "".join(rng.choice(list(UPPER), 2)) + f"{int(rng.integers(0, 10 ** 7)):07d}"


def sample_pno(rng):
    return "".join(rng.choice(list(UPPER), 2)) + f"{int(rng.integers(0, 10 ** 6)):06d}" + str(rng.choice(list(UPPER)))


class Passport(DocType):
    name = "passport"
    title = "Passport (Utopia specimen)"
    schema = SCHEMA
    page = (W, H)
    slots = LAYOUT.slots()
    wide = LAYOUT.wide()
    header_rule = HEADER
    weights = WEIGHTS
    date_fields = ("viz_dob", "viz_issue", "viz_exp")
    class_labels = {f.name: f.label for f in FIELDS}
    unary_labels = ("", "")

    def sample_spec(self, rng):
        issue = R.random_date(rng, EARLIEST_ISSUE, R.TODAY)
        age = int(np.clip(rng.normal(36, 16), 0, 85))
        dob = R.add_years(issue, -age) - dt.timedelta(days=int(rng.integers(1, 365)))
        v = dict(viz_num=sample_number(rng), viz_surname=SURNAMES[rng.integers(len(SURNAMES))],
                 viz_given=GIVEN[rng.integers(len(GIVEN))], viz_dob=dob, viz_sex=str(rng.choice(["M", "F"])),
                 viz_issue=issue, viz_exp=expiry_for(dob, issue), viz_pno=sample_pno(rng))
        v["mrz1"], v["mrz2"] = build_mrz(v)
        content = dict(place=PLACES[rng.integers(len(PLACES))], authority="PASSPORT OFFICE " + PLACES[rng.integers(len(PLACES))])
        base = int(rng.integers(240, 250))
        style = D.sample_style(rng, SCHEMA.field_names, tint=(base - 6, base, base - 2), ink=(15, 45))
        return v, content, style

    def anchor(self, field, style):
        return LAYOUT.origin(field, "", LAYOUT.font(field), style)

    def place(self, field, text, fnt, style):
        return LAYOUT.origin(field, text, fnt, style)

    def font_scale(self, field):
        return LAYOUT.scale(field)

    def alt_fonts(self, field):
        return MRZ_ALT if field in ("mrz1", "mrz2") else None

    def render_clean(self, values, content, style):
        img = D.paper((W, H), style)
        D.waves(img, style, shift=(22, 10, 18), n=56, area=(0, 60, W, 432))
        p = D.Pen(img, style)
        p.rect(0, 436, W, 552, fill=tuple(min(255, c + 4) for c in style.tint))  # the clear MRZ band
        lab, bold = font(FONT_SANS, 10), font(FONT_BOLD, 20)
        p.text(40, 16, "REPUBLIC OF UTOPIA", bold)
        p.text(824, 16, "PASSPORT", bold, align="r")
        p.rule(HEADER[0], HEADER[1], 824, t=2)
        p.portrait(40, 70, 236, 330, shade=tuple(c - 28 for c in style.tint))
        for x, y, s in [(300, 62, "Type"), (360, 62, "Code of State"), (520, 62, "Passport No."),
                        (300, 108, "Surname"), (300, 152, "Given names"), (300, 196, "Nationality"),
                        (300, 240, "Date of birth"), (470, 240, "Sex"), (560, 240, "Place of birth"),
                        (300, 284, "Date of issue"), (470, 284, "Authority"), (300, 328, "Date of expiry"),
                        (470, 328, "Personal No."), (300, 372, "Holder's signature")]:
            p.text(x, y, s, lab)
        p.text(300, 76, "P", font(FONT_BOLD, 16))
        p.text(360, 76, "UTO", font(FONT_BOLD, 16))
        p.text(300, 210, "UTOPIAN", font(FONT_SANS, 16))
        p.text(560, 254, content["place"], font(FONT_SANS, 16))
        p.text(470, 298, content["authority"], font(FONT_SANS, 13))
        p.squiggle(304, 398, n=20, amp=5, step=7)
        return img, D.draw_fields(img, LAYOUT, SCHEMA, values, style)

    # ------------------------------------------------ forgery value models
    def violating_value(self, f, v, rng):
        x = v[f]
        if f == "viz_num":
            if rng.random() < 0.15:
                return x + str(int(rng.integers(10)))
            return x[:2] + "".join(c if rng.random() > 0.3 else str(int(rng.integers(10))) for c in x[2:]) \
                if rng.random() < 0.3 else x[:-1] + str((int(x[-1]) + int(rng.integers(1, 10))) % 10 if x[-1].isdigit() else 7)
        if f in ("viz_surname", "viz_given"):
            pool = SURNAMES if f == "viz_surname" else GIVEN
            return rng.choice([n for n in pool if n != x])
        if f == "viz_dob":
            return R.add_years(x, int(rng.choice([-12, -8, -5, -3, -2, 2, 3, 5]))) + dt.timedelta(days=int(rng.integers(-40, 41)))
        if f == "viz_sex":
            return "F" if x == "M" else "M"
        if f == "viz_issue":
            return x + dt.timedelta(days=int(rng.integers(30, 1500)) * (1 if rng.random() < 0.5 else -1))
        if f == "viz_exp":
            return R.add_years(x, int(rng.integers(1, 6)))
        if f == "viz_pno":
            return x[:2] + f"{(int(x[2:8]) + int(rng.integers(1, 10 ** 5))) % 10 ** 6:06d}" + x[8:] \
                if x[2:8].isdigit() else x[:-1] + "Q"
        if f == "mrz1":
            sur, giv = split_name(x)
            if rng.random() < 0.5:
                sur = rng.choice([n for n in SURNAMES if n != sur])
            else:
                giv = rng.choice([n for n in GIVEN if n != giv])
            return x[:5] + mrz_name(sur, giv)
        if f == "mrz2":
            part = str(rng.choice(["num", "dob", "dob", "sex", "exp", "exp", "pno"]))
            fix = rng.random() < 0.5  # a careful forger recomputes the check digits
            num, dob, sex, exp, pno = x[0:9], x[13:19], x[20], x[21:27], x[28:42]
            if part == "num":
                num = num[:-1] + str((int(num[-1]) + 1) % 10) if num[-1].isdigit() else num[:-1] + "1"
            elif part == "dob":
                yy = (int(dob[:2]) + int(rng.choice([-5, -3, -2, 2, 3]))) % 100 if dob[:2].isdigit() else 70
                dob = f"{yy:02d}" + dob[2:]
            elif part == "sex":
                sex = "F" if sex == "M" else "M"
            elif part == "exp":
                yy = (int(exp[:2]) + int(rng.integers(1, 6))) % 100 if exp[:2].isdigit() else 30
                exp = f"{yy:02d}" + exp[2:]
            else:
                pno = pno[:2] + str(rng.choice([d for d in string.digits if d != pno[2]])) + pno[3:]
            return line2(num.rstrip("<"), dob, sex, exp, pno.rstrip("<"), fix=fix, old=x)
        raise KeyError(f)

    def coherent_edit(self, values, rng):
        u = rng.random()
        new = dict(values)
        if u < 0.3:  # a different date of birth on both zones, check digits recomputed
            adult = R.age_on(values["viz_dob"], values["viz_issue"]) >= 18
            for _ in range(100):
                dob = R.add_years(values["viz_dob"], int(rng.choice([-6, -4, -3, -2, -1, 1, 2, 3])))
                dob += dt.timedelta(days=int(rng.integers(-60, 61)))
                if dob < values["viz_issue"] and (R.age_on(dob, values["viz_issue"]) >= 18) == adult:
                    break
            new["viz_dob"] = dob
            primary = "viz_dob"
        elif u < 0.6:  # a different name on both zones
            if rng.random() < 0.5:
                new["viz_surname"] = rng.choice([n for n in SURNAMES if n != values["viz_surname"]])
                primary = "viz_surname"
            else:
                new["viz_given"] = rng.choice([n for n in GIVEN if n != values["viz_given"]])
                primary = "viz_given"
        elif u < 0.85:  # renewed: a later issue date and the expiry that follows from it
            lo = values["viz_issue"] + dt.timedelta(days=30)
            issue = R.random_date(rng, lo, R.TODAY) if lo < R.TODAY else values["viz_issue"] - dt.timedelta(days=400)
            new["viz_issue"] = issue
            new["viz_exp"] = expiry_for(values["viz_dob"], issue)
            primary = "viz_exp"
        else:  # another passport number
            new["viz_num"] = sample_number(rng)
            primary = "viz_num"
        try:
            new["mrz1"], new["mrz2"] = build_mrz(new)
        except (AttributeError, ValueError):  # randomised OCR-training values may not rebuild
            pass
        edited = [f.name for f in SCHEMA.fields if f.fmt(new[f.name]) != f.fmt(values[f.name])]
        return new, edited, primary

    def random_values(self, rng, v):
        v = dict(v)

        def word(lo, hi):
            return "".join(rng.choice(list(UPPER), int(rng.integers(lo, hi))))
        if rng.random() < 0.5:
            v["viz_surname"] = word(2, 13)
        if rng.random() < 0.5:
            v["viz_given"] = " ".join(word(2, 8) for _ in range(int(rng.integers(1, 3))))
        for f in ("viz_dob", "viz_issue", "viz_exp"):
            if rng.random() < 0.5:
                v[f] = dt.date(int(rng.integers(1925, 2040)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
        if rng.random() < 0.3:
            v["viz_sex"] = str(rng.choice(["M", "F", "X"]))
        if rng.random() < 0.5:
            v["viz_num"] = "".join(rng.choice(list(ALNUM), 9))
        if rng.random() < 0.5:
            v["viz_pno"] = "".join(rng.choice(list(ALNUM), int(rng.integers(6, 12))))
        v["mrz1"], v["mrz2"] = build_mrz(v)
        if rng.random() < 0.3:
            v["mrz1"] = "".join(rng.choice(list(UPPER + "<<<<"), 44))
        if rng.random() < 0.3:
            v["mrz2"] = "".join(rng.choice(list(ALNUM + "<<<"), 44))
        return v
