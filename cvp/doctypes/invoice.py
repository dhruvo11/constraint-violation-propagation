"""Indian GST tax invoice.

Up to four line items (qty x rate = amount), the taxable value, CGST + SGST for an
intra-state supply or IGST for an inter-state one, a round-off to whole rupees, the total
and the total in words. The seller and buyer GSTINs carry check characters, the buyer's
state must be the place of supply, and the invoice number carries the financial year of
the invoice date. Money is integer paise throughout, so every rule is exact.

Constraint structure: quantity and rate of a line meet only in that line's product, so
each line's (qty, rate) pair is a Proposition-1 class; every other field is a singleton,
but several pairs of singletons share a signature (a date and an invoice number both
break only the financial-year rule, say), which the visual branch has to resolve.
"""
import datetime as dt
import re
import string

import numpy as np

from ..core import rules as R
from ..core.render import FONT_BOLD, FONT_SANS, font
from ..core.schema import Constraint, Field, Schema
from . import _draw as D
from .base import BLANK, DocType, Layout

N = 4
QTY = [f"qty{k}" for k in range(1, N + 1)]
RATE = [f"rate{k}" for k in range(1, N + 1)]
AMT = [f"amt{k}" for k in range(1, N + 1)]
TAXES = ["cgst", "sgst", "igst"]
GST_START = dt.date(2017, 7, 1)
MAX_TOTAL = 99_999_00  # paise; keeps the amount in words on one line
INVNO_RE = re.compile(r"^[A-Z]{2,4}/(\d{2}-\d{2})/\d{4}$")
ALNUM = string.digits + string.ascii_uppercase
MONEY = string.digits + ",."


# ---------------------------------------------------------------- fields
def _text(s):
    return s or None


def _int(s):
    return int(s) if s.isdigit() else None


FIELDS = [
    Field("seller_gstin", "Seller GSTIN", ALNUM, str, _text),
    Field("inv_no", "Invoice number", ALNUM + "/-", str, _text),
    Field("date", "Invoice date", string.digits + "/", R.fmt_dmy, R.parse_dmy),
    Field("buyer_gstin", "Buyer GSTIN", ALNUM, str, _text),
    Field("pos", "Place of supply", ALNUM + "- ", str, _text),
    *[Field(q, f"Quantity, line {k}", string.digits, D.blank_or(str), D.parse_or_blank(_int))
      for k, q in enumerate(QTY, 1)],
    *[Field(r, f"Rate, line {k}", MONEY, D.blank_or(R.inr), D.parse_or_blank(R.parse_inr))
      for k, r in enumerate(RATE, 1)],
    *[Field(a, f"Amount, line {k}", MONEY, D.blank_or(R.inr), D.parse_or_blank(R.parse_inr))
      for k, a in enumerate(AMT, 1)],
    Field("subtotal", "Taxable value", MONEY, R.inr, R.parse_inr),
    Field("cgst", "CGST", MONEY, R.inr, R.parse_inr),
    Field("sgst", "SGST", MONEY, R.inr, R.parse_inr),
    Field("igst", "IGST", MONEY, R.inr, R.parse_inr),
    Field("round_off", "Round off", MONEY + "-", R.inr, R.parse_inr),
    Field("total", "Total", MONEY, R.inr, R.parse_inr),
    Field("words", "Amount in words", string.ascii_uppercase + " ", R.rupees_words, R.parse_rupees_words),
]


# ---------------------------------------------------------------- constraints
def flag(ok):
    return not ok, float(not ok)


def _gstin(f):
    return lambda v: flag(R.gstin_valid(v[f]))


def _inv_format(v):
    return flag(bool(INVNO_RE.match(v["inv_no"])))


def _fy(v):
    m = re.search(r"/(\d{2}-\d{2})/", v["inv_no"])
    return flag(bool(m) and m.group(1) == R.financial_year(v["date"]))


def _date(v):
    d = v["date"]
    days = max(0, (d - R.TODAY).days) + max(0, (GST_START - d).days)
    return days > 0, days / 365.0


def _pos(v):
    m = re.fullmatch(r"(\d{2})-([A-Z ]+)", v["pos"])
    return flag(bool(m) and R.STATES.get(m.group(1)) == m.group(2))


def _supply(v):
    return flag(v["buyer_gstin"][:2] == v["pos"][:2])


def line_check(q, r, a, scale=100_000.0):
    """qty x rate = amount; an unused line has all three blank."""
    def check(v):
        blank = [v[q] == BLANK, v[r] == BLANK, v[a] == BLANK]
        if all(blank):
            return False, 0.0
        if any(blank):
            return True, 1.0
        d = v[q] * v[r] - v[a]
        return d != 0, d / scale
    return check


def sum_check(parts, total, scale=100_000.0):
    def check(v):
        d = sum(v[a] for a in parts if v[a] != BLANK) - v[total]
        return d != 0, d / scale
    return check


def intra_state(v):
    return v["seller_gstin"][:2] == v["pos"][:2]


def _tax(f, pct):
    def check(v):
        due = R.pct_of(v["subtotal"], pct) if (f == "igst") != intra_state(v) else 0
        d = v[f] - due
        return d != 0, d / 10_000.0
    return check


def _total(v):
    d = v["subtotal"] + v["cgst"] + v["sgst"] + v["igst"] + v["round_off"] - v["total"]
    return d != 0, d / 10_000.0


def _round(v):
    d = max(0, abs(v["round_off"]) - 50)
    return d > 0, d / 100.0


def _whole(v):
    d = v["total"] % 100
    return d != 0, d / 100.0


def _words(v):
    d = v["words"] * 100 - v["total"]
    return d != 0, d / 100_000.0


SUMMED = ("subtotal", "cgst", "sgst", "igst", "round_off")
CONSTRAINTS = [
    Constraint("c_gs", "checksum", ("seller_gstin",), _gstin("seller_gstin"), "Seller GSTIN check character valid"),
    Constraint("c_gb", "checksum", ("buyer_gstin",), _gstin("buyer_gstin"), "Buyer GSTIN check character valid"),
    Constraint("c_no", "format", ("inv_no",), _inv_format, "Invoice no. is XX/YY-YY/NNNN", on_missing=(1.0, 1.0)),
    Constraint("c_fy", "logical", ("inv_no", "date"), _fy, "Invoice no. carries the date's financial year"),
    Constraint("c_dt", "temporal", ("date",), _date, "1 Jul 2017 <= invoice date <= today"),
    Constraint("c_ps", "format", ("pos",), _pos, "Place-of-supply code matches its state"),
    Constraint("c_sp", "logical", ("buyer_gstin", "pos"), _supply, "Buyer's GSTIN state = place of supply"),
    *[Constraint(f"c_ln{k}", "arithmetic", (q, r, a), line_check(q, r, a), f"Qty x Rate = Amount (line {k})",
                 sensitive=(q, r, a), exchangeable=((q, r),), family="c_ln")
      for k, (q, r, a) in enumerate(zip(QTY, RATE, AMT), 1)],
    Constraint("c_sum", "arithmetic", tuple(AMT) + ("subtotal",), sum_check(AMT, "subtotal"),
               "Sum of line amounts = taxable value", sensitive=tuple(AMT) + ("subtotal",),
               exchangeable=(tuple(AMT),)),
    Constraint("c_cg", "arithmetic", ("seller_gstin", "pos", "subtotal", "cgst"), _tax("cgst", 9),
               "CGST = 9% of taxable value (intra-state), else 0", sensitive=("cgst",)),
    Constraint("c_sg", "arithmetic", ("seller_gstin", "pos", "subtotal", "sgst"), _tax("sgst", 9),
               "SGST = 9% of taxable value (intra-state), else 0", sensitive=("sgst",)),
    Constraint("c_ig", "arithmetic", ("seller_gstin", "pos", "subtotal", "igst"), _tax("igst", 18),
               "IGST = 18% of taxable value (inter-state), else 0", sensitive=("igst",)),
    Constraint("c_tot", "arithmetic", SUMMED + ("total",), _total, "Taxable value + taxes + round-off = total",
               sensitive=SUMMED + ("total",), exchangeable=(SUMMED,)),
    Constraint("c_rnd", "range", ("round_off",), _round, "|round-off| <= 0.50"),
    Constraint("c_whl", "format", ("total",), _whole, "Total in whole rupees"),
    Constraint("c_wd", "logical", ("total", "words"), _words, "Amount in words = total",
               sensitive=("total", "words")),
]

SCHEMA = Schema("invoice", FIELDS, CONSTRAINTS,
                groups={f"line{k}": [q, r] for k, (q, r) in enumerate(zip(QTY, RATE), 1)})


# ---------------------------------------------------------------- template
W, H = 640, 896
HEADER = (40, 114)
ROW_Y = [302 + 34 * k for k in range(N)]
TOT_Y = {"subtotal": 448, "cgst": 474, "sgst": 500, "igst": 526, "round_off": 552}
LAYOUT = Layout({
    "seller_gstin": (94, 89, "l", FONT_SANS, 13, 170),
    "inv_no": (128, 153, "l", FONT_SANS, 13, 136),
    "date": (482, 153, "l", FONT_SANS, 13, 100),
    "buyer_gstin": (94, 243, "l", FONT_SANS, 13, 170),
    "pos": (380, 204, "l", FONT_SANS, 12, 150),
    **{q: (388, y, "r", FONT_SANS, 13, 50) for q, y in zip(QTY, ROW_Y)},
    **{r: (488, y, "r", FONT_SANS, 13, 92) for r, y in zip(RATE, ROW_Y)},
    **{a: (596, y, "r", FONT_SANS, 13, 104) for a, y in zip(AMT, ROW_Y)},
    **{f: (596, y, "r", FONT_SANS, 13, 120) for f, y in TOT_Y.items()},
    "total": (596, 586, "r", FONT_BOLD, 15, 140),
    "words": (40, 640, "l", FONT_SANS, 12, 545),
})

SELLERS = ["SRI GANESH ELECTRONICS", "NAVYA TRADERS", "SHREE BALAJI ENTERPRISES", "KAVERI OFFICE SUPPLIES",
           "METRO HARDWARE STORES", "ANNAPURNA KITCHENWARE", "SUNRISE COMPUTERS", "VINAYAKA STATIONERS",
           "GREENLEAF FURNISHINGS", "PRAGATHI APPLIANCES", "LAKSHMI TEXTILES", "CITYLINE DIGITAL"]
BUYERS = ["ORBIT SOFTWARE SOLUTIONS PVT LTD", "BLUEWAVE LOGISTICS", "NORTHSTAR CONSULTING LLP",
          "PIXEL PRINTS AND GRAPHICS", "RIVERSIDE CAFE", "UNITY HEALTHCARE SUPPLIES", "VERTEX ENGINEERING WORKS",
          "SAPPHIRE SCHOOL OF MUSIC", "HORIZON EVENTS", "TRUEVALUE CLINIC"]
CITIES = {"03": ("LUDHIANA", "141"), "06": ("GURUGRAM", "122"), "07": ("NEW DELHI", "110"),
          "08": ("JAIPUR", "302"), "09": ("LUCKNOW", "226"), "10": ("PATNA", "800"), "19": ("KOLKATA", "700"),
          "21": ("BHUBANESWAR", "751"), "24": ("AHMEDABAD", "380"), "27": ("PUNE", "411"),
          "29": ("BENGALURU", "560"), "30": ("PANAJI", "403"), "32": ("KOCHI", "682"), "33": ("CHENNAI", "600"),
          "36": ("HYDERABAD", "500")}
STREETS = ["MG ROAD", "2ND CROSS, JAYANAGAR", "STATION ROAD", "MAIN BAZAAR", "RING ROAD", "CHURCH STREET",
           "NEHRU NAGAR", "GANDHI MARKET", "INDUSTRIAL AREA", "TEMPLE STREET"]
ITEMS = [("LED Monitor 24 inch", "8528", (8500, 14500)), ("Wireless Keyboard", "8471", (900, 2500)),
         ("Optical Mouse", "8471", (250, 900)), ("Laser Printer", "8443", (9000, 18000)),
         ("A4 Paper Ream (500)", "4802", (280, 420)), ("Office Chair", "9401", (3500, 9000)),
         ("Steel Almirah", "9403", (8000, 15000)), ("LED Bulb 9W", "8539", (90, 180)),
         ("Extension Board", "8536", (350, 900)), ("Water Purifier", "8421", (6000, 15000)),
         ("Mixer Grinder", "8509", (2500, 6000)), ("Pressure Cooker 5L", "7615", (1500, 3200)),
         ("Ceiling Fan", "8414", (1800, 4200)), ("UPS 1 kVA", "8504", (4500, 8500)),
         ("Network Switch 8 Port", "8517", (1200, 3500)), ("Whiteboard 4x3 ft", "9610", (1800, 3500)),
         ("Toner Cartridge", "8443", (1800, 4500)), ("HDMI Cable 2m", "8544", (250, 700))]
WEIGHTS = {"seller_gstin": 1.0, "inv_no": 1.0, "date": 1.5, "buyer_gstin": 1.0, "pos": 1.0,
           **{q: 1.5 for q in QTY}, **{r: 1.5 for r in RATE}, **{a: 2.0 for a in AMT},
           "subtotal": 1.5, "cgst": 1.0, "sgst": 1.0, "igst": 1.0, "round_off": 0.5, "total": 3.0, "words": 1.0}


def address(rng, state):
    city, pin = CITIES[state]
    return f"No. {int(rng.integers(1, 400))}, {STREETS[rng.integers(len(STREETS))]}, {city} {pin} " \
           f"{int(rng.integers(0, 100)):03d}"


def settle(v):
    """Recompute everything that follows from the lines: amounts, taxable value, taxes,
    round-off to whole rupees, total and words."""
    v = dict(v)
    for q, r, a in zip(QTY, RATE, AMT):
        if v[q] != BLANK:
            v[a] = v[q] * v[r]
    v["subtotal"] = sum(v[a] for a in AMT if v[a] != BLANK)
    intra = intra_state(v)
    v["cgst"] = v["sgst"] = R.pct_of(v["subtotal"], 9) if intra else 0
    v["igst"] = 0 if intra else R.pct_of(v["subtotal"], 18)
    gross = v["subtotal"] + v["cgst"] + v["sgst"] + v["igst"]
    v["total"] = (gross + 50) // 100 * 100
    v["round_off"] = v["total"] - gross
    v["words"] = v["total"] // 100
    return v


def typo(s, rng, positions):
    """Change one character to another of the same kind (digit for digit, letter for letter)."""
    s = list(s)
    i = int(positions[rng.integers(len(positions))])
    pool = string.digits if s[i].isdigit() else string.ascii_uppercase
    s[i] = rng.choice([c for c in pool if c != s[i]])
    return "".join(s)


class Invoice(DocType):
    name = "invoice"
    title = "GST tax invoice"
    schema = SCHEMA
    page = (W, H)
    slots = LAYOUT.slots()
    wide = LAYOUT.wide()
    header_rule = HEADER
    weights = WEIGHTS
    date_fields = ("date",)
    class_labels = {"seller_gstin": "Seller GSTIN", "inv_no": "Invoice number", "date": "Invoice date",
                    "buyer_gstin": "Buyer GSTIN", "pos": "Place of supply",
                    **{f"line{k}": f"Qty or rate, line {k}" for k in range(1, N + 1)},
                    **{a: f"Amount, line {k}" for k, a in enumerate(AMT, 1)},
                    "subtotal": "Taxable value", "cgst": "CGST", "sgst": "SGST", "igst": "IGST",
                    "round_off": "Round off", "total": "Total", "words": "Amount in words"}
    unary_labels = ("", "")

    def sample_spec(self, rng):
        states = list(R.STATES)
        s_state = states[rng.integers(len(states))]
        b_state = s_state if rng.random() < 0.6 else rng.choice([s for s in states if s != s_state])
        date = R.random_date(rng, dt.date(2018, 4, 1), R.TODAY)
        seller = SELLERS[rng.integers(len(SELLERS))]
        prefix = "".join(w[0] for w in seller.split()[:3])
        v = {"seller_gstin": R.make_gstin(rng, s_state), "buyer_gstin": R.make_gstin(rng, b_state),
             "pos": f"{b_state}-{R.STATES[b_state]}", "date": date,
             "inv_no": f"{prefix}/{R.financial_year(date)}/{int(rng.integers(1, 10000)):04d}"}
        n = int(rng.choice([2, 3, 4], p=[0.3, 0.4, 0.3]))
        picks = rng.choice(len(ITEMS), n, replace=False)
        for _ in range(50):
            for k in range(N):
                if k < n:
                    lo, hi = ITEMS[picks[k]][2]
                    v[RATE[k]] = int(rng.integers(lo, hi)) * 100 + (50 if rng.random() < 0.15 else 0)
                    cheap = hi < 1000
                    v[QTY[k]] = int(rng.choice([1, 2, 3, 4, 5, 6, 10, 12, 20, 25] if cheap else [1, 1, 1, 2, 2, 3, 4, 5]))
                else:
                    v[QTY[k]] = v[RATE[k]] = v[AMT[k]] = BLANK
            v = settle(v)
            if v["total"] <= MAX_TOTAL:
                break
            v = {k2: v[k2] for k2 in ("seller_gstin", "buyer_gstin", "pos", "date", "inv_no")}
        content = dict(seller=seller, address=address(rng, s_state),
                       contact=f"Ph: +91 {int(rng.integers(70000, 99999))} {int(rng.integers(10000, 99999))}"
                               f"   Email: accounts@{seller.split()[0].lower()}{seller.split()[1].lower()}.example",
                       buyer=BUYERS[rng.integers(len(BUYERS))], buyer_address=address(rng, b_state),
                       items=[[ITEMS[picks[k]][0], ITEMS[picks[k]][1]] for k in range(n)],
                       bank=f"A/c No. {int(rng.integers(10**11, 10**12))}   IFSC: UTOB000{int(rng.integers(1000, 9999))}"
                            f"   Branch: {CITIES[s_state][0].title()}")
        style = D.sample_style(rng, SCHEMA.field_names)
        return v, content, style

    def anchor(self, field, style):
        return LAYOUT.origin(field, "", LAYOUT.font(field), style)

    def place(self, field, text, fnt, style):
        return LAYOUT.origin(field, text, fnt, style)

    def font_scale(self, field):
        return LAYOUT.scale(field)

    def render_clean(self, values, content, style):
        img = D.paper((W, H), style)
        img = D.watermark(img, style, content["seller"], font(FONT_BOLD, 24), step=(70, 160), start=(-140, 140))
        p = D.Pen(img, style)
        lab, bold, small = font(FONT_SANS, 12), font(FONT_BOLD, 12), font(FONT_SANS, 11)
        p.text(0, 24, content["seller"], font(FONT_BOLD, 20), align="page")
        p.text(0, 52, content["address"], small, align="page")
        p.text(0, 68, content["contact"], small, align="page")
        p.text(40, 90, "GSTIN:", bold)
        p.rule(HEADER[0], HEADER[1], 600, t=2)
        p.text(0, 122, "TAX INVOICE", font(FONT_BOLD, 16), align="page", spacing=2)
        p.text(40, 154, "Invoice No.", lab)
        p.text(380, 154, "Invoice Date", lab)
        p.text(40, 186, "Bill To:", bold)
        p.text(40, 204, content["buyer"], font(FONT_SANS, 13))
        p.text(40, 222, content["buyer_address"], small)
        p.text(40, 244, "GSTIN:", bold)
        p.text(380, 186, "Place of Supply:", bold)
        p.text(380, 226, "Reverse Charge: No", small)
        p.rule(40, 268, 600)
        hb = font(FONT_BOLD, 11)
        for x, s, a in [(44, "#", "l"), (66, "Description of Goods", "l"), (290, "HSN", "l"), (388, "Qty", "r"),
                        (488, "Rate", "r"), (596, "Amount", "r")]:
            p.text(x, 274, s, hb, align=a)
        p.rule(40, 292, 600)
        for k, (desc, hsn) in enumerate(content["items"]):
            y = ROW_Y[k] + 1
            p.text(46, y, str(k + 1), lab)
            p.text(66, y, desc, lab)
            p.text(290, y, hsn, lab)
        p.rule(40, 438, 600)
        for f, s in [("subtotal", "Taxable Value"), ("cgst", "CGST @ 9%"), ("sgst", "SGST @ 9%"),
                     ("igst", "IGST @ 18%"), ("round_off", "Round Off")]:
            p.text(340, TOT_Y[f] + 1, s, lab)
        p.rule(340, 578, 600)
        p.text(340, 588, "TOTAL", font(FONT_BOLD, 14))
        p.rule(40, 614, 600)
        p.text(40, 622, "Amount Chargeable (in words):", small)
        p.rule(40, 668, 600)
        p.text(40, 680, "Bank Details", font(FONT_BOLD, 11))
        p.text(40, 696, content["bank"], small)
        p.text(40, 726, "Declaration: We declare that this invoice shows the actual price of the goods",
               font(FONT_SANS, 10))
        p.text(40, 740, "described and that all particulars are true and correct.", font(FONT_SANS, 10))
        p.text(596, 772, "for " + content["seller"], font(FONT_BOLD, 11), align="r")
        p.squiggle(470, 812, n=18)
        p.text(596, 836, "Authorised Signatory", small, align="r")
        p.text(0, 868, "This is a computer generated invoice.", font(FONT_SANS, 10), align="page")
        return img, D.draw_fields(img, LAYOUT, SCHEMA, values, style)

    # ------------------------------------------------ forgery value models
    def pick_field(self, values, rng):
        cand = [f for f in SCHEMA.field_names
                if values.get(f) != BLANK and not (f in TAXES and values[f] == 0)]
        p = np.array([WEIGHTS[f] for f in cand], float)
        return cand[rng.choice(len(cand), p=p / p.sum())]

    def violating_value(self, f, v, rng):
        x = v[f]
        if f in ("seller_gstin", "buyer_gstin"):
            if rng.random() < 0.5:  # one character changed: the check character no longer matches
                return typo(x, rng, range(2, 14))
            pos = v["pos"][:2]
            if f == "buyer_gstin":  # moved to another state, check character recomputed
                state = rng.choice([s for s in R.STATES if s != pos])
            elif x[:2] == pos:  # intra-state supply becomes inter-state: the tax split breaks
                state = rng.choice([s for s in R.STATES if s != pos])
            else:
                state = pos
            first14 = state + x[2:14]
            return first14 + R.gstin_check(first14)
        if f == "inv_no":
            m = re.search(r"/(\d{2})-(\d{2})/", x)
            if m and rng.random() < 0.8:
                y = (int(m.group(1)) + int(rng.choice([-2, -1, 1]))) % 100
                return x[:m.start()] + f"/{y:02d}-{(y + 1) % 100:02d}/" + x[m.end():]
            return x[:-4] + str(int(rng.integers(0, 1000))) if rng.random() < 0.5 else x + str(int(rng.integers(10)))
        if f == "date":
            if rng.random() < 0.2:
                return R.TODAY + dt.timedelta(days=int(rng.integers(20, 400)))
            for _ in range(100):
                d = R.add_years(x, int(rng.choice([-2, -1, 1])))
                d += dt.timedelta(days=int(rng.integers(-20, 21)))
                if R.financial_year(d) != R.financial_year(x) or d > R.TODAY or d < GST_START:
                    return d
            return R.TODAY + dt.timedelta(days=30)
        if f == "pos":
            code = x[:2]
            if rng.random() < 0.7 or code not in R.STATES:
                s = rng.choice([s for s in R.STATES if s != code])
                return f"{s}-{R.STATES[s]}"
            return f"{code}-{R.STATES[rng.choice([s for s in R.STATES if s != code])]}"
        if f in QTY:
            return int(rng.choice([n for n in range(1, 60) if n != x]))
        if f in RATE or f in AMT or f in ("subtotal",) + tuple(TAXES):
            for _ in range(100):
                step = int(rng.integers(5, 300)) * 100 if f not in TAXES else int(rng.integers(100, 50_000))
                new = x + step * (1 if rng.random() < 0.8 else -1)
                if 0 < new < MAX_TOTAL and new != x:
                    return new
            return x + 1000
        if f == "round_off":
            for _ in range(100):
                new = int(rng.integers(-99, 100))
                if new != x:
                    return new
        if f == "total":
            for _ in range(100):
                new = x + int(rng.integers(1, 90)) * 10_000 * (1 if rng.random() < 0.8 else -1)
                new += int(rng.integers(1, 100)) if rng.random() < 0.2 else 0
                if 0 < new < MAX_TOTAL and new != x:
                    return new
            return x + 10_000
        if f == "words":
            for _ in range(100):
                new = x + int(rng.integers(1, 9000)) * (1 if rng.random() < 0.8 else -1)
                if 0 < new < MAX_TOTAL // 100:
                    return new
            return x + 1
        raise KeyError(f)

    def coherent_edit(self, values, rng):
        u = rng.random()
        rows = [k for k in range(N) if values[QTY[k]] != BLANK]
        if u < 0.6 and rows:  # inflate one line and carry it through the whole bill
            for _ in range(20):
                k = rows[rng.integers(len(rows))]
                new = dict(values)
                if rng.random() < 0.5:
                    new[QTY[k]] = values[QTY[k]] + int(rng.integers(1, 6))
                    primary = QTY[k]
                else:
                    new[RATE[k]] = values[RATE[k]] + int(rng.integers(1, 40)) * 5000
                    primary = RATE[k]
                new = settle(new)
                if new["total"] <= MAX_TOTAL:
                    edited = [f.name for f in SCHEMA.fields if f.fmt(new[f.name]) != f.fmt(values[f.name])]
                    return new, edited, primary
        new = dict(values)
        if u < 0.8:  # another day of the same financial year
            for _ in range(100):
                d = values["date"] + dt.timedelta(days=int(rng.integers(-200, 200)))
                if d != values["date"] and GST_START <= d <= R.TODAY and \
                        R.financial_year(d) == R.financial_year(values["date"]):
                    new["date"] = d
                    return new, ["date"], "date"
        if u < 0.9 or True:
            if rng.random() < 0.5:
                new["inv_no"] = values["inv_no"][:-4] + f"{int(rng.integers(1, 10000)):04d}"
                if new["inv_no"] != values["inv_no"]:
                    return new, ["inv_no"], "inv_no"
            state = values["buyer_gstin"][:2] if values["buyer_gstin"][:2] in R.STATES else None
            new = dict(values, buyer_gstin=R.make_gstin(rng, state))
            return new, ["buyer_gstin"], "buyer_gstin"

    def random_values(self, rng, v):
        v = dict(v)
        for q, r, a in zip(QTY, RATE, AMT):
            if v[q] == BLANK:
                continue
            if rng.random() < 0.5:
                v[q] = int(rng.integers(0, 1000))
            if rng.random() < 0.5:
                v[r] = int(rng.integers(0, MAX_TOTAL))
            if rng.random() < 0.5:
                v[a] = int(rng.integers(0, MAX_TOTAL))
        for f in ("subtotal", "cgst", "sgst", "igst", "total"):
            if rng.random() < 0.5:
                v[f] = int(rng.integers(0, MAX_TOTAL))
        if rng.random() < 0.5:
            v["round_off"] = int(rng.integers(-99, 100))
        if rng.random() < 0.5:
            v["words"] = int(rng.integers(0, MAX_TOTAL // 100))
        if rng.random() < 0.5:
            v["date"] = dt.date(int(rng.integers(2010, 2031)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
        for f in ("seller_gstin", "buyer_gstin"):
            if rng.random() < 0.4:
                v[f] = "".join(rng.choice(list(ALNUM), 15))
        if rng.random() < 0.3:
            y = int(rng.integers(0, 100))
            v["inv_no"] = "".join(rng.choice(list(string.ascii_uppercase), int(rng.integers(2, 5)))) + \
                f"/{y:02d}-{int(rng.integers(0, 100)):02d}/{int(rng.integers(0, 10000)):04d}"
        if rng.random() < 0.3:
            names = list(R.STATES.values())
            v["pos"] = f"{int(rng.integers(0, 100)):02d}-{names[rng.integers(len(names))]}"
        return v
