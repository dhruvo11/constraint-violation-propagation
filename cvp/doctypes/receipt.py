"""Shop receipt (thermal paper): up to five items (qty x rate = amount), item count and
total quantity, subtotal, CGST and SGST at 2.5 % each, total, cash tendered and change;
a GSTIN with its check character and a bill number that carries the bill's year and month.

Constraint structure: the total-quantity line separates each item's quantity from its
rate, so every field is its own class (Proposition 1 gives no ambiguity here), yet cash
and change both break only 'change = cash - total', and a bill number and a date both
break only the year-month rule: those pairs share a signature.
"""
import datetime as dt
import re
import string

import numpy as np

from ..core import rules as R
from ..core.render import FONT_MONO, font
from ..core.schema import Constraint, Field, Schema
from . import _draw as D
from .base import BLANK, DocType, Layout
from .invoice import flag, line_check, sum_check, typo

FONT_MONO_BOLD = FONT_MONO.replace("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf")
N = 5
QTY = [f"qty{k}" for k in range(1, N + 1)]
RATE = [f"rate{k}" for k in range(1, N + 1)]
AMT = [f"amt{k}" for k in range(1, N + 1)]
OPENED = dt.date(2019, 1, 1)  # the store's first day of trading
ALNUM = string.digits + string.ascii_uppercase
MONEY = string.digits + "."
BILL_RE = re.compile(r"^\d{4}-\d{4}$")


def money(p):
    return f"{p // 100}.{p % 100:02d}"


def parse_money(s):
    m = re.fullmatch(r"(\d+)\.(\d{2})", s)
    return int(m.group(1)) * 100 + int(m.group(2)) if m else None


def _int(s):
    return int(s) if s.isdigit() else None


def _time(s):
    m = re.fullmatch(r"(\d{2}):(\d{2})", s)
    return (int(m.group(1)), int(m.group(2))) if m else None


FIELDS = [
    Field("gstin", "GSTIN", ALNUM, str, lambda s: s or None),
    Field("bill_no", "Bill number", string.digits + "-", str, lambda s: s or None),
    Field("date", "Date", string.digits + "/", R.fmt_dmy, R.parse_dmy),
    Field("time", "Time", string.digits + ":", lambda t: f"{t[0]:02d}:{t[1]:02d}", _time),
    *[Field(q, f"Quantity, item {k}", string.digits, D.blank_or(str), D.parse_or_blank(_int))
      for k, q in enumerate(QTY, 1)],
    *[Field(r, f"Rate, item {k}", MONEY, D.blank_or(money), D.parse_or_blank(parse_money))
      for k, r in enumerate(RATE, 1)],
    *[Field(a, f"Amount, item {k}", MONEY, D.blank_or(money), D.parse_or_blank(parse_money))
      for k, a in enumerate(AMT, 1)],
    Field("n_items", "Item count", string.digits, str, _int),
    Field("tot_qty", "Total quantity", string.digits, str, _int),
    Field("subtotal", "Subtotal", MONEY, money, parse_money),
    Field("cgst", "CGST", MONEY, money, parse_money),
    Field("sgst", "SGST", MONEY, money, parse_money),
    Field("total", "Total", MONEY, money, parse_money),
    Field("cash", "Cash tendered", MONEY, money, parse_money),
    Field("change", "Change", MONEY, money, parse_money),
]


def _bill_format(v):
    return flag(bool(BILL_RE.match(v["bill_no"])))


def _bill_month(v):
    return flag(v["bill_no"][:4] == v["date"].strftime("%y%m"))


def _date(v):
    d = v["date"]
    days = max(0, (d - R.TODAY).days) + max(0, (OPENED - d).days)
    return days > 0, days / 365.0


def _time_ok(v):
    h, m = v["time"]
    minutes = h * 60 + m
    ok = m < 60 and 7 * 60 <= minutes <= 23 * 60
    return not ok, 0.0 if ok else (abs(minutes - 15 * 60) / 600.0)


def _count(v):
    d = sum(v[a] != BLANK for a in AMT) - v["n_items"]
    return d != 0, d / 5.0


def _qty(v):
    d = sum(v[q] for q in QTY if v[q] != BLANK) - v["tot_qty"]
    return d != 0, d / 10.0


def _gst(f):
    def check(v):
        d = v[f] - R.pct_of(v["subtotal"], 25, 1000)
        return d != 0, d / 1000.0
    return check


def _total(v):
    d = v["subtotal"] + v["cgst"] + v["sgst"] - v["total"]
    return d != 0, d / 10_000.0


def _change(v):
    d = v["cash"] - v["total"] - v["change"]
    return d != 0, d / 10_000.0


def _tender(v):
    d = max(0, v["total"] - v["cash"])
    return d > 0, d / 10_000.0


CONSTRAINTS = [
    Constraint("c_gst", "checksum", ("gstin",), lambda v: flag(R.gstin_valid(v["gstin"])), "GSTIN check character valid"),
    Constraint("c_bno", "format", ("bill_no",), _bill_format, "Bill no. is YYMM-NNNN", on_missing=(1.0, 1.0)),
    Constraint("c_bym", "logical", ("bill_no", "date"), _bill_month, "Bill no. carries the date's year and month"),
    Constraint("c_dt", "temporal", ("date",), _date, "Opening day <= date <= today"),
    Constraint("c_tm", "temporal", ("time",), _time_ok, "07:00 <= time <= 23:00"),
    *[Constraint(f"c_ln{k}", "arithmetic", (q, r, a), line_check(q, r, a, 10_000.0), f"Qty x Rate = Amount (item {k})",
                 sensitive=(q, r, a), exchangeable=((q, r),), family="c_ln")
      for k, (q, r, a) in enumerate(zip(QTY, RATE, AMT), 1)],
    Constraint("c_cnt", "arithmetic", tuple(AMT) + ("n_items",), _count, "Item count = number of items",
               sensitive=("n_items",)),
    Constraint("c_qty", "arithmetic", tuple(QTY) + ("tot_qty",), _qty, "Total quantity = sum of quantities",
               sensitive=tuple(QTY) + ("tot_qty",), exchangeable=(tuple(QTY),)),
    Constraint("c_sum", "arithmetic", tuple(AMT) + ("subtotal",), sum_check(AMT, "subtotal", 10_000.0),
               "Sum of amounts = subtotal", sensitive=tuple(AMT) + ("subtotal",), exchangeable=(tuple(AMT),)),
    Constraint("c_cg", "arithmetic", ("subtotal", "cgst"), _gst("cgst"), "CGST = 2.5% of subtotal", sensitive=("cgst",)),
    Constraint("c_sg", "arithmetic", ("subtotal", "sgst"), _gst("sgst"), "SGST = 2.5% of subtotal", sensitive=("sgst",)),
    Constraint("c_tot", "arithmetic", ("subtotal", "cgst", "sgst", "total"), _total, "Subtotal + taxes = total",
               sensitive=("subtotal", "cgst", "sgst", "total"), exchangeable=(("subtotal", "cgst", "sgst"),)),
    Constraint("c_chg", "arithmetic", ("total", "cash", "change"), _change, "Change = cash - total",
               sensitive=("total", "cash", "change")),
    Constraint("c_tnd", "logical", ("total", "cash"), _tender, "Cash tendered >= total"),
]

SCHEMA = Schema("receipt", FIELDS, CONSTRAINTS)


# ---------------------------------------------------------------- template
W, H = 432, 736
HEADER = (24, 124)
ROW_Y = [234 + 38 * k for k in range(N)]
M13, MB15 = (FONT_MONO, 13), (FONT_MONO_BOLD, 15)
LAYOUT = Layout({
    "gstin": (84, 99, "l", *M13, 130),
    "bill_no": (100, 155, "l", *M13, 90),
    "date": (288, 155, "l", *M13, 100),
    "time": (288, 177, "l", *M13, 60),
    **{q: (238, y, "r", *M13, 44) for q, y in zip(QTY, ROW_Y)},
    **{r: (318, y, "r", *M13, 76) for r, y in zip(RATE, ROW_Y)},
    **{a: (408, y, "r", *M13, 84) for a, y in zip(AMT, ROW_Y)},
    "n_items": (88, 431, "l", *M13, 44),
    "tot_qty": (288, 431, "l", *M13, 50),
    "subtotal": (408, 457, "r", *M13, 100),
    "cgst": (408, 481, "r", *M13, 100),
    "sgst": (408, 505, "r", *M13, 100),
    "total": (408, 537, "r", *MB15, 110),
    "cash": (408, 567, "r", *M13, 100),
    "change": (408, 591, "r", *M13, 100),
})

STORES = [("FRESHMART SUPERMARKET", "No. 14, 4th Main, Malleshwaram", "Bengaluru 560003", "29"),
          ("DAILY NEEDS STORE", "Shop 3, Station Road, Andheri East", "Mumbai 400069", "27"),
          ("GREEN BASKET MART", "22, Anna Salai, Teynampet", "Chennai 600018", "33"),
          ("CORNER KIRANA", "8-2-120, Road No. 2, Banjara Hills", "Hyderabad 500034", "36"),
          ("SAVEMORE BAZAAR", "A-41, Lajpat Nagar II", "New Delhi 110024", "07"),
          ("HAPPY HOME PROVISIONS", "MG Road, Ernakulam", "Kochi 682016", "32")]
CASHIERS = ["RAVI", "ASHA", "IMRAN", "DEEPA", "SURESH", "KAVYA", "JOHN", "MEENA"]
GOODS = [("TOOR DAL 1KG", 135, 175), ("BASMATI RICE 5KG", 520, 780), ("SUNFLOWER OIL 1L", 130, 190),
         ("SUGAR 1KG", 42, 52), ("TEA POWDER 250G", 110, 160), ("MILK 1L", 54, 68), ("BREAD BROWN", 40, 55),
         ("EGGS (12)", 72, 96), ("BUTTER 500G", 250, 290), ("PANEER 200G", 80, 110), ("CURD 400G", 35, 55),
         ("BISCUITS FAMILY PK", 30, 60), ("TOOTHPASTE 150G", 95, 140), ("DETERGENT 1KG", 110, 220),
         ("ONION 1KG", 30, 70), ("TOMATO 1KG", 25, 80), ("APPLES 1KG", 140, 240), ("BANANAS (6)", 40, 70),
         ("SOAP (4 PACK)", 120, 180), ("ATTA 5KG", 230, 330)]
WEIGHTS = {"gstin": 0.5, "bill_no": 0.5, "date": 1.5, "time": 0.5, **{q: 1.5 for q in QTY},
           **{r: 1.5 for r in RATE}, **{a: 2.0 for a in AMT}, "n_items": 0.5, "tot_qty": 0.5,
           "subtotal": 1.5, "cgst": 1.0, "sgst": 1.0, "total": 3.0, "cash": 1.0, "change": 1.0}


def settle(v):
    v = dict(v)
    for q, r, a in zip(QTY, RATE, AMT):
        if v[q] != BLANK:
            v[a] = v[q] * v[r]
    v["n_items"] = sum(v[a] != BLANK for a in AMT)
    v["tot_qty"] = sum(v[q] for q in QTY if v[q] != BLANK)
    v["subtotal"] = sum(v[a] for a in AMT if v[a] != BLANK)
    v["cgst"] = v["sgst"] = R.pct_of(v["subtotal"], 25, 1000)
    v["total"] = v["subtotal"] + v["cgst"] + v["sgst"]
    return v


def tender(total, rng):
    """Cash handed over: the exact amount, or the next convenient note."""
    if rng.random() < 0.25:
        return total
    step = int(rng.choice([1000, 5000, 10_000, 50_000, 200_000]))
    return (total // step + 1) * step


class Receipt(DocType):
    name = "receipt"
    title = "Shop receipt"
    schema = SCHEMA
    page = (W, H)
    slots = LAYOUT.slots()
    wide = LAYOUT.wide()
    header_rule = HEADER
    weights = WEIGHTS
    date_fields = ("date",)
    class_labels = {f.name: f.label for f in FIELDS}
    unary_labels = ("", "")

    def from_json(self, values):
        out = super().from_json(values)
        if isinstance(out.get("time"), list):
            out["time"] = tuple(out["time"])
        return out

    def sample_spec(self, rng):
        store = STORES[rng.integers(len(STORES))]
        date = R.random_date(rng, OPENED, R.TODAY)
        v = {"gstin": R.make_gstin(rng, store[3]), "date": date,
             "time": (int(rng.integers(8, 23)), int(rng.integers(0, 60))),
             "bill_no": f"{date:%y%m}-{int(rng.integers(1, 10000)):04d}"}
        n = int(rng.integers(2, N + 1))
        picks = rng.choice(len(GOODS), n, replace=False)
        for k in range(N):
            if k < n:
                _, lo, hi = GOODS[picks[k]]
                v[RATE[k]] = int(rng.integers(lo, hi + 1)) * 100 + (50 if rng.random() < 0.2 else 0)
                v[QTY[k]] = int(rng.choice([1, 1, 1, 2, 2, 3, 4, 6]))
            else:
                v[QTY[k]] = v[RATE[k]] = v[AMT[k]] = BLANK
        v = settle(v)
        v["cash"] = tender(v["total"], rng)
        v["change"] = v["cash"] - v["total"]
        content = dict(store=store[0], address=[store[1], store[2]],
                       phone=f"Ph: {int(rng.integers(20, 99))}-{int(rng.integers(2000, 9999))} {int(rng.integers(1000, 9999))}",
                       cashier=CASHIERS[rng.integers(len(CASHIERS))], items=[GOODS[picks[k]][0] for k in range(n)])
        base = int(rng.integers(240, 250))
        style = D.sample_style(rng, SCHEMA.field_names, tint=(base, base + 2, base + 4), ink=(30, 80))
        return v, content, style

    def anchor(self, field, style):
        return LAYOUT.origin(field, "", LAYOUT.font(field), style)

    def place(self, field, text, fnt, style):
        return LAYOUT.origin(field, text, fnt, style)

    def font_scale(self, field):
        return LAYOUT.scale(field)

    def render_clean(self, values, content, style):
        img = D.paper((W, H), style)
        p = D.Pen(img, style)
        m11, m12, b12 = font(FONT_MONO, 11), font(FONT_MONO, 12), font(FONT_MONO_BOLD, 12)
        p.text(0, 22, content["store"], font(FONT_MONO_BOLD, 18), align="page")
        p.text(0, 48, content["address"][0], m11, align="page")
        p.text(0, 64, content["address"][1], m11, align="page")
        p.text(0, 80, content["phone"], m11, align="page")
        p.text(24, 100, "GSTIN:", m12)
        p.rule(HEADER[0], HEADER[1], 408, t=2)
        p.text(0, 132, "TAX INVOICE", font(FONT_MONO_BOLD, 13), align="page")
        p.text(24, 156, "Bill No:", m12)
        p.text(232, 156, "Date:", m12)
        p.text(24, 178, "Cashier:", m12)
        p.text(100, 178, content["cashier"], m12)
        p.text(232, 178, "Time:", m12)
        p.rule(24, 202, 408)
        for x, s, a in [(24, "ITEM", "l"), (238, "QTY", "r"), (318, "RATE", "r"), (408, "AMOUNT", "r")]:
            p.text(x, 208, s, b12, align=a)
        p.rule(24, 226, 408)
        for k, item in enumerate(content["items"]):
            p.text(24, ROW_Y[k] + 1, item, m12)
        p.rule(24, 422, 408)
        p.text(24, 432, "Items:", m12)
        p.text(232, 432, "Qty:", m12)
        for y, s in [(458, "Sub Total"), (482, "CGST @ 2.5%"), (506, "SGST @ 2.5%")]:
            p.text(24, y, s, m12)
        p.rule(24, 528, 408)
        p.text(24, 538, "TOTAL", font(FONT_MONO_BOLD, 15))
        p.text(24, 568, "Cash", m12)
        p.text(24, 592, "Change", m12)
        p.rule(24, 618, 408)
        p.text(0, 630, "Thank you! Visit again", m11, align="page")
        p.text(0, 646, "Goods once sold will not be taken back", font(FONT_MONO, 10), align="page")
        bars = np.random.default_rng(style.noise_seed % 9973)  # decorative barcode, fixed per receipt
        x = 96
        while x < 336:
            wbar = int(bars.integers(1, 4))
            p.rect(x, 672, x + wbar - 1, 712, fill=tuple(style.ink))
            x += wbar + int(bars.integers(1, 4))
        return img, D.draw_fields(img, LAYOUT, SCHEMA, values, style)

    # ------------------------------------------------ forgery value models
    def violating_value(self, f, v, rng):
        x = v[f]
        if f == "gstin":
            if rng.random() < 0.7:
                return typo(x, rng, range(2, 14))
            return x[:14] + rng.choice([c for c in ALNUM if c != x[14]])
        if f == "bill_no":
            if rng.random() < 0.8 and BILL_RE.match(x):  # another month's bill number
                y, m = int(x[:2]), int(x[2:4])
                m2 = (m - 1 + int(rng.choice([-3, -2, -1, 1, 2]))) % 12 + 1
                return f"{y:02d}{m2:02d}" + x[4:] if m2 != m else f"{(y + 1) % 100:02d}{m:02d}" + x[4:]
            return x + str(int(rng.integers(10))) if rng.random() < 0.5 else x[:-1]
        if f == "date":
            if rng.random() < 0.2:
                return R.TODAY + dt.timedelta(days=int(rng.integers(10, 300)))
            for _ in range(100):
                d = x + dt.timedelta(days=int(rng.integers(-200, 200)))
                if d.strftime("%y%m") != x.strftime("%y%m") or d > R.TODAY or d < OPENED:
                    return d
            return R.TODAY + dt.timedelta(days=10)
        if f == "time":
            h = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 23]))
            return (h, int(rng.integers(1 if h == 23 else 0, 60)))
        if f in QTY:
            return int(rng.choice([n for n in range(1, 20) if n != x]))
        if f in ("n_items", "tot_qty"):
            return int(rng.choice([n for n in range(1, 30) if n != x]))
        for _ in range(100):  # money
            step = int(rng.integers(1, 200)) * (100 if f in RATE + AMT + ["subtotal", "total", "cash"] else 10)
            new = x + step * (1 if rng.random() < 0.8 else -1)
            if new > 0 and new != x:
                return new
        return x + 100

    def coherent_edit(self, values, rng):
        u = rng.random()
        rows = [k for k in range(N) if values[QTY[k]] != BLANK]
        if u < 0.65 and rows:  # change one item and carry it through the bill, cash and change
            k = rows[rng.integers(len(rows))]
            new = dict(values)
            if rng.random() < 0.5:
                new[QTY[k]] = values[QTY[k]] + int(rng.integers(1, 4))
                primary = QTY[k]
            else:
                new[RATE[k]] = values[RATE[k]] + int(rng.integers(1, 60)) * 100
                primary = RATE[k]
            new = settle(new)
            new["cash"] = max(values["cash"], tender(new["total"], rng))
            new["change"] = new["cash"] - new["total"]
            edited = [f.name for f in SCHEMA.fields if f.fmt(new[f.name]) != f.fmt(values[f.name])]
            return new, edited, primary
        new = dict(values)
        if u < 0.85:  # another day of the same month
            for _ in range(100):
                d = values["date"] + dt.timedelta(days=int(rng.integers(-27, 28)))
                if d != values["date"] and d.strftime("%y%m") == values["date"].strftime("%y%m") and \
                        OPENED <= d <= R.TODAY:
                    new["date"] = d
                    return new, ["date"], "date"
        if rng.random() < 0.5:
            new["time"] = (int(rng.integers(8, 23)), int(rng.integers(0, 60)))
            if new["time"] != tuple(values["time"]):
                return new, ["time"], "time"
        new["bill_no"] = values["bill_no"][:5] + f"{int(rng.integers(1, 10000)):04d}"
        return new, ["bill_no"], "bill_no"

    def random_values(self, rng, v):
        v = dict(v)
        for q, r, a in zip(QTY, RATE, AMT):
            if v[q] == BLANK:
                continue
            if rng.random() < 0.5:
                v[q] = int(rng.integers(0, 100))
            if rng.random() < 0.5:
                v[r] = int(rng.integers(0, 100_000))
            if rng.random() < 0.5:
                v[a] = int(rng.integers(0, 1_000_000))
        for f in ("subtotal", "cgst", "sgst", "total", "cash", "change"):
            if rng.random() < 0.5:
                v[f] = int(rng.integers(0, 1_000_000))
        for f in ("n_items", "tot_qty"):
            if rng.random() < 0.5:
                v[f] = int(rng.integers(0, 100))
        if rng.random() < 0.5:
            v["date"] = dt.date(int(rng.integers(2010, 2031)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
        if rng.random() < 0.5:
            v["time"] = (int(rng.integers(0, 24)), int(rng.integers(0, 60)))
        if rng.random() < 0.4:
            v["gstin"] = "".join(rng.choice(list(ALNUM), 15))
        if rng.random() < 0.4:
            v["bill_no"] = f"{int(rng.integers(0, 10000)):04d}-{int(rng.integers(0, 10000)):04d}"
        return v
