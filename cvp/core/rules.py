"""Rule helpers shared by document types: check digits, money, amounts in words, dates.

Everything here is a published or conventional algorithm:
  ICAO 9303 check digits (machine-readable zones of passports and ID cards),
  the GSTIN check character (Luhn mod 36, as used by India's GST network),
  Indian digit grouping and amounts in words (lakh, crore), financial years.
Money is handled as integer paise so that every arithmetic rule is exact.
"""
import datetime as dt
import re

ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
TODAY = dt.date(2026, 11, 4)  # fixed reference date so experiments are reproducible


# ---------------------------------------------------------------- check digits
def icao_check(s):
    """ICAO 9303 check digit: weights 7,3,1 over digits, A=10..Z=35 and filler '<'=0."""
    total = 0
    for i, c in enumerate(s):
        v = 0 if c == "<" else ALNUM.index(c)
        total += v * (7, 3, 1)[i % 3]
    return str(total % 10)


def gstin_check(first14):
    """Check character of a GSTIN: Luhn mod 36 over its first 14 characters."""
    total = 0
    for i, c in enumerate(first14):
        p = ALNUM.index(c) * (1 if i % 2 == 0 else 2)
        total += p // 36 + p % 36
    return ALNUM[(36 - total % 36) % 36]


GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")

# GST state codes (a subset, with names short enough for one line).
STATES = {"03": "PUNJAB", "06": "HARYANA", "07": "DELHI", "08": "RAJASTHAN", "09": "UTTAR PRADESH",
          "10": "BIHAR", "19": "WEST BENGAL", "21": "ODISHA", "24": "GUJARAT", "27": "MAHARASHTRA",
          "29": "KARNATAKA", "30": "GOA", "32": "KERALA", "33": "TAMIL NADU", "36": "TELANGANA"}


def gstin_valid(g):
    return bool(GSTIN_RE.match(g)) and g[:2] in STATES and gstin_check(g[:14]) == g[14]


def make_gstin(rng, state=None, kind=None):
    """A syntactically valid GSTIN for a fictional business: state code, a PAN-format
    segment (five letters, four digits, a letter), entity number, 'Z', check character."""
    state = state or list(STATES)[rng.integers(len(STATES))]
    L = [chr(65 + int(x)) for x in rng.integers(0, 26, 6)]
    kind = kind or "CFP"[rng.integers(3)]
    pan = "".join(L[:3]) + kind + L[3] + "".join(str(int(x)) for x in rng.integers(0, 10, 4)) + L[4]
    first14 = state + pan + str(int(rng.integers(1, 4))) + "Z"
    return first14 + gstin_check(first14)


# ---------------------------------------------------------------- money (integer paise)
def inr(paise):
    """Indian digit grouping with two decimals: 123456789 paise -> '12,34,567.89'."""
    neg, p = paise < 0, abs(int(paise))
    rupees, frac = divmod(p, 100)
    s = str(rupees)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        s = ",".join(([head] if head else []) + groups + [tail])
    return ("-" if neg else "") + f"{s}.{frac:02d}"


MONEY_RE = re.compile(r"-?\d[\d,]*\.\d{2}")


def parse_inr(s):
    if not MONEY_RE.fullmatch(s) or ",," in s or s.lstrip("-").startswith(","):
        return None
    neg = s.startswith("-")
    rupees, frac = s.lstrip("-").replace(",", "").split(".")
    v = int(rupees) * 100 + int(frac)
    return -v if neg else v


def pct_of(paise, num, den=100):
    """Round-half-up share of an amount: GST components are computed this way."""
    return (paise * num * 2 + den) // (2 * den)


# ---------------------------------------------------------------- amounts in words
ONES = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN", "ELEVEN",
        "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN"]
TENS = ["", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]
_VOCAB = {**{w: i for i, w in enumerate(ONES) if w}, **{w: 10 * i for i, w in enumerate(TENS) if w}}
_SCALE = {"THOUSAND": 1000, "LAKH": 100_000, "CRORE": 10_000_000}


def _two(n):
    return ONES[n] if n < 20 else TENS[n // 10] + (" " + ONES[n % 10] if n % 10 else "")


def _three(n):
    h, r = divmod(n, 100)
    return " ".join(([ONES[h] + " HUNDRED"] if h else []) + ([_two(r)] if r else []))


def number_words(n):
    """Indian numbering in words: 123456 -> 'ONE LAKH TWENTY THREE THOUSAND FOUR HUNDRED FIFTY SIX'."""
    if n == 0:
        return "ZERO"
    crore, n = divmod(n, 10_000_000)
    lakh, n = divmod(n, 100_000)
    thousand, n = divmod(n, 1000)
    parts = ([_three(crore) + " CRORE"] if crore else []) + ([_two(lakh) + " LAKH"] if lakh else []) + \
            ([_two(thousand) + " THOUSAND"] if thousand else []) + ([_three(n)] if n else [])
    return " ".join(parts)


def rupees_words(rupees):
    return f"RUPEES {number_words(int(rupees))} ONLY"


def parse_rupees_words(s):
    """Inverse of rupees_words (lenient about grammar, strict about vocabulary)."""
    toks = s.split()
    if len(toks) < 3 or toks[0] != "RUPEES" or toks[-1] != "ONLY":
        return None
    total = cur = 0
    for t in toks[1:-1]:
        if t in _VOCAB:
            cur += _VOCAB[t]
        elif t == "HUNDRED":
            cur *= 100
        elif t in _SCALE:
            total += cur * _SCALE[t]
            cur = 0
        elif t != "ZERO":
            return None
    return total + cur


# ---------------------------------------------------------------- dates
def fmt_dmy(d):
    return d.strftime("%d/%m/%Y")


def parse_dmy(s):
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", s)
    if not m:
        return None
    try:
        return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def fmt_dmony(d):
    """'12 AUG 1974', the visual-zone date format of ICAO travel documents."""
    return f"{d.day:02d} {MONTHS[d.month - 1]} {d.year}"


def parse_dmony(s):
    m = re.fullmatch(r"(\d{2}) ([A-Z]{3}) (\d{4})", s)
    if not m or m.group(2) not in MONTHS:
        return None
    try:
        return dt.date(int(m.group(3)), MONTHS.index(m.group(2)) + 1, int(m.group(1)))
    except ValueError:
        return None


def add_years(d, n):
    try:
        return d.replace(year=d.year + n)
    except ValueError:  # 29 February
        return d.replace(year=d.year + n, day=28)


def age_on(dob, day):
    return day.year - dob.year - ((day.month, day.day) < (dob.month, dob.day))


def financial_year(d):
    """Indian financial year (April to March) as printed in invoice numbers: '25-26'."""
    y = d.year if d.month >= 4 else d.year - 1
    return f"{y % 100:02d}-{(y + 1) % 100:02d}"


def random_date(rng, lo, hi):
    return lo + dt.timedelta(days=int(rng.integers(0, max(1, (hi - lo).days + 1))))
