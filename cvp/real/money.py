"""Amounts and counts as printed on real receipts.

Receipts write the same number in many ways: "60.000" and "60,000" (thousands separators,
as in Indonesian rupiah), "48982.68" (a decimal point), "Rp. 111,000", "@11000",
"-2 600" (a space between groups), "(1,000)" (accounting negative). A separator followed by
exactly three digits in every group is a thousands separator; otherwise the last separator
is the decimal point.
"""
import re

_NUMBER = re.compile(r"[^0-9.,]")


def _number(s):
    s = _NUMBER.sub("", s).strip(".,")
    if not any(ch.isdigit() for ch in s):
        return None
    dot, comma = s.rfind("."), s.rfind(",")
    if dot >= 0 and comma >= 0:
        dec = "." if dot > comma else ","
    else:
        sep = "." if dot >= 0 else ("," if comma >= 0 else None)
        groups = s.split(sep) if sep else [s]
        thousands = sep is None or (0 < len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:]))
        dec = None if thousands else sep
    if dec is None:
        return float(re.sub(r"[.,]", "", s))
    whole, frac = s.rsplit(dec, 1)
    return float(f"{re.sub(r'[.,]', '', whole) or '0'}.{frac}")


def amount(s):
    """A money amount, or None when the text holds no amount (or is a percentage)."""
    if s is None or isinstance(s, (list, dict)):
        return None
    s = str(s).strip()
    if "%" in s:
        return None
    v = _number(s)
    if v is None:
        return None
    negative = s.startswith("-") or s.endswith("-") or (s.startswith("(") and s.endswith(")"))
    return -v if negative else v


def count(s):
    """An item count such as "2", "1X", "x2", "1.00xITEMs" or "(Qty=4.00)"; never negative."""
    if s is None or isinstance(s, (list, dict)):
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", str(s))
    return float(m.group().replace(",", ".")) if m else None
