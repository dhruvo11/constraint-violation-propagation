"""The arithmetic a genuine receipt satisfies, checked on a real receipt's values.

Each rule applies only when every value it reads is printed: receipts show different
subsets of fields. Shops follow different conventions, so a rule holds when one of its
standard forms does. Prices may already include the tax, add-ons (toppings, sizes) may be
priced separately from their item, discounts are printed with either sign, and part of the
bill may be paid by card. Money is compared within 0.5% (rounding to the nearest hundred
rupiah, say) and at least one currency unit.
"""

TOL = 0.005
RULES = {
    "line": "count x unit price = price, for each item that prints a unit price",
    "sum": "item prices (with add-ons, less item discounts) add up to the subtotal, or, when no subtotal "
           "is printed, with tax and service to the total",
    "total": "subtotal + tax + service - discount = total (or the prices already include the tax)",
    "change": "cash (+ card) - total = change",
    "tender": "cash (+ card) covers the total",
    "count": "item counts add up to the printed number of items",
}


def close(a, b, tol=TOL):
    return abs(a - b) <= max(1.0, tol * abs(b))


def _known(*xs):
    return all(x is not None for x in xs)


def check(r):
    """{rule: None when it does not apply, else whether it holds}. 'line' gives one result
    per item that prints a unit price (a list), or None when no item does."""
    out = {}
    lines = [close(it.cnt * it.unitprice, it.price) for it in r.items if it.cnt and _known(it.unitprice, it.price)]
    out["line"] = lines or None

    prices = [it.price for it in r.items]
    sums = None
    if prices and _known(*prices):
        base = sum(prices)
        add = sum(sum(it.addons) for it in r.items)
        dsc = sum(abs(it.discount) for it in r.items if it.discount)
        sums = (base, base + add, base - dsc, base + add - dsc)
    if sums and r.subtotal is not None:
        out["sum"] = any(close(v, r.subtotal) for v in sums)
    elif sums and r.total is not None:  # no subtotal printed: the items add up to the total
        tax, svc, d = r.tax or 0.0, r.service or 0.0, abs(r.discount or 0.0)
        out["sum"] = any(close(v + tax + svc - d, r.total) or close(v + svc - d, r.total) for v in sums)
    else:
        out["sum"] = None

    if _known(r.total, r.subtotal):
        tax, svc, dsc = r.tax or 0.0, r.service or 0.0, abs(r.discount or 0.0)
        out["total"] = close(r.subtotal + tax + svc - dsc, r.total) or close(r.subtotal + svc - dsc, r.total)
    else:
        out["total"] = None

    paid = None if r.cash is None else r.cash + (r.card or 0.0)
    out["change"] = (close(paid - r.total, r.change) or close(r.cash - r.total, r.change)) \
        if _known(paid, r.total, r.change) else None
    out["tender"] = (paid >= r.total - 1.0) if _known(paid, r.total) else None

    cnts = [it.cnt for it in r.items]
    out["count"] = (abs(sum(cnts) - r.qty) < 1e-6) if r.qty is not None and cnts and _known(*cnts) else None
    return out
