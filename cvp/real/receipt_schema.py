"""A real receipt's own constraint graph.

Synthetic document types have one fixed schema. Real receipts print different fields: a
unit price on some items, a service charge, a card payment, or a printed item count. So
each receipt gets its own Schema, built from the values it prints, and the theory of
cvp.core.schema (Proposition 1 classes, signatures, candidate sets) applies to it
unchanged. For example, a receipt whose items print no unit price puts all item prices in
one class: only the subtotal reads them, through their sum, as with the transcript's marks.

The predicates and tolerances are those of receipt_rules, so validity measured there and
attribution measured here use the same rules. Add-on prices and item discounts are context
(constants): edits target the main printed values.
"""
from ..core.schema import Constraint, Field, Schema
from .receipt_rules import RULES, close

AMOUNT_FIELDS = [("subtotal", "Subtotal"), ("tax", "Tax"), ("service", "Service charge"), ("discount", "Discount"),
                 ("total", "Total"), ("cash", "Cash"), ("card", "Card payment"), ("change", "Change"), ("qty", "Item count")]


def _rel(a, b):
    return (a - b) / max(1.0, abs(b))


def _field(name, label):
    return Field(name, label, charset="0123456789.,-", fmt=str, parse=None)


def values(r):
    """The receipt's printed values, keyed by the schema's field names."""
    v = {}
    for k, it in enumerate(r.items, 1):
        v[f"price{k}"], v[f"cnt{k}"], v[f"unit{k}"] = it.price, it.cnt, it.unitprice
    for f, _ in AMOUNT_FIELDS:
        v[f] = getattr(r, f)
    return v


def schema(r):
    """The constraint graph of receipt `r`: a field for every value it prints, and every rule
    whose values it prints. Built from the genuine receipt (its structure), then evaluated on
    any reading of it."""
    printed = {k: x for k, x in values(r).items() if x is not None}
    fields, cons = [], []
    every = range(1, len(r.items) + 1)
    for k in every:
        for f, label in ((f"price{k}", "Price"), (f"cnt{k}", "Count"), (f"unit{k}", "Unit price")):
            if f in printed:
                fields.append(_field(f, f"{label}, item {k}"))
    fields += [_field(f, label) for f, label in AMOUNT_FIELDS if f in printed]
    has = {f.name for f in fields}

    for k in every:  # count x unit price = price (as receipt_rules: items with a non-zero count)
        c, u, p = f"cnt{k}", f"unit{k}", f"price{k}"
        if printed.get(c) and u in has and p in has:
            cons.append(Constraint(f"line{k}", "arithmetic", (c, u, p),
                                   lambda v, c=c, u=u, p=p: (not close(v[c] * v[u], v[p]), _rel(v[c] * v[u], v[p])),
                                   f"Count x unit price = price (item {k})", sensitive=(c, u, p),
                                   exchangeable=((c, u),), family="line"))
    prices = tuple(f"price{k}" for k in every)
    add = sum(sum(it.addons) for it in r.items)
    dsc = sum(abs(it.discount) for it in r.items if it.discount)
    if prices and all(p in has for p in prices) and "subtotal" in has:  # every item prints its price
        def sum_check(v):
            base = sum(v[p] for p in prices)
            xs = (base, base + add, base - dsc, base + add - dsc)
            return not any(close(x, v["subtotal"]) for x in xs), min((_rel(x, v["subtotal"]) for x in xs), key=abs)
        cons.append(Constraint("sum", "arithmetic", prices + ("subtotal",), sum_check, RULES["sum"],
                               sensitive=prices + ("subtotal",), exchangeable=(prices,) if len(prices) > 1 else ()))
    elif prices and all(p in has for p in prices) and "total" in has:  # no subtotal: items add up to the total
        extra = tuple(f for f in ("tax", "service") if f in has)
        scope = prices + extra + (("discount",) if "discount" in has else ()) + ("total",)

        def sum_total_check(v):
            base = sum(v[p] for p in prices)
            tax, svc, d = v.get("tax") or 0.0, v.get("service") or 0.0, abs(v.get("discount") or 0.0)
            xs = [x + tax + svc - d for x in (base, base + add, base - dsc, base + add - dsc)]
            xs += [x + svc - d for x in (base, base + add, base - dsc, base + add - dsc)]
            return not any(close(x, v["total"]) for x in xs), min((_rel(x, v["total"]) for x in xs), key=abs)
        terms = prices + extra
        cons.append(Constraint("sum", "arithmetic", scope, sum_total_check, RULES["sum"], sensitive=scope,
                               exchangeable=(terms,) if len(terms) > 1 else ()))
    if {"subtotal", "total"} <= has:
        terms = tuple(f for f in ("subtotal", "tax", "service") if f in has)
        scope = terms + (("discount",) if "discount" in has else ()) + ("total",)

        def total_check(v):
            tax, svc, d = v.get("tax") or 0.0, v.get("service") or 0.0, abs(v.get("discount") or 0.0)
            excl, incl = v["subtotal"] + tax + svc - d, v["subtotal"] + svc - d
            ok = close(excl, v["total"]) or close(incl, v["total"])
            return not ok, min((_rel(excl, v["total"]), _rel(incl, v["total"])), key=abs)
        cons.append(Constraint("total", "arithmetic", scope, total_check, RULES["total"], sensitive=scope,
                               exchangeable=(terms,) if len(terms) > 1 else ()))
    paid = tuple(f for f in ("cash", "card") if f in has)
    if "cash" in has and "total" in has and "change" in has:
        def change_check(v):
            p = v["cash"] + (v.get("card") or 0.0)
            ok = close(p - v["total"], v["change"]) or close(v["cash"] - v["total"], v["change"])
            return not ok, min((_rel(p - v["total"], v["change"]), _rel(v["cash"] - v["total"], v["change"])), key=abs)
        scope = paid + ("total", "change")
        cons.append(Constraint("change", "arithmetic", scope, change_check, RULES["change"], sensitive=scope,
                               exchangeable=(paid,) if len(paid) > 1 else ()))
    if "cash" in has and "total" in has:
        def tender_check(v):
            p = v["cash"] + (v.get("card") or 0.0)
            return p < v["total"] - 1.0, max(0.0, v["total"] - p) / max(1.0, v["total"])
        cons.append(Constraint("tender", "logical", paid + ("total",), tender_check, RULES["tender"],
                               exchangeable=(paid,) if len(paid) > 1 else ()))
    cnts = tuple(f"cnt{k}" for k in every)
    if "qty" in has and cnts and all(c in has for c in cnts):  # every item prints its count
        cons.append(Constraint("count", "arithmetic", cnts + ("qty",),
                               lambda v: (abs(sum(v[c] for c in cnts) - v["qty"]) > 1e-6,
                                          _rel(sum(v[c] for c in cnts), v["qty"])),
                               RULES["count"], sensitive=cnts + ("qty",), exchangeable=(cnts,) if len(cnts) > 1 else ()))
    return Schema(f"cord:{r.id}", fields, cons)
