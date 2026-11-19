"""Plain-English explanations of a CVP result: what each rule says, what the numbers were, and
why the verdict follows. Pure functions on plain data, so the page and the tests share them."""
import datetime as dt

# Rule texts written with symbols, in words (synthetic types); the others read well as they are.
OVERRIDE = {
    ("transcript", "c1"): "The five subject marks add up to the total",
    ("transcript", "c2"): "Percentage = total ÷ 500 × 100",
    ("transcript", "c3"): "The grade matches the percentage",
    ("certificate", "c_bnd"): "The grade matches the score",
    ("certificate", "c_dur"): "The course length matches its start and end dates",
}
REAL_RULES = {
    "sum": "The item prices add up to the subtotal",
    "total": "Subtotal + tax + service − discount = total",
    "change": "Cash paid − total = change",
    "tender": "The payment covers the total",
    "count": "The item counts add up to the printed number of items",
    "line": "Count × unit price = price",
}
CATEGORY = {"genuine": "genuine (not edited)", "T1": "an obvious edit (pasted text)",
            "T2": "a smooth, AI-style edit", "T3": "an obvious edit with the totals fixed too",
            "T4": "a smooth edit with the totals fixed too", "ai": "a real AI image editor"}


def pretty(text):
    for a, b in (("<=", "≤"), (">=", "≥"), (" x ", " × "), (" - ", " − ")):
        text = text.replace(a, b)
    return text


def rule_text(doctype, name, text):
    if doctype == "real":
        base = "line" if name.startswith("line") else name
        t = REAL_RULES.get(base, text)
        return t + (f" (item {name[4:]})" if base == "line" and name[4:] else "")
    return OVERRIDE.get((doctype, name), pretty(text[0].upper() + text[1:] if text else name))


def fmt(v):
    """A value as a person would write it."""
    if v is None:
        return "unreadable"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (dt.date, dt.datetime)):
        return v.strftime("%d %b %Y")
    if isinstance(v, (int, float)):
        f = float(v)
        if f.is_integer():
            return f"{f:,.0f}"
        return f"{f:,.2f}"
    return str(v)


def _sum_text(terms):
    return " + ".join(fmt(x) for x in terms) if len(terms) <= 8 else f"{fmt(terms[0])} + … + {fmt(terms[-1])}"


def arithmetic(doctype, name, values, scope, labels):
    """A sentence with the numbers for the common arithmetic rules, or None."""
    v = values
    get = v.get
    try:
        if doctype == "transcript" and name == "c1":
            ms = [get(f"m{i}") for i in range(1, 6)]
            if None in ms or get("total") is None:
                return None
            return f"{_sum_text(ms)} = {fmt(sum(ms))}; the total says {fmt(get('total'))}"
        if doctype == "transcript" and name == "c2":
            if get("total") is None or get("pct") is None:
                return None
            return f"{fmt(get('total'))} ÷ 500 × 100 = {100 * get('total') / 500:.2f}; the percentage says {fmt(get('pct'))}"
        if doctype in ("invoice", "receipt") and name == "c_sum":
            amts = [get(f) for f in scope if f.startswith("amt")]
            if None in amts or get("subtotal") is None:
                return None
            label = labels.get("subtotal", "subtotal").lower()
            return f"{_sum_text(amts)} = {fmt(sum(amts))}; the {label} says {fmt(get('subtotal'))}"
        if doctype in ("invoice", "receipt") and name.startswith("c_ln"):
            q, r_, a = (get(f) for f in scope)
            if None in (q, r_, a):
                return None
            return f"{fmt(q)} × {fmt(r_)} = {fmt(q * r_)}; the amount says {fmt(a)}"
        if doctype in ("invoice", "receipt") and name == "c_tot":
            parts = [f for f in scope if f != "total"]
            xs = [get(f) for f in parts]
            if None in xs or get("total") is None:
                return None
            return f"{_sum_text(xs)} = {fmt(sum(xs))}; the total says {fmt(get('total'))}"
        if doctype == "receipt" and name == "c_chg":
            if None in (get("cash"), get("total"), get("change")):
                return None
            return (f"{fmt(get('cash'))} − {fmt(get('total'))} = {fmt(get('cash') - get('total'))}; "
                    f"the change says {fmt(get('change'))}")
        if doctype == "real":
            return _real_arithmetic(name, v, scope)
    except (TypeError, ValueError):
        return None
    return None


def _real_arithmetic(name, v, scope):
    get = v.get
    if name == "sum":
        prices = [get(f) for f in scope if f.startswith("price")]
        target = "subtotal" if "subtotal" in scope else "total"
        if None in prices or get(target) is None:
            return None
        extra = ""
        if target == "total":
            add = sum(get(f) or 0.0 for f in ("tax", "service") if f in scope) - abs(get("discount") or 0.0)
            if add:
                extra = f" ({'+' if add >= 0 else '−'} {fmt(abs(add))} tax, service and discount)"
                return (f"{_sum_text(prices)}{extra} = {fmt(sum(prices) + add)}; the {target} says "
                        f"{fmt(get(target))}")
        return f"{_sum_text(prices)} = {fmt(sum(prices))}; the {target} says {fmt(get(target))}"
    if name == "total":
        sub, tax, svc, d = get("subtotal"), get("tax") or 0.0, get("service") or 0.0, abs(get("discount") or 0.0)
        if sub is None or get("total") is None:
            return None
        terms = [fmt(sub)] + [f"{fmt(x)}" for x in (tax, svc) if x]
        expr = " + ".join(terms) + (f" − {fmt(d)}" if d else "")
        return f"{expr} = {fmt(sub + tax + svc - d)}; the total says {fmt(get('total'))}"
    if name == "change":
        paid = (get("cash") or 0.0) + (get("card") or 0.0)
        if get("total") is None or get("change") is None:
            return None
        who = "cash" + (" + other payment" if get("card") else "")
        return (f"{fmt(paid)} ({who}) − {fmt(get('total'))} = {fmt(paid - get('total'))}; "
                f"the change says {fmt(get('change'))}")
    if name == "tender":
        paid = (get("cash") or 0.0) + (get("card") or 0.0)
        if get("total") is None:
            return None
        return f"paid {fmt(paid)} for a total of {fmt(get('total'))}"
    if name.startswith("line"):
        k = name[4:]
        c, u, p = get(f"cnt{k}"), get(f"unit{k}"), get(f"price{k}")
        if None in (c, u, p):
            return None
        return f"{fmt(c)} × {fmt(u)} = {fmt(c * u)}; the price says {fmt(p)}"
    if name == "count":
        cs = [get(f) for f in scope if f.startswith("cnt")]
        if None in cs or get("qty") is None:
            return None
        return f"{_sum_text(cs)} = {fmt(sum(cs))}; the item count says {fmt(get('qty'))}"
    return None


def values_text(scope, values, labels):
    return " · ".join(f"{labels.get(f, f)}: {fmt(values.get(f))}" for f in scope)


def join(names):
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def narrate(c):
    """The verdict and the reasoning in plain words.

    c: status ('tampered', 'genuine', 'warning': only an item-level rule broke, 'unknown': nothing
    could be checked), mode ('synthetic', 'cord', 'aiforge' or 'receipt'), labels {field: label},
    blamed field, suspects (fields the broken rules fit), broken (texts of the broken rules), heat
    {field: 0-1}, unread (labels), n_fields, n_rules, and optionally prior {field: p} (the learned
    head's field prior) and score/threshold (the overall visual score and its alarm level).
    Returns dict(headline, reason, steps); **bold** marks field names."""
    L = c["labels"]
    status, mode = c["status"], c["mode"]
    heat, prior = c.get("heat") or {}, c.get("prior") or {}
    suspects, broken, blamed = c.get("suspects") or [], c.get("broken") or [], c.get("blamed")

    def b(f):
        return f"**{L.get(f, f)}**"
    top = max(heat, key=heat.get) if heat else None
    hot = max(suspects, key=lambda f: heat.get(f, 0.0)) if suspects else None
    receipt = mode == "receipt"
    headline = {"tampered": "Tampered", "warning": "Check this one", "unknown": "Can't verify",
                "genuine": "No sign of tampering" if receipt else "Looks genuine"}[status]

    # why the blamed field
    pick = None
    if status == "tampered" and blamed:
        if len(suspects) == 1:
            pick = "it is the only field that fits the broken rules"
        elif not suspects:
            pick = f"no rule is broken, but the image looks most edited there (heat {heat.get(blamed, 0):.2f})"
        elif blamed == hot or heat.get(blamed, 0) >= heat.get(hot, 0) - 1e-6:
            pick = f"of the {len(suspects)} suspects it looks the most edited (heat {heat.get(blamed, 0):.2f})"
        else:
            pick = (f"the image is warm on more than one suspect ({b(hot)} {heat.get(hot, 0):.2f}, {b(blamed)} "
                    f"{heat.get(blamed, 0):.2f}), and what the rules taught the model to expect tips it to {b(blamed)}")
            if blamed in prior and hot in prior:
                pick += f" (prior {prior[blamed]:.2f} against {prior[hot]:.2f})"

    # the one-sentence reason
    if status == "tampered":
        if broken:
            rq = f"“{broken[0]}”" + (f" and {len(broken) - 1} other rule{'s' if len(broken) > 2 else ''}"
                                     if len(broken) > 1 else "")
            verb = "does not hold" if len(broken) == 1 else "do not hold"
            if len(suspects) == 1:
                reason = f"{rq} {verb}, and only {b(suspects[0])} can explain that."
            elif blamed and blamed == hot:
                reason = (f"{rq} {verb}. The rules alone can't tell which of {len(suspects)} fields was changed; "
                          f"the image looks most edited at {b(blamed)}.")
            elif blamed:
                reason = (f"{rq} {verb}. The rules alone can't tell which of {len(suspects)} fields was changed; "
                          f"CVP weighs the image and the rules and picks {b(blamed)}.")
            else:
                reason = f"{rq} {verb}."
        else:
            reason = (f"Every rule holds, but the image shows clear edit traces at {b(blamed)} "
                      f"(heat {heat.get(blamed, 0):.2f})." if blamed else "Every rule holds, but the image shows edit traces.")
    elif status == "warning":
        rq = f"“{broken[0]}”" if broken else "An item-level rule"
        reason = (f"{rq} does not hold as read. Item lines are easy to misread, so this is a warning, not an "
                  f"alarm: compare the readings with the photo.")
    elif status == "unknown":
        reason = ("No rule could be checked: the bill's subtotal, tax, total, cash and change were not found "
                  "together. Try a sharper, straighter photo.")
    elif receipt:
        reason = "Every rule that could be checked holds. In experimental mode only the arithmetic can raise an alarm."
    elif top is not None and heat[top] >= 0.5 and c.get("score") is not None:
        reason = (f"Every rule holds. The image is warm at {b(top)} (heat {heat[top]:.2f}), but the overall visual "
                  f"score ({c['score']:.2f}) stays below the alarm level ({c['threshold']:.2f}).")
    elif top is not None and heat[top] >= 0.5:
        reason = (f"Every rule holds. The image is warm at {b(top)} (heat {heat[top]:.2f}), but this image model is "
                  f"not trusted to raise an alarm on its own.")
    else:
        reason = "Every rule holds and the image shows no clear edit traces."

    # the steps: read, check, shortlist, look, decide
    steps = []
    n, unread = c.get("n_fields", 0), c.get("unread") or []
    t = f"Read {n - len(unread)} of {n} fields" + (" from the photo, end to end, with docTR." if receipt else
                                                    " with OCR." if mode == "synthetic" else
                                                    " with docTR, at each value's labelled position.")
    if unread:
        t += f" Could not read: {', '.join(unread)}."
    steps.append(dict(title="Read the document", text=t))
    nr = c.get("n_rules", 0)
    if nr == 0:
        t = "None of the rules could be checked on what was read."
    elif broken:
        t = f"Checked {nr} rule{'s' if nr > 1 else ''}: {len(broken)} broken (" + "; ".join(f"“{x}”" for x in broken) + ")."
    else:
        t = f"Checked {nr} rule{'s' if nr > 1 else ''}: all hold."
    steps.append(dict(title="Check the rules", text=t))
    if broken and len(suspects) == 1:
        t = f"Only one field fits the broken rules: {b(suspects[0])}."
    elif broken and suspects:
        t = (f"The broken rules fit {len(suspects)} fields: {join(b(f) for f in suspects)}. The rules can't tell "
             f"these apart (Proposition 1 in our paper), so the image decides.")
    elif receipt:
        t = "No rule is broken. In experimental mode the image alone never raises an alarm."
    else:
        t = "No rule is broken, so every field stays a suspect and the image alone decides."
    steps.append(dict(title="Shortlist the suspects", text=t))
    if suspects and len(suspects) > 1:
        ranked = sorted(suspects, key=lambda f: -heat.get(f, 0.0))
        t = "Heat on the suspects: " + ", ".join(f"{b(f)} {heat.get(f, 0.0):.2f}" for f in ranked[:4]) + \
            (", …" if len(ranked) > 4 else "") + " (0 = untouched, 1 = surely edited)."
    elif top is not None and heat[top] >= 0.05:
        t = f"The strongest edit traces are at {b(top)} (heat {heat[top]:.2f}, where 0 = untouched and 1 = surely edited)."
    else:
        t = "The image shows no clear edit traces anywhere" + (f" (highest heat {heat[top]:.2f})." if top else ".")
    if receipt and status != "unknown":
        t += " Here the heatmap only ranks suspects."
    steps.append(dict(title="Look at the pixels", text=t))
    if status == "tampered":
        t = f"Tampered. Blame {b(blamed)}: {pick}." if blamed else "Tampered."
    elif status == "warning":
        t = "A warning only: an item-level rule broke, which a misreading can also cause."
    elif status == "unknown":
        t = "Can't verify: nothing could be checked."
    elif receipt:
        t = "No sign of tampering: every rule that could be checked holds."
    elif top is not None and heat[top] >= 0.5:
        t = "Genuine: nothing is broken, and the image alone is not enough to raise the alarm."
    else:
        t = "Genuine: nothing is broken and nothing looks edited."
    steps.append(dict(title="Decide", text=t))
    return dict(headline=headline, reason=reason, steps=steps)


def answer(truth, blamed_field, status, labels, suspects=()):
    """How the verdict compares with the answer key (known for our forgeries and AIForge-Doc):
    ('right' | 'partial' | 'wrong', a sentence)."""
    edited = truth.get("edited") or []
    tampered = status == "tampered"

    def b(f):
        return f"**{labels.get(f, f)}**"
    if truth.get("forged") and not edited:
        what = truth.get("what") or "a field"
        return (("partial", f"Caught, although the edit changed {what}, which no rule reads.") if tampered else
                ("wrong", f"Missed: the edit changed {what}, which no rule reads."))
    if not edited:
        if status == "warning":
            return "partial", "This document was not edited: the warning comes from a misreading."
        return ("right", "This document was not edited, and CVP said so.") if not tampered else \
            ("wrong", "False alarm: this document was not edited.")
    names = join(b(f) for f in edited)
    was = "was" if len(edited) == 1 else "were"
    if status == "warning":
        among = any(f in suspects for f in edited)
        return "partial", (f"Flagged as a warning: {names} {was} changed" +
                           (", and it is among the suspects." if among else "."))
    if not tampered:
        return "wrong", f"Missed: {names} {was} changed."
    if blamed_field in edited:
        return "right", f"Right: {b(blamed_field)} is {'the' if len(edited) == 1 else 'one'} field that was changed."
    if blamed_field is None:
        return "partial", f"Caught, but no field was named: {names} {was} changed."
    return "partial", f"Caught, but it blamed {b(blamed_field)}: {names} {was} changed."
