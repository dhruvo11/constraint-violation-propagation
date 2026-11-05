"""The app's real-receipt view: forge a CORD v2 test receipt (or leave it genuine), run CVP on
the photo, and show the evidence. Forgeries use the evaluation's seeds
(scripts/real_extract_cord.py), so the app shows the very documents that were measured.
"""
import numpy as np
from PIL import Image, ImageDraw

from . import cord, forge, view
from . import receipt_schema as RS
from .receipt_rules import RULES

CAT_LABELS = {"genuine": "Genuine (no edit)", "T1": "T1 - obvious edit (pasted text)", "T2": "T2 - smooth AI-style edit",
              "T3": "T3 - obvious edit, totals fixed too", "T4": "T4 - smooth edit, totals fixed too"}
RANDOM = "random"
SPLIT = "test"
_SYSTEM = []


def unavailable():
    """Why the real-receipt view cannot run here, or None."""
    if not cord.files(SPLIT):
        return "CORD v2 has not been downloaded (data/real/cord-v2); see README."
    try:
        import doctr  # noqa: F401
    except ImportError:
        return "docTR is not installed (see requirements-real.txt)."
    return None


def system():
    if not _SYSTEM:
        from .system import RealCVP
        _SYSTEM.append(RealCVP())
    return _SYSTEM[0]


def count():
    return cord.count_receipts(SPLIT)


def labels(r):
    return {f.name: f.label for f in RS.schema(cord.from_lines(r.lines)).fields}


def index(k):
    """Receipt number k, kept inside the test set."""
    return max(0, min(count() - 1, int(k or 0)))


def field_choices(k):
    r = cord.receipt(SPLIT, index(k))
    lab = labels(r)
    return [RANDOM] + [lab[f] for f in forge.editable(r)]


def overlay(v, res, truth_rects, s):
    a = (np.clip(res["heatmap"], 0, 1) ** 1.5 * 0.65)[..., None]
    red = np.zeros_like(v, np.float32)
    red[..., 0] = 255
    im = Image.fromarray((v * (1 - a) + red * a).astype(np.uint8))
    d = ImageDraw.Draw(im)

    def box(f, colour, width, pad):
        if f in res["boxes"]:
            x0, y0, x1, y1 = view.scale_box(res["boxes"][f], s)
            d.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=colour, width=width)
    if res["violated"]:
        for f in res["consistent"]:
            if f not in res["candidates"]:
                box(f, (230, 160, 0), 1, 4)  # shares the violations: thin orange
        for f in res["candidates"]:
            box(f, (230, 160, 0), 2, 5)  # the signature class: orange
    for rect in truth_rects:
        x0, y0, x1, y1 = view.scale_box(rect, s)
        d.rectangle([x0, y0, x1, y1], outline=(0, 160, 60), width=1)  # ground truth: green
    if res["tampered"] and res["field"]:
        box(res["field"], (220, 0, 0), 3, 8)  # attributed: red
    return im


def money(x):
    return "--" if x is None else (f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}")


def report(res, info, lab, threshold):
    S = res["schema"]
    out = [f"### Verdict: **{'TAMPERED' if res['tampered'] else 'GENUINE'}**  "
           f"(rules broken: {len(res['violated'])}; visual max heat {res['p_visual']:.3f}, threshold {threshold:.3f})"]
    if res["tampered"] and res["field"]:
        out.append(f"**Attributed field:** {lab[res['field']]} (red box)")
    if res["violated"]:
        out.append("**Constraint branch:** broken " + "; ".join(f"`{c}` ({S.constraints[S.c_index[c]].text})"
                                                              for c in res["violated"]) +
                   f". These fit {len(res['consistent'])} field(s): " + ", ".join(lab[f] for f in res["consistent"]) +
                   " (orange boxes); the heatmap picks among them.")
    else:
        out.append("**Constraint branch:** every rule holds as read, so the decision rests on the visual branch "
                   "(all fields are candidates).")
    if info["edited"]:
        ok = res["tampered"] and res["field"] in info["edited"]
        out.append(f"**Ground truth:** {CAT_LABELS[info['category']]}: " +
                   ", ".join(f"{lab[f]} {money(info['old'][f])} -> {money(info['new'][f])}" for f in info["edited"]) +
                   f" (green boxes): {'CORRECT' if ok else 'MISSED'}")
    else:
        out.append(f"**Ground truth:** genuine: {'CORRECT' if not res['tampered'] else 'FALSE ALARM'}")
    rows = ["\n| Rule | Checks | Status |", "|---|---|---|"]
    for c in S.constraints:
        rows.append(f"| {c.name} | {c.text} | {'VIOLATED' if c.name in res['violated'] else 'ok'} |")
    rows += ["\n| Field | Printed (true) | Read as | Max heat |", "|---|---|---|---|"]
    truth = dict(res["truth"], **info["new"])
    for f in S.field_names:
        rows.append(f"| {lab[f]} | {money(truth.get(f))} | {money(res['values'].get(f))} | {res['heat'].get(f, 0):.3f} |")
    return "\n\n".join(out) + "\n" + "\n".join(rows)


def analyse(k, cat_label, field_label):
    """(photo as received, CVP evidence, report) for test receipt k forged as asked."""
    k = index(k)
    r = cord.receipt(SPLIT, k)
    cat = next((c for c, lbl in CAT_LABELS.items() if lbl == cat_label), "genuine")
    lab = labels(r)
    field = {v: f for f, v in lab.items()}.get(field_label) if cat in ("T1", "T2") else None
    img, info = forge.forge(r, cat, np.random.default_rng(1000 * k + forge.CATEGORIES.index(cat)), field=field)
    sysm = system()
    res = sysm(r, img)
    v, s = view.visual_view(img)
    return Image.fromarray(v), overlay(v, res, info["rects"], s), report(res, info, lab, sysm.threshold)
