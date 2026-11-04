"""Tampering taxonomy (Table IV).

T1 crude   : single-field edit, strong visual trace, constraint violation present.
T2 blended : single-field edit re-synthesised in the document's own font, noise and
             compression (an emulation of generative inpainting), weak visual trace,
             constraint violation present.
T3 coherent: the edit is propagated to every dependent field (crude rendering), so the
             constraint system is satisfied; strong visual trace.
T4 complete: coherent and blended; beyond the reach of both branches by construction.
"""
import datetime as dt

import numpy as np
from PIL import Image, ImageDraw

from .render import (FONT_BOLD, FONT_MONO, FONT_SANS, FONT_SERIF, Document, anchor, decode,
                     finish, font, jpeg_bytes, render_clean)
from .schema import (BANDS, DEPTS, FIELDS, GRADES, MARK_FIELDS, MAX_TOTAL, TODAY, beta,
                     coherent_values, fmt)

CATEGORIES = ["genuine", "T1", "T2", "T3", "T4"]
ALT_FONTS = [(FONT_BOLD, 16), (FONT_SERIF, 16), (FONT_MONO, 15), (FONT_SANS, 18), (FONT_SANS, 14)]
H_IMG, W_IMG = 832, 640


def _pad(box, p):
    x0, y0, x1, y1 = box
    return [max(0, x0 - p), max(0, y0 - p), min(W_IMG, x1 + p), min(H_IMG, y1 + p)]


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


# ---------------------------------------------------------------- value models
def violating_value(f, values, rng):
    """New value for a single-field edit (T1/T2); the edit always breaks a constraint."""
    v = values[f]
    if f in MARK_FIELDS:
        if rng.random() < 0.15:  # out of range -> {c1, c4_i}
            return int(rng.integers(101, 200))
        for _ in range(100):
            step = int(rng.integers(5, 31)) * (1 if rng.random() < 0.8 else -1)
            new = int(np.clip(v + step, 0, 100))
            if new != v:
                return new
        return (v + 7) % 100
    if f == "total":
        new = v + int(rng.integers(10, 91)) * (1 if rng.random() < 0.8 else -1)
        return int(np.clip(new, 100, 499)) if new != v else v + 11
    if f == "pct":
        for _ in range(100):
            new = round(v + float(rng.uniform(2, 15)) * (1 if rng.random() < 0.8 else -1), 2)
            if 0 <= new <= 100 and abs(new - v) > 0.5:
                return new
        return round(max(0.0, v - 5.0), 2)
    if f == "grade":
        i = GRADES.index(v)
        options = [g for g in GRADES[max(0, i - 2):i] or GRADES[i + 1:i + 2]]
        return options[rng.integers(len(options))]
    if f == "date":
        return TODAY + dt.timedelta(days=int(rng.integers(20, 700)))
    if f == "roll":
        op = rng.integers(3)
        if op == 0:
            bad = ["CSX", "CSA", "ECX", "MEE", "CIX", "EEC", "AIX"]
            return v[:6] + bad[rng.integers(len(bad))] + v[9:]
        if op == 1:
            return ["1RUB", "1RVA", "1RUC", "2RUA"][rng.integers(4)] + v[4:]
        return v[:-1] if rng.random() < 0.5 else v + str(int(rng.integers(0, 10)))
    raise KeyError(f)


def coherent_edit(values, rng):
    """Multi-field edit that keeps every constraint satisfied (T3/T4)."""
    u = rng.random()
    new = dict(values)
    if u < 0.6:
        m = MARK_FIELDS[rng.integers(5)]
        for _ in range(100):
            x = int(np.clip(values[m] + int(rng.integers(5, 31)), 0, 100))
            if x != values[m]:
                break
        marks = [x if k == m else values[k] for k in MARK_FIELDS]
        new = coherent_values(marks, values["roll"], values["date"])
        edited = [f for f in FIELDS if fmt(f, new[f]) != fmt(f, values[f])]
        return new, edited, m
    if u < 0.8:
        for _ in range(100):
            d = values["date"] + dt.timedelta(days=int(rng.integers(-400, 400)))
            if d <= TODAY and d != values["date"] and d.year >= 2016:
                break
        new["date"] = d
        return new, ["date"], "date"
    for _ in range(100):
        r = values["roll"][:9] + f"{int(rng.integers(1, 10000)):04d}"
        if r != values["roll"]:
            break
    new["roll"] = r
    return new, ["roll"], "roll"


# ---------------------------------------------------------------- renderers
def crude_edit(arr, doc, f, new_text, rng, alt_font=None):
    """Erase the field with a flat fill and retype it in a mismatched font."""
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    path, size = alt_font if alt_font is not None else ALT_FONTS[rng.integers(len(ALT_FONTS))]
    ff = font(path, size)
    ax, ay = anchor(f, doc.style)
    xy = (ax + int(rng.integers(-2, 3)), ay + int(rng.integers(-2, 3)))
    new_box = list(d.textbbox(xy, new_text, font=ff))
    rect = _pad(_union(doc.boxes[f], new_box), int(rng.integers(2, 6)))
    x0, y0, x1, y1 = rect
    ring = np.concatenate([arr[max(0, y0 - 4):y0, x0:x1].reshape(-1, 3),
                           arr[y1:y1 + 4, x0:x1].reshape(-1, 3)])
    fill = tuple(int(c) for c in np.median(ring, axis=0)) if len(ring) else doc.style.tint
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill)
    g = int(rng.integers(0, 40))
    d.text(xy, new_text, fill=(g, g, g), font=ff)
    return np.asarray(img), rect, new_box


# Generator recipes: (noise scale range, extra blur range, feathered mask).
# "g2" is the attack evaluated as T2/T4; "g1" is an older, cruder generator whose
# forgeries a deployed detector could have been trained on.
GENERATORS = {"g2": ((0.5, 0.85), (0.3, 0.7), True), "g1": ((0.15, 0.45), (0.9, 1.5), False)}


def blended_edit(arr, doc, new_values, fields, rng, generator="g2"):
    """Re-synthesise the fields in the document's own font, texture and compression, with
    the mild over-smoothing and noise loss characteristic of generative inpainting."""
    (n0, n1), (b0, b1), feather = GENERATORS[generator]
    clean, boxes = render_clean(new_values, doc.content, doc.style)
    gen = finish(clean, doc.style, sigma=doc.style.sigma * float(rng.uniform(n0, n1)),
                 blur=doc.style.blur + float(rng.uniform(b0, b1)), seed=int(rng.integers(1 << 30)))
    out = arr.astype(np.float32)
    rects = []
    for f in fields:
        rect = _pad(_union(doc.boxes[f], boxes[f]), 8)
        x0, y0, x1, y1 = rect
        alpha = np.zeros((H_IMG, W_IMG), np.float32)
        alpha[y0:y1, x0:x1] = 1.0
        if feather:
            alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize(
                (W_IMG // 4, H_IMG // 4), Image.BILINEAR).resize((W_IMG, H_IMG), Image.BILINEAR),
                np.float32) / 255.0
        alpha = alpha[..., None]
        alpha = np.maximum(alpha, 0)
        out = alpha * gen + (1 - alpha) * out
        rects.append(_pad(rect, 4))
    return np.clip(out, 0, 255).astype(np.uint8), rects, {f: boxes[f] for f in fields}


# ---------------------------------------------------------------- top level
def apply(doc: Document, category, rng, field=None, generator="g2"):
    """Returns (image after final JPEG save, info dict)."""
    arr = doc.image
    values = dict(doc.values)
    boxes = {k: list(v) for k, v in doc.boxes.items()}
    rects, edited, primary = [], [], None
    if category in ("T1", "T2"):
        f = field or FIELDS[rng.choice(len(FIELDS), p=FIELD_PROBS)]
        values[f] = violating_value(f, doc.values, rng)
        edited, primary = [f], f
        if category == "T1":
            arr, rect, nb = crude_edit(arr, doc, f, fmt(f, values[f]), rng)
            rects, boxes[f] = [rect], nb
        else:
            arr, rects, nb = blended_edit(arr, doc, values, [f], rng, generator)
            boxes.update(nb)
    elif category in ("T3", "T4"):
        values, edited, primary = coherent_edit(doc.values, rng)
        if category == "T3":
            alt = ALT_FONTS[rng.integers(len(ALT_FONTS))]
            for f in edited:
                arr, rect, nb = crude_edit(arr, doc, f, fmt(f, values[f]), rng, alt_font=alt)
                rects.append(rect)
                boxes[f] = nb
        else:
            arr, rects, nb = blended_edit(arr, doc, values, edited, rng, generator)
            boxes.update(nb)
    q2 = int(rng.integers(75, 93))
    data = jpeg_bytes(arr, q2)  # every document, genuine or not, is re-saved once more
    out = decode(data)
    info = dict(jpeg=data, category=category, generator=generator if category in ("T2", "T4") else None, edited=edited, primary=primary, rects=rects, boxes=boxes,
                values=values, q2=q2, mark_out_of_range=bool(primary in MARK_FIELDS and
                                                          category in ("T1", "T2") and
                                                          values[primary] > 100))
    return out, info


# Single-field edits: half of them on a subject mark (the ambiguous class).
FIELD_PROBS = np.array([0.1] * 5 + [0.1] * 5)
