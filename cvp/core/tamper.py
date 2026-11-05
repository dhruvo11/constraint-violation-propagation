"""Tampering taxonomy (Table IV), shared by every document type.

T1 crude   : single-field edit, strong visual trace, constraint violation present.
T2 blended : single-field edit re-synthesised in the document's own font, noise and
             compression (an emulation of generative inpainting), weak visual trace,
             constraint violation present.
T3 coherent: the edit is propagated to every dependent field (crude rendering), so the
             constraint system is satisfied; strong visual trace.
T4 complete: coherent and blended; beyond the reach of both branches by construction.

What a forger writes is document-specific (the type's `violating_value` and
`coherent_edit`); how it is rendered into the image is generic and lives here.
"""
import numpy as np
from PIL import Image, ImageDraw

from .render import FONT_BOLD, FONT_MONO, FONT_SANS, FONT_SERIF, Document, decode, finish, font, jpeg_bytes

CATEGORIES = ["genuine", "T1", "T2", "T3", "T4"]
ALT_FONTS = [(FONT_BOLD, 16), (FONT_SERIF, 16), (FONT_MONO, 15), (FONT_SANS, 18), (FONT_SANS, 14)]

# Generator recipes: (noise scale range, extra blur range, feathered mask).
# "g2" is the attack evaluated as T2/T4; "g1" is an older, cruder generator whose
# forgeries a deployed detector could have been trained on.
GENERATORS = {"g2": ((0.5, 0.85), (0.3, 0.7), True), "g1": ((0.15, 0.45), (0.9, 1.5), False)}


def _pad(box, p, shape):
    x0, y0, x1, y1 = box
    return [max(0, x0 - p), max(0, y0 - p), min(shape[1], x1 + p), min(shape[0], y1 + p)]


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def crude_edit(arr, doc, doctype, f, new_text, rng, alt_font=None):
    """Erase the field with a flat fill and retype it in a mismatched font."""
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    fonts = doctype.alt_fonts(f)  # some fields admit only certain fonts (an MRZ is monospaced)
    if alt_font is None:
        pool = fonts or ALT_FONTS
        path, size = pool[rng.integers(len(pool))]
    else:
        path, size = fonts[ALT_FONTS.index(alt_font) % len(fonts)] if fonts else alt_font
    ff = font(path, max(8, round(size * doctype.font_scale(f))))
    ax, ay = doctype.place(f, new_text, ff, doc.style)
    xy = (ax + int(rng.integers(-2, 3)), ay + int(rng.integers(-2, 3)))
    new_box = list(d.textbbox(xy, new_text, font=ff))
    rect = _pad(_union(doc.boxes[f], new_box), int(rng.integers(2, 6)), arr.shape)
    x0, y0, x1, y1 = rect
    ring = np.concatenate([arr[max(0, y0 - 4):y0, x0:x1].reshape(-1, 3),
                           arr[y1:y1 + 4, x0:x1].reshape(-1, 3)])
    fill = tuple(int(c) for c in np.median(ring, axis=0)) if len(ring) else doc.style.tint
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill)
    g = int(rng.integers(0, 40))
    d.text(xy, new_text, fill=(g, g, g), font=ff)
    return np.asarray(img), rect, new_box


def blended_edit(arr, doc, doctype, new_values, fields, rng, generator="g2"):
    """Re-synthesise the fields in the document's own font, texture and compression, with
    the mild over-smoothing and noise loss characteristic of generative inpainting."""
    (n0, n1), (b0, b1), feather = GENERATORS[generator]
    h, w = arr.shape[:2]
    clean, boxes = doctype.render_clean(new_values, doc.content, doc.style)
    gen = finish(clean, doc.style, sigma=doc.style.sigma * float(rng.uniform(n0, n1)),
                 blur=doc.style.blur + float(rng.uniform(b0, b1)), seed=int(rng.integers(1 << 30)))
    out = arr.astype(np.float32)
    rects = []
    for f in fields:
        rect = _pad(_union(doc.boxes[f], boxes[f]), 8, arr.shape)
        x0, y0, x1, y1 = rect
        alpha = np.zeros((h, w), np.float32)
        alpha[y0:y1, x0:x1] = 1.0
        if feather:
            alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize(
                (w // 4, h // 4), Image.BILINEAR).resize((w, h), Image.BILINEAR), np.float32) / 255.0
        alpha = alpha[..., None]
        out = alpha * gen + (1 - alpha) * out
        rects.append(_pad(rect, 4, arr.shape))
    return np.clip(out, 0, 255).astype(np.uint8), rects, {f: boxes[f] for f in fields}


def apply(doc: Document, doctype, category, rng, field=None, generator="g2"):
    """Forge `doc` according to `category`. Returns (image after the final JPEG save, info)."""
    schema = doctype.schema
    arr = doc.image
    values = dict(doc.values)
    boxes = {k: list(v) for k, v in doc.boxes.items()}
    rects, edited, primary = [], [], None
    if category in ("T1", "T2"):
        f = field or doctype.pick_field(doc.values, rng)
        values[f] = doctype.violating_value(f, doc.values, rng)
        edited, primary = [f], f
        text = schema.by_name[f].fmt(values[f])
        if category == "T1":
            arr, rect, nb = crude_edit(arr, doc, doctype, f, text, rng)
            rects, boxes[f] = [rect], nb
        else:
            arr, rects, nb = blended_edit(arr, doc, doctype, values, [f], rng, generator)
            boxes.update(nb)
    elif category in ("T3", "T4"):
        values, edited, primary = doctype.coherent_edit(doc.values, rng)
        if category == "T3":
            alt = ALT_FONTS[rng.integers(len(ALT_FONTS))]
            for f in edited:
                arr, rect, nb = crude_edit(arr, doc, doctype, f, schema.by_name[f].fmt(values[f]), rng, alt_font=alt)
                rects.append(rect)
                boxes[f] = nb
        else:
            arr, rects, nb = blended_edit(arr, doc, doctype, values, edited, rng, generator)
            boxes.update(nb)
    q2 = int(rng.integers(75, 93))
    data = jpeg_bytes(arr, q2)  # every document, genuine or not, is re-saved once more
    out = decode(data)
    unary = bool(primary is not None and category in ("T1", "T2")
                 and schema.unary_violated(primary, schema.evaluate(values)[0]))
    info = dict(jpeg=data, doctype=doctype.name, category=category,
                generator=generator if category in ("T2", "T4") else None, edited=edited, primary=primary,
                rects=rects, boxes=boxes, values=values, q2=q2, unary_violation=unary)
    return out, info
