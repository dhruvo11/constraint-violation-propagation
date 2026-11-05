"""Forgeries of real receipt photos, with known ground truth (the taxonomy of
cvp/core/tamper.py, on real images).

T1 crude: the value is painted over in the paper's colour and retyped in another font.
T2 blended: the old digits are inpainted away and new ones drawn in the receipt's own ink
    colour, size and softness (an inpainting-like edit that leaves little visual trace).
T3/T4: coherent versions. An item price changes and every printed value that depends on
    it follows (subtotal, tax and service at their rates, total, then change or the cash
    paid), so every rule still holds; rendered crude (T3) or blended (T4).

Every output, genuine documents included, is re-encoded the same way (JPEG, quality 90),
so a detector cannot tell forgeries by their compression alone. A receipt is rebuilt from
its labelled words (cord.from_lines), so each schema field maps to the exact words that
print it.
"""
import io
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..core import render
from . import cord
from . import receipt_schema as RS

CATEGORIES = ["genuine", "T1", "T2", "T3", "T4"]
CRUDE = {"T1", "T3"}
FONTS_CRUDE = [render.FONT_SANS, render.FONT_SERIF, render.FONT_BOLD]
FONTS_BLEND = [render.FONT_MONO, render.FONT_MONO.replace("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf")]
QUALITY = 90
_CORE = re.compile(r"\d[\d.,\s]*\d|\d")


# ------------------------------------------------------------------ where fields are printed
def field_words(r):
    """Schema field name -> the value words that print it, numbering items as from_lines does."""
    groups = sorted({line["group_id"] for line in r.lines if line["category"].startswith("menu.")})
    item = {g: k for k, g in enumerate(groups, 1)}
    names = {"menu.price": "price", "menu.cnt": "cnt", "menu.unitprice": "unit"}
    out = {}
    for line in r.lines:
        cat, vw = line["category"], cord.value_words(line)
        if not vw:
            continue
        if cat in names:
            name = f"{names[cat]}{item[line['group_id']]}"
        elif cat in cord.AMOUNTS:
            name = cord.AMOUNTS[cat]
        elif cat == "total.menuqty_cnt":
            name = "qty"
        else:
            continue
        out.setdefault(name, []).extend(vw)
    return out


def bbox(words, size, pad=2):
    xs = [w["quad"][f"x{i}"] for w in words for i in range(1, 5)]
    ys = [w["quad"][f"y{i}"] for w in words for i in range(1, 5)]
    W, H = size
    return max(0, min(xs) - pad), max(0, min(ys) - pad), min(W, max(xs) + pad), min(H, max(ys) + pad)


# ------------------------------------------------------------------ printing like the receipt
def style(text):
    """(prefix, thousands separator, decimal separator, decimals, suffix) of a printed number."""
    m = _CORE.search(text)
    if not m:
        return "", "", None, 0, ""
    core = m.group().replace(" ", "")
    pre, suf = text[:m.start()], text[m.end():]
    d, c = core.rfind("."), core.rfind(",")
    if d >= 0 and c >= 0:
        dec = "." if d > c else ","
        return pre, ("," if dec == "." else "."), dec, len(core) - max(d, c) - 1, suf
    sep = "." if d >= 0 else ("," if c >= 0 else None)
    if sep is None:
        return pre, (" " if " " in m.group() else ""), None, 0, suf
    groups = core.split(sep)
    if 0 < len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:]):
        return pre, sep, None, 0, suf
    return pre, "", sep, len(groups[-1]), suf


def printed(value, like):
    """`value` printed in the style of the text `like`: same separators, decimals, prefix, suffix."""
    pre, th, dec, decimals, suf = style(like)
    s = f"{abs(value):,.{decimals}f}".replace(",", "\0").replace(".", "\1")
    return pre + s.replace("\0", th).replace("\1", dec or ".") + suf


def granularity(x):
    for g in (1000, 100, 10, 1):
        if x and float(x) % g == 0:
            return g
    return 0.01


def other_value(old, rng, is_count=False):
    """A different, plausible value: counts change by 1-3, amounts by 15-80% at the amount's
    own granularity (an edit no rounding tolerance absorbs). Like a careful forger, it keeps
    the number of digits where it can, so the new value fits where the old one was printed."""
    if is_count:
        return float(max(1, old + int(rng.integers(1, 4)) * (1 if old <= 1 or rng.random() < 0.6 else -1)))
    g = granularity(old)
    digits = len(str(int(abs(old))))
    new = None
    for _ in range(20):
        factor = rng.uniform(0.55, 0.85) if rng.random() < 0.4 else rng.uniform(1.15, 1.8)
        new = max(g, round(old * factor / g) * g)
        if abs(new - old) > max(1.0, 0.02 * abs(old)) and len(str(int(abs(new)))) == digits:
            return new
    return new if abs(new - old) > max(1.0, 0.02 * abs(old)) else old + 3 * g


# ------------------------------------------------------------------ rendering
def _ink(region):
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m > 0


def _fit(font_path, text, width, height):
    """The largest size at which `text` fits in width x height (nothing may be clipped:
    a clipped leading digit would turn a forged 166,909 into a misread 66,909)."""
    size = max(6, int(height * 1.3))
    while size > 6:
        f = ImageFont.truetype(font_path, size)
        l, t, r, b = f.getbbox(text)
        if (b - t) <= height and (r - l) <= width - 1:
            return f
        size -= 1
    return ImageFont.truetype(font_path, 6)


def _density(font, text, w, h):
    layer = Image.new("L", (w, h), 0)
    ImageDraw.Draw(layer).text((0, 0), text, fill=255, font=font)
    a = np.asarray(layer) > 127
    return a.sum() / max(1, a.any(0).sum() * a.any(1).sum())


def draw_value(img, box, text, crude, rng):
    """Replace the value printed in `box` of `img` (RGB uint8, edited in place) by `text`."""
    x0, y0, x1, y1 = box
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    reg = img[y0:y1, x0:x1].copy()
    m = _ink(reg)
    if m.any():  # the strokes' core colour: their darkest quarter, not edge pixels mixed with paper
        lum = reg[m].mean(1)
        ink = np.median(reg[m][lum <= np.percentile(lum, 25)], 0)
    else:
        ink = np.array([40.0, 40.0, 40.0])
    paper = np.median(reg[~m], 0) if (~m).any() else np.array([235.0, 235.0, 235.0])
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    th = int(rows.max() - rows.min() + 1) if rows.size else (y1 - y0)
    top = int(rows.min()) if rows.size else 0
    right = int(cols.max()) + 1 if cols.size else (x1 - x0)
    if crude:
        img[y0:y1, x0:x1] = paper.astype(np.uint8)  # the flat patch of a crude edit
        font, colour = _fit(FONTS_CRUDE[int(rng.integers(len(FONTS_CRUDE)))], text, right, th), (10, 10, 10)
    else:
        target = m[rows.min():rows.max() + 1, cols.min():cols.max() + 1].mean() if m.any() else 0.3
        fonts = [_fit(p, text, right, th) for p in FONTS_BLEND]  # regular or bold: whichever matches the
        font = min(fonts, key=lambda f: abs(_density(f, text, x1 - x0 + 40, y1 - y0 + 40) - target))  # print's weight
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[y0:y1, x0:x1] = cv2.dilate(m.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1)
        img[:] = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        noise = float(np.std(reg[~m])) if (~m).sum() > 10 else 3.0  # keep the paper's grain
        patch = img[y0:y1, x0:x1].astype(np.float32)
        patch[mask[y0:y1, x0:x1] > 0] += rng.normal(0, max(1.0, 0.8 * noise), (int((mask[y0:y1, x0:x1] > 0).sum()), 3))
        img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
        colour = tuple(float(c) for c in ink)
    layer = Image.new("L", (x1 - x0, y1 - y0), 0)
    d = ImageDraw.Draw(layer)
    l, t, r, b = font.getbbox(text)
    d.text((max(0, right - (r - l)) - l, top - t), text, fill=255, font=font)  # right-aligned, as amounts are
    if not crude:
        layer = layer.filter(ImageFilter.GaussianBlur(0.8))  # the photo's softness
    a = np.asarray(layer, np.float32)[..., None] / 255.0
    reg = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = np.clip(reg * (1 - a) + np.array(colour, np.float32) * a, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ coherent edits
def _convention(v, keys):
    """Does the receipt add its tax to the total ('excl') or are prices tax-inclusive ('incl')?"""
    sub = v.get("subtotal") if v.get("subtotal") is not None else sum(v[p] for p in keys)
    if v.get("total") is None or not v.get("tax"):
        return "excl"
    excl = sub + v["tax"] + (v.get("service") or 0) - abs(v.get("discount") or 0)
    return "excl" if abs(excl - v["total"]) <= max(1.0, 0.005 * abs(v["total"])) else "incl"


def coherent(rg, rng):
    """{field: new value} for a coherent edit of receipt `rg`: an item price changes and the
    printed values that depend on it follow, so every rule still holds. None if the receipt
    has no item price a rule reads, or no consistent repair exists."""
    S = RS.schema(rg)
    if "sum" not in S.constraint_names:
        return None
    v = RS.values(rg)
    prices = [f for f in S.field_names if f.startswith("price")]
    nonzero = [f for f in prices if v[f]]
    if not nonzero:
        return None
    f = nonzero[int(rng.integers(len(nonzero)))]
    k = int(f[5:])
    new = other_value(v[f], rng)
    nv = dict(v)
    if f"line{k}" in S.constraint_names:  # keep count x unit price = price
        g = granularity(v[f"unit{k}"])
        nv[f"unit{k}"] = max(g, round(new / v[f"cnt{k}"] / g) * g)
        new = v[f"cnt{k}"] * nv[f"unit{k}"]
    nv[f] = new
    d = new - v[f]
    conv = _convention(v, prices)
    base_old = v["subtotal"] if v.get("subtotal") is not None else sum(v[p] for p in prices)
    if v.get("subtotal") is not None:
        nv["subtotal"] = v["subtotal"] + d
    dt = d
    for t in ("tax", "service"):
        if v.get(t):
            nv[t] = round(v[t] * (base_old + d) / base_old)
            if t == "service" or conv == "excl":
                dt += nv[t] - v[t]
    if v.get("total") is not None:
        nv["total"] = v["total"] + dt
        if v.get("card") is not None and abs(v["card"] - v["total"]) < 1.0:
            nv["card"] = nv["total"]
        if v.get("cash") is not None:
            exact = abs(v["cash"] + (v.get("card") or 0) - v["total"]) <= 1  # paid exactly: stays exact
            if exact:
                nv["cash"] = nv["total"] - (nv.get("card") or 0)
            elif nv["cash"] + (nv.get("card") or 0) < nv["total"]:
                step = 50000 if nv["total"] > 100000 else 10000
                nv["cash"] = float(np.ceil(nv["total"] / step) * step)
            if v.get("change") is not None and not exact:
                nv["change"] = nv["cash"] + (nv.get("card") or 0) - nv["total"] \
                    if abs(v["cash"] + (v.get("card") or 0) - v["total"] - v["change"]) <= 1 else nv["cash"] - nv["total"]
    w, _ = S.evaluate(nv)
    if S.violated(w):
        return None
    return {x: nv[x] for x in nv if nv[x] is not None and nv[x] != v.get(x)}


# ------------------------------------------------------------------ the forger
def forge(r, category, rng):
    """(image as RGB uint8, info) for receipt `r` (loaded with images=True). info: category,
    edited fields with their old and new values, and the boxes of the edited fields."""
    rg = cord.from_lines(r.lines, rid=r.id, size=r.size)
    img = np.array(Image.open(io.BytesIO(r.image)).convert("RGB"))
    size = (img.shape[1], img.shape[0])
    words = field_words(r)
    S = RS.schema(rg)
    v = RS.values(rg)
    changes = {}
    if category in ("T1", "T2"):
        editable = [f for f in S.field_names if S.constrained(f) and f in words and v[f]]
        if editable:
            f = editable[int(rng.integers(len(editable)))]
            changes = {f: other_value(v[f], rng, is_count=f.startswith(("cnt", "qty")))}
    elif category in ("T3", "T4"):
        changes = coherent(rg, rng) or {}
        changes = {f: x for f, x in changes.items() if f in words}
    rects = []
    for f, new in changes.items():
        for w in words[f]:  # every numeric word of the field, each in its own box
            if not any(ch.isdigit() for ch in w["text"]):
                continue
            box = bbox([w], size)  # a longer value is drawn smaller rather than outside the box
            draw_value(img, box, printed(new, w["text"]), category in CRUDE, rng)
            rects.append(box)
    img = render.jpeg(img, QUALITY)
    edited = sorted(changes)
    return img, dict(category=category if edited or category == "genuine" else "genuine", edited=edited,
                     old={f: v[f] for f in edited}, new={f: changes[f] for f in edited}, rects=rects)
