"""Drawing helpers shared by the templates: paper, background patterns, text and rules
offset by the page jitter, field rendering with ground-truth boxes, and style sampling.

Everything is deterministic given (values, content, style): the blended forger
re-renders the page and relies on getting the identical background back.
"""
import numpy as np
from PIL import Image, ImageDraw

from ..core.render import Style
from .base import BLANK


def paper(size, style):
    return Image.new("RGB", size, tuple(style.tint))


def watermark(img, style, text, fnt, step=(40, 150), start=(-60, 180), count=(-2, 6)):
    """Faint tiled diagonal text: background texture a flat fill destroys."""
    W, H = img.size
    wm = Image.new("L", (W, H), 0)
    wd = ImageDraw.Draw(wm)
    for k in range(*count):
        wd.text((start[0] + step[0] * k, start[1] + step[1] * k), text, fill=255, font=fnt)
    wm = wm.rotate(style.wm_angle, resample=Image.BILINEAR, center=(W // 2, H // 2))
    shade = Image.new("RGB", (W, H), tuple(max(0, c - style.wm_shade) for c in style.tint))
    return Image.composite(shade, img, wm)


def waves(img, style, shift=(16, 8, 20), n=70, amp=(5, 16), period=(80, 220), area=None, width=1):
    """Guilloche-like field of fine sine curves (security-print background)."""
    W, H = img.size
    x0, y0, x1, y1 = area or (0, 0, W, H)
    d = ImageDraw.Draw(img)
    r = np.random.default_rng(style.noise_seed % (1 << 31))
    color = tuple(int(np.clip(c - s, 0, 255)) for c, s in zip(style.tint, shift))
    xs = np.arange(x0, x1 + 3, 3, dtype=float)
    for i in range(n):
        base = y0 + (y1 - y0) * (i + 0.5) / n
        A, P, ph = r.uniform(*amp), r.uniform(*period), r.uniform(0, 2 * np.pi)
        ys = np.clip(base + A * np.sin(2 * np.pi * xs / P + ph), y0, y1)
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=color, width=width)


class Pen:
    """Static text and lines, shifted by the document's global offset."""

    def __init__(self, img, style):
        self.d = ImageDraw.Draw(img)
        self.ox, self.oy, self.ink = style.ox, style.oy, tuple(style.ink)
        self.W = img.width

    def text(self, x, y, s, f, align="l", fill=None, spacing=0):
        if spacing:  # letter-spaced headings
            w = sum(self.d.textlength(c, font=f) for c in s) + spacing * (len(s) - 1)
            x = (self.W - w) / 2 if align == "page" else x - w / 2 if align == "c" else x - w if align == "r" else x
            for c in s:
                self.d.text((x + self.ox, y + self.oy), c, fill=fill or self.ink, font=f)
                x += self.d.textlength(c, font=f) + spacing
            return
        w = self.d.textlength(s, font=f)
        x = (self.W - w) / 2 if align == "page" else x - w / 2 if align == "c" else x - w if align == "r" else x
        self.d.text((x + self.ox, y + self.oy), s, fill=fill or self.ink, font=f)

    def rule(self, x0, y, x1, t=1, fill=None):
        self.d.rectangle([x0 + self.ox, y + self.oy, x1 + self.ox, y + t - 1 + self.oy], fill=fill or self.ink)

    def rect(self, x0, y0, x1, y1, outline=None, fill=None, width=1):
        self.d.rectangle([x0 + self.ox, y0 + self.oy, x1 + self.ox, y1 + self.oy], outline=outline, fill=fill,
                         width=width)

    def ellipse(self, x0, y0, x1, y1, outline=None, fill=None, width=1):
        self.d.ellipse([x0 + self.ox, y0 + self.oy, x1 + self.ox, y1 + self.oy], outline=outline, fill=fill,
                       width=width)

    def squiggle(self, x, y, n=25, amp=4, step=6, width=2):
        pts = [(x + self.ox + step * i, y + self.oy + (amp if i % 2 else -amp)) for i in range(n)]
        self.d.line(pts, fill=self.ink, width=width)

    def portrait(self, x0, y0, x1, y1, shade):
        """Placeholder photo: a flat panel with a head-and-shoulders silhouette."""
        self.rect(x0, y0, x1, y1, fill=shade, outline=self.ink)
        cx, w, h = (x0 + x1) / 2, x1 - x0, y1 - y0
        dark = tuple(max(0, c - 45) for c in shade)
        self.ellipse(cx - w * 0.2, y0 + h * 0.18, cx + w * 0.2, y0 + h * 0.52, fill=dark)
        self.d.pieslice([cx - w * 0.38 + self.ox, y0 + h * 0.56 + self.oy, cx + w * 0.38 + self.ox,
                         y0 + h * 1.25 + self.oy], 180, 360, fill=dark)


def draw_fields(img, layout, schema, values, style):
    """Draw every field at its layout position; returns {field: tight text box}."""
    d = ImageDraw.Draw(img)
    ink = tuple(style.ink)
    boxes = {}
    for f in schema.fields:
        s = f.fmt(values[f.name])
        fnt = layout.font(f.name)
        xy = layout.origin(f.name, s, fnt, style)
        if s:
            d.text(xy, s, fill=ink, font=fnt)
            boxes[f.name] = list(d.textbbox(xy, s, font=fnt))
        else:
            boxes[f.name] = [xy[0], xy[1], xy[0] + 2, xy[1] + 2]
    return boxes


def sample_style(rng, fields, tint=None, ink=(15, 55), sigma=(2.0, 5.0), blur=(0.3, 0.7), q1=(82, 96),
                 offset=8, jitter=2):
    """Per-document paper, ink, layout jitter and acquisition parameters."""
    if tint is None:
        base = int(rng.integers(244, 253))
        tint = (base + int(rng.integers(0, 3)), base + int(rng.integers(-2, 2)), base - int(rng.integers(2, 12)))
    else:
        tint = tuple(int(np.clip(c + rng.integers(-3, 4), 0, 255)) for c in tint)
    g = int(rng.integers(*ink))
    return Style(ox=int(rng.integers(-offset, offset + 1)), oy=int(rng.integers(-offset, offset + 1)),
                 jitter={f: [int(rng.integers(-jitter, jitter + 1)), int(rng.integers(-jitter, jitter + 1))]
                         for f in fields},
                 tint=tint, ink=(g, g, g + int(rng.integers(0, 15))),
                 sigma=float(rng.uniform(*sigma)), blur=float(rng.uniform(*blur)), q1=int(rng.integers(*q1)),
                 wm_angle=float(rng.uniform(25, 40)), wm_shade=int(rng.integers(10, 18)),
                 noise_seed=int(rng.integers(1 << 30)))


def blank_or(fmt):
    """Formatter for optional fields: BLANK prints nothing."""
    return lambda v: "" if v == BLANK else fmt(v)


def parse_or_blank(parse):
    """Parser for optional fields: nothing printed reads as BLANK, not as unreadable."""
    return lambda s: BLANK if s == "" else parse(s)
