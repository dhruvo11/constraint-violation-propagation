"""Rendering utilities shared by every document template: fonts, the acquisition model
(blur, sensor noise, first JPEG compression) and the Document record.

Every field's bounding box is recorded at render time, so each sample comes with
per-field localisation ground truth and no manual annotation.
"""
import datetime as dt
import io
import os
from dataclasses import asdict, dataclass, field

import matplotlib
import numpy as np
from PIL import Image, ImageFilter, ImageFont

_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
FONT_SANS = os.path.join(_FONT_DIR, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")
FONT_SERIF = os.path.join(_FONT_DIR, "DejaVuSerif.ttf")
FONT_MONO = os.path.join(_FONT_DIR, "DejaVuSansMono.ttf")
_font_cache = {}


def font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


@dataclass
class Style:
    """Per-document rendering parameters: layout jitter, paper and ink, acquisition."""
    ox: int
    oy: int
    jitter: dict
    tint: tuple
    ink: tuple
    sigma: float
    blur: float
    q1: int
    wm_angle: float
    wm_shade: int
    noise_seed: int


@dataclass
class Document:
    """A rendered document: the image plus everything needed to re-render or score it."""
    doctype: str
    image: np.ndarray
    values: dict
    boxes: dict
    content: dict
    style: Style
    meta: dict = field(default_factory=dict)


def jpeg_bytes(arr, q):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=int(q))
    return buf.getvalue()


def decode(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def jpeg(arr, q):
    return decode(jpeg_bytes(arr, q))


def finish(img, style, sigma=None, blur=None, seed=None):
    """Acquisition model: optical blur, sensor noise, first JPEG compression."""
    sigma = style.sigma if sigma is None else sigma
    blur = style.blur if blur is None else blur
    rng = np.random.default_rng(style.noise_seed if seed is None else seed)
    img = img.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    lum = rng.normal(0, sigma, (h, w, 1))
    chroma = rng.normal(0, sigma * 0.3, (h, w, 3))
    a = np.clip(a + lum + chroma, 0, 255).astype(np.uint8)
    return jpeg(a, style.q1)


def render(doctype, values, content, style):
    img, boxes = doctype.render_clean(values, content, style)
    return Document(doctype=doctype.name, image=finish(img, style), values=dict(values), boxes=boxes,
                    content=content, style=style)


def generate(doctype, rng):
    values, content, style = doctype.sample_spec(rng)
    return render(doctype, values, content, style)


def to_json(values):
    return {k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in values.items()}


def style_dict(style):
    return asdict(style)
