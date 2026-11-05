"""What the visual branch sees of a real photo, made the same way in training and testing:
scaled so its longer side is at most MAX_SIDE (text then has roughly the size the branch
learned on, and a 12-megapixel photo fits in memory), then encoded once as JPEG at QUALITY.
Field boxes scale with it. Reading (OCR) always uses the full-resolution image.
"""
import io

import numpy as np
from PIL import Image

from ..core import render

MAX_SIDE = 1600
QUALITY = 95


def visual_view(img, with_bytes=False, min_side=0):
    """(view as RGB uint8, scale[, its JPEG bytes]) of photo `img` (RGB uint8). `min_side`
    pads a narrow view (edge pixels, right and bottom, so boxes keep their coordinates) before
    its single encoding: training patches need at least that many pixels."""
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1.0:
        img = np.asarray(Image.fromarray(img).resize((round(w * s), round(h * s)), Image.BILINEAR))
    if min_side and (img.shape[0] < min_side or img.shape[1] < min_side):
        img = np.pad(img, ((0, max(0, min_side - img.shape[0])), (0, max(0, min_side - img.shape[1])), (0, 0)), mode="edge")
    data = render.jpeg_bytes(img, QUALITY)
    view = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    return (view, s, data) if with_bytes else (view, s)


def scale_box(box, s):
    return [int(round(c * s)) for c in box]
