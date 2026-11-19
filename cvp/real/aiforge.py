"""AIForge-Doc (Wu et al., 2026): CORD receipts whose one field a commercial AI image editor
rewrote (v1: Gemini 2.5 Flash Image and Ideogram v2 Edit; v2: GPT-Image-2 on the same edits).
The data is gated on Hugging Face (CC BY-NC-SA 4.0) and lives in data/real/aiforge/<version>.
Shared by scripts/real_aiforge.py (the evaluation) and the web app (its samples)."""
import io
import json
import os
import re

import numpy as np
from PIL import Image

from .. import paths
from . import forge, money

REPOS = {"v1": "Scam-AI/AIForge-Doc-v1", "v2": "Scam-AI/AIForge-Doc-v2"}
ROOT = paths.real_dir("aiforge")
TOOLS = {"openai-gpt-image-2": "GPT-Image-2", "gemini-nano": "Gemini 2.5 Flash Image",
         "qwen-inpaint": "Ideogram v2 Edit"}  # the v1 metadata calls Ideogram "qwen-inpaint" (see its card)
# CORD keys of text fields (item names and codes): no rule reads them, so an edit of one is unconstrained
# even where its box overlaps the printed words of a number next to it
TEXT_FIELDS = ("menu.nm", "menu.num", "menu.sub.nm", "menu.sub_nm")


def local(version):
    return os.path.join(ROOT, version)


def available(version):
    return os.path.exists(os.path.join(local(version), "metadata.jsonl"))


def metadata(version):
    rows = [json.loads(line) for line in open(os.path.join(local(version), "metadata.jsonl"))]
    rows = [r for r in rows if r.get("source_dataset") == "cord"]
    if version == "v1" and not any("image_id" in r for r in rows):
        # v1 does not record the source image; v2 reuses v1's specifications under the same
        # spec_id and does, so v1 forgeries are located through v2's metadata
        v2 = {r["spec_id"]: r["image_id"] for r in metadata("v2")}
        rows = [dict(r, image_id=v2[r["spec_id"]]) for r in rows if r["spec_id"] in v2]
    return rows


def cord_index(row):
    """(split, k) of the CORD receipt a forgery was made from: AIForge-Doc numbers CORD v2
    train 0-799, validation 800-899, test 900-999."""
    for key in ("image_id", "source_index", "image", "authentic"):
        m = re.search(r"cord[_-]?(\d+)", str(row.get(key, "")))
        if m:
            n = int(m.group(1))
            break
    else:
        raise ValueError(f"no CORD index in {row}")
    return ("train", n) if n < 800 else ("validation", n - 800) if n < 900 else ("test", n - 900)


def rgb(path_or_bytes):
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    return np.asarray(Image.open(src).convert("RGB"))


def edited_field(words, mask, size, fields):
    """The schema field whose printed words the edit mask covers most (None: a field no rule reads)."""
    best, best_px = None, 0
    for f in fields:
        if f not in words:
            continue
        x0, y0, x1, y1 = (int(round(c)) for c in forge.bbox(words[f], size, pad=0))
        px = int(mask[max(0, y0):y1 + 1, max(0, x0):x1 + 1].sum())
        if px > best_px:
            best, best_px = f, px
    return best


def new_value(field, text):
    return money.count(text) if field == "qty" or field.startswith("cnt") else money.amount(text)
