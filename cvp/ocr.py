"""OCR and positional field assignment.

The page is registered against the template using the header rule, each schema slot is
cropped, and a small CRNN (conv stack + BiLSTM, CTC-trained on synthetic slot crops)
reads it with decoding restricted to the slot's character set. The output is a value,
a confidence and a bounding box per field: everything downstream sees only what the
OCR read, never the ground truth.
"""
import datetime as dt
import os
import re
import string

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .render import ANCHORS, HEADER_RULE, SLOT_WIDTH
from .schema import FIELDS, GRADES, MARK_FIELDS

ALPHABET = string.digits + string.ascii_uppercase + "./+"
CHAR_INDEX = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 = CTC blank
CHARSET = {**{m: string.digits for m in MARK_FIELDS}, "total": string.digits,
           "pct": string.digits + ".", "date": string.digits + "/", "grade": "OABCDF+",
           "roll": string.digits + string.ascii_uppercase}
CROP_H, CROP_W = 32, 160
THRESH = 150
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "ocr_crnn.pt")


class CRNN(nn.Module):
    def __init__(self, n_classes=len(ALPHABET) + 1):
        super().__init__()

        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

        self.cnn = nn.Sequential(*block(1, 32), nn.MaxPool2d(2), *block(32, 64), nn.MaxPool2d(2),
                                 *block(64, 128), *block(128, 128), nn.MaxPool2d((2, 1)),
                                 *block(128, 192), nn.MaxPool2d((4, 1)))
        self.rnn = nn.LSTM(192, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):  # x: B,1,32,160 -> B,T=40,C
        f = self.cnn(x).squeeze(2).transpose(1, 2)
        return self.fc(self.rnn(f)[0])


def register(gray):
    """Estimate the global page offset from the header rule."""
    x0, y0 = HEADER_RULE
    top = max(0, y0 - 16)
    band = gray[top:y0 + 16] < THRESH
    rows = np.nonzero(band.sum(1) > 300)[0]
    if len(rows) == 0:
        return 0, 0
    r = rows[0]
    cols = np.nonzero(band[r])[0]
    return int(cols[0] - x0), int(r + top - y0)


def slot_rect(f, ox, oy, shape=(832, 640)):
    ax, ay = ANCHORS[f]
    return (max(0, ax + ox - 10), max(0, ay + oy - 8),
            min(shape[1], ax + ox + SLOT_WIDTH[f]), min(shape[0], ay + oy + 30))


def crop_slot(gray, f, ox, oy):
    x0, y0, x1, y1 = slot_rect(f, ox, oy, gray.shape)
    c = gray[y0:y1, x0:x1]
    scale = CROP_H / c.shape[0]
    w = min(CROP_W, max(1, int(round(c.shape[1] * scale))))
    im = np.asarray(Image.fromarray(c.astype(np.uint8)).resize((w, CROP_H), Image.BILINEAR), np.float32)
    out = np.full((CROP_H, CROP_W), float(np.median(c)), np.float32)
    out[:, :w] = im
    return out / 255.0, (x0, y0, x1, y1)


def ink_box(gray, rect):
    """Tight box of the ink inside a slot, ignoring table rules."""
    x0, y0, x1, y1 = rect
    c = gray[y0:y1, x0:x1]
    thr = min(THRESH, 0.5 * (np.median(c) + np.percentile(c, 1)))
    m = c < thr
    m[m.sum(1) > 0.8 * m.shape[1]] = False
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return [x0, y0, x0 + 4, y0 + 4]
    return [int(x0 + xs.min()), int(y0 + ys.min()), int(x0 + xs.max() + 1), int(y0 + ys.max() + 1)]


def parse(f, s):
    if f in MARK_FIELDS or f == "total":
        return int(s) if s.isdigit() else None
    if f == "pct":
        return float(s) if re.fullmatch(r"\d{1,3}\.\d{2}", s) else None
    if f == "grade":
        return s if s in GRADES else None
    if f == "date":
        m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", s)
        if not m:
            return None
        try:
            return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    if f == "roll":
        return s or None
    raise KeyError(f)


_MASKS = {f: torch.tensor([0.0] + [0.0 if c in CHARSET[f] else -1e4 for c in ALPHABET]) for f in FIELDS}


def decode(logits, f):
    """Greedy CTC decoding restricted to the slot's character set."""
    p = (logits + _MASKS[f]).softmax(-1)
    conf, idx = p.max(-1)
    out, prev, confs = [], 0, []
    for k, c in zip(idx.tolist(), conf.tolist()):
        if k != 0 and k != prev:
            out.append(ALPHABET[k - 1])
            confs.append(c)
        prev = k
    return "".join(out), float(np.mean(confs)) if confs else 0.0


class Reader:
    def __init__(self, path=MODEL_PATH, device="cpu"):
        self.model = CRNN()
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def read(self, img):
        """OCR + FieldAssign. Returns dict(values, text, conf, boxes, offset)."""
        gray = np.asarray(img, np.float32).mean(2) if img.ndim == 3 else img.astype(np.float32)
        ox, oy = register(gray)
        crops, rects = zip(*[crop_slot(gray, f, ox, oy) for f in FIELDS])
        x = torch.from_numpy(np.stack(crops))[:, None].to(self.device)
        logits = self.model(x).float().cpu()
        values, texts, conf, boxes = {}, {}, {}, {}
        for i, f in enumerate(FIELDS):
            texts[f], conf[f] = decode(logits[i], f)
            values[f] = parse(f, texts[f])
            boxes[f] = ink_box(gray, rects[i])
        return dict(values=values, text=texts, conf=conf, boxes=boxes, offset=(ox, oy))
