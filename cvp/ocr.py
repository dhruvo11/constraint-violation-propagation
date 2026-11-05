"""OCR and positional field assignment for template documents.

The page is registered against the template using the header rule, each schema slot is
cropped, and a small CRNN (conv stack + BiLSTM, CTC-trained on synthetic slot crops)
reads it with decoding restricted to the slot's character set. The output is a value,
a confidence and a bounding box per field: everything downstream sees only what the
OCR read, never the ground truth.
"""
import os
import string

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ALPHABET = string.digits + string.ascii_uppercase + "./+,-:<% "
CHAR_INDEX = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 = CTC blank
CROP_H, CROP_W = 32, 160
WIDE_W = 640  # long lines: machine-readable zones, amounts in words
THRESH = 150
MAX_GAP = 8       # px: a broken-up rule is still one rule
MAX_OFFSET = 40   # px: pages are placed within a few pixels of the template
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

    def forward(self, x):  # x: B,1,32,W -> B,T=W/4,C (W = 160 for slots, 640 for long lines)
        f = self.cnn(x).squeeze(2).transpose(1, 2)
        return self.fc(self.rnn(f)[0])


def register(gray, header_rule, min_run=300):
    """Estimate the global page offset from the header rule."""
    x0, y0 = header_rule
    top = max(0, y0 - 16)
    band = gray[top:y0 + 16] < THRESH
    rows = np.nonzero(band.sum(1) > min_run)[0]
    if len(rows) == 0:
        return 0, 0
    r = rows[0]
    cols = np.nonzero(band[r])[0]
    # Left end of the longest dark run, so a page border or text in the same row cannot shift
    # it; gaps of a few pixels (a rule broken up by compression or rescaling) do not split a run.
    breaks = np.nonzero(np.diff(cols) > MAX_GAP)[0]
    starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(cols) - 1]
    i = int(np.argmax(cols[ends] - cols[starts]))
    ox, oy = int(cols[starts[i]] - x0), int(r + top - y0)
    if abs(ox) > MAX_OFFSET or abs(oy) > MAX_OFFSET:  # not the header rule: assume no offset
        return 0, 0
    return ox, oy


def slot_rect(doctype, f, ox, oy, shape):
    ax, ay, width, *ud = doctype.slots[f]
    up, down = ud if ud else (8, 30)
    return (max(0, ax + ox - 10), max(0, ay + oy - up),
            min(shape[1], ax + ox + width), min(shape[0], ay + oy + down))


def crop_width(doctype, f):
    return WIDE_W if f in doctype.wide else CROP_W


def crop_slot(gray, doctype, f, ox, oy):
    x0, y0, x1, y1 = slot_rect(doctype, f, ox, oy, gray.shape)
    c = gray[y0:y1, x0:x1]
    cw = crop_width(doctype, f)
    if c.size == 0:  # the slot fell off the page: an empty (unreadable) crop
        return np.full((CROP_H, cw), float(np.median(gray)) / 255.0, np.float32), (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))
    scale = CROP_H / c.shape[0]
    w = min(cw, max(1, int(round(c.shape[1] * scale))))
    im = np.asarray(Image.fromarray(c.astype(np.uint8)).resize((w, CROP_H), Image.BILINEAR), np.float32)
    out = np.full((CROP_H, cw), float(np.median(c)), np.float32)
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


_masks = {}


def _mask(charset):
    if charset not in _masks:
        _masks[charset] = torch.tensor([0.0] + [0.0 if c in charset else -1e4 for c in ALPHABET])
    return _masks[charset]


def decode(logits, charset):
    """Greedy CTC decoding restricted to the slot's character set."""
    p = (logits + _mask(charset)).softmax(-1)
    conf, idx = p.max(-1)
    out, prev, confs = [], 0, []
    for k, c in zip(idx.tolist(), conf.tolist()):
        if k != 0 and k != prev:
            out.append(ALPHABET[k - 1])
            confs.append(c)
        prev = k
    return "".join(out), float(np.mean(confs)) if confs else 0.0


def gray_of(img):
    return np.asarray(img, np.float32).mean(2) if img.ndim == 3 else img.astype(np.float32)


class Reader:
    def __init__(self, path=MODEL_PATH, device="cpu"):
        self.model = CRNN()
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def read(self, img, doctype):
        """OCR + FieldAssign. Returns dict(values, text, conf, boxes, offset)."""
        gray = gray_of(img)
        ox, oy = register(gray, doctype.header_rule, doctype.rule_min)
        fields = doctype.schema.fields
        crops, rects = zip(*[crop_slot(gray, doctype, f.name, ox, oy) for f in fields])
        logits = [None] * len(fields)
        for width in sorted({c.shape[1] for c in crops}):  # one batch per crop width
            idx = [i for i, c in enumerate(crops) if c.shape[1] == width]
            x = torch.from_numpy(np.stack([crops[i] for i in idx]))[:, None].to(self.device)
            for i, lg in zip(idx, self.model(x).float().cpu()):
                logits[i] = lg
        values, texts, conf, boxes = {}, {}, {}, {}
        for i, f in enumerate(fields):
            texts[f.name], conf[f.name] = decode(logits[i], f.charset)
            values[f.name] = f.parse(texts[f.name])
            boxes[f.name] = ink_box(gray, rects[i])
        return dict(values=values, text=texts, conf=conf, boxes=boxes, offset=(ox, oy))
