"""How CVP reads a real receipt's values: every value word where the dataset's box puts it
(field locations known, as with a template), padded by a few pixels. The same policy
serves genuine and forged receipts alike.

A left margin of a character or two (amounts are right-aligned, so a longer value grows
to the left) is available but off: on CORD v2 test it raised the genuine receipts flagged
as read from 11 to 16 in 100, by pulling stray ink ("@", "x") into the crop. Forgeries
keep a value's length instead (cvp/real/forge.py), as a careful forger would.
"""
import numpy as np

PAD = 3
MARGIN_CHARS = 0


def _span(w):
    q = w["quad"]
    xs, ys = [q[f"x{i}"] for i in range(1, 5)], [q[f"y{i}"] for i in range(1, 5)]
    return min(xs), min(ys), max(xs), max(ys)


def left_bound(word, all_words):
    """Right edge of the nearest word to the left of `word` on the same row (0 if none)."""
    x0, y0, x1, y1 = _span(word)
    best = 0
    for o in all_words:
        if o is word:
            continue
        a0, b0, a1, b1 = _span(o)
        overlap = min(y1, b1) - max(y0, b0)
        if a1 <= x0 + 1 and overlap > 0.5 * min(y1 - y0, b1 - b0):
            best = max(best, a1)
    return best


def value_box(word, all_words, size, margin_chars=None):
    """The crop that reads `word`: its box, padded, and (if `margin_chars`, default
    MARGIN_CHARS at call time) widened to the left without reaching the neighbouring word."""
    margin_chars = MARGIN_CHARS if margin_chars is None else margin_chars
    x0, y0, x1, y1 = _span(word)
    W, H = size
    if margin_chars:
        char = (x1 - x0) / max(1, len(word["text"]))
        left = max(left_bound(word, all_words) + 1, x0 - margin_chars * char - PAD)
    else:
        left = x0 - PAD
    return int(max(0, left)), int(max(0, y0 - PAD)), int(min(W, x1 + PAD)), int(min(H, y1 + PAD))


def read_values(img, lines, reco):
    """{id(word): text read by `reco`} for every value word of `lines` in image `img`."""
    from .cord import value_words
    words = [w for line in lines for w in line["words"]]
    vw = [w for line in lines for w in value_words(line)]
    size = (img.shape[1], img.shape[0])
    crops = []
    for w in vw:
        x0, y0, x1, y1 = value_box(w, words, size)
        crops.append(img[y0:y1, x0:x1] if (x1 > x0 + 1 and y1 > y0 + 1) else np.full((8, 8, 3), 255, np.uint8))
    out = reco(crops) if crops else []
    return {id(w): t for w, (t, _) in zip(vw, out)}
