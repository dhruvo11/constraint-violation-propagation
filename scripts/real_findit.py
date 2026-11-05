"""Real human forgeries (Find it again!, ICDAR 2023): do they break receipt arithmetic?

Every scan is read end to end (docTR detection + recognition, words grouped into rows,
cvp/real/rows.py), its summary amounts paired with their keys, and the receipt rules
checked (cvp/real/findit.py). The OCR of each page is cached in
features/real/findit_ocr.pkl, so analysis re-runs instantly. The transcriptions the dataset
ships are used as a second reading (text order only, no positions).

Writes results/real/findit_rules.json.

  real_findit.py                 OCR every split (cached), then analyse
  real_findit.py --no-ocr        analyse the cache only
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import paths  # noqa: E402
from cvp.real import findit, rows  # noqa: E402

CACHE = os.path.join(paths.BASE, "features", "real", "findit_ocr.pkl")


def load_cache():
    return pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}


def ocr_all(order=("test", "val", "train")):
    from doctr.models import ocr_predictor
    cache = load_cache()
    model = ocr_predictor(pretrained=True)
    t0, done = time.time(), 0
    for split in order:
        for s in findit.scans(split):
            if s.id in cache:
                continue
            img = np.asarray(Image.open(s.image).convert("RGB"))
            page = model([img]).pages[0]
            cache[s.id] = rows.words_of(page, (img.shape[1], img.shape[0]))
            done += 1
            if done % 25 == 0:
                os.makedirs(os.path.dirname(CACHE), exist_ok=True)
                pickle.dump(cache, open(CACHE, "wb"))  # save as we go: an interruption loses little
                print(f"{split}: {done} scans read, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pickle.dump(cache, open(CACHE, "wb"))
    return cache


def analyse(cache):
    out = {}
    for source in ("ocr", "transcription"):
        st = {False: Counter(), True: Counter()}
        for split in findit.SPLITS:
            for s in findit.scans(split):
                if source == "ocr" and s.id not in cache:
                    continue
                v = rows.summary(rows.group_rows(cache[s.id])) if source == "ocr" else findit.amounts(s)
                res = findit.check(v)
                c = st[s.forged]
                c["receipts"] += 1
                if any(x is not None for x in res.values()):
                    c["checkable"] += 1
                    c["flagged"] += any(x is False for x in res.values())
                for k, x in res.items():
                    if x is not None:
                        c[f"{k} applies"] += 1
                        c[f"{k} holds"] += x
        out[source] = {("forged" if f else "genuine"): dict(st[f], flag_rate=st[f]["flagged"] / max(1, st[f]["checkable"]))
                       for f in (False, True)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()
    cache = load_cache() if a.no_ocr else ocr_all()
    res = analyse(cache)
    res["missing_from_archive"] = findit.MISSING
    os.makedirs(os.path.join(paths.BASE, "results", "real"), exist_ok=True)
    with open(os.path.join(paths.BASE, "results", "real", "findit_rules.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for source in ("ocr", "transcription"):
        for kind in ("genuine", "forged"):
            c = res[source][kind]
            print(f"{source:13s} {kind:7s}: {c.get('receipts', 0)} receipts, {c.get('checkable', 0)} checkable, "
                  f"flagged {100 * c['flag_rate']:.1f}% | " +
                  " ".join(f"{r} {c.get(r + ' holds', 0)}/{c.get(r + ' applies', 0)}" for r in ("total", "change", "tender", "rounding")))


if __name__ == "__main__":
    main()
