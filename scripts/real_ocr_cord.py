"""Read real receipts with a real OCR: every value word of CORD v2 is cropped where the
dataset's box puts it (field locations known, as with a template) and read by docTR's
recognizer; the receipt is rebuilt from what was read and checked against the rules.

Measures how often fields are read exactly, and how many genuine receipts real reading
errors add to the false alarms that real shops' arithmetic already causes.
Writes results/real/cord_ocr.json (and per-receipt readings, results/real/cord_ocr_reads.json).

  real_ocr_cord.py                     the test split
  real_ocr_cord.py --split validation
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import paths  # noqa: E402
from cvp.real import cord, read  # noqa: E402
from cvp.real import receipt_rules as RR  # noqa: E402

FIELDS = ["subtotal", "tax", "service", "discount", "total", "cash", "change", "card", "qty"]


def failed(res):
    return any(v is False or (isinstance(v, list) and not all(v)) for v in res.values())


def checked(res):
    return any(v is not None for v in res.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=cord.SPLITS)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from doctr.models import recognition_predictor
    reco = recognition_predictor("crnn_vgg16_bn", pretrained=True)
    t0 = time.time()
    words = fields = word_ok = 0
    field_ok = {f: [0, 0] for f in FIELDS + ["item price", "item count", "item unit price"]}
    gt_flag = ocr_flag = both = n_checked = 0
    reads = []
    for k, r in enumerate(cord.receipts(a.split, images=True)):
        if a.limit and k >= a.limit:
            break
        img = Image.open(io.BytesIO(r.image)).convert("RGB")
        vw = [w for line in r.lines for w in cord.value_words(line)]
        text = read.read_values(np.asarray(img), r.lines, reco)  # the shared reading policy
        words += len(vw)
        word_ok += sum(w["text"] == text[id(w)] for w in vw)
        gt = cord.from_lines(r.lines, rid=r.id, size=img.size)
        rd = cord.from_lines(r.lines, read=lambda w: text[id(w)], rid=r.id, size=img.size)
        for f in FIELDS:
            if getattr(gt, f) is not None:
                field_ok[f][1] += 1
                field_ok[f][0] += getattr(gt, f) == getattr(rd, f)
        for gi, ri in zip(gt.items, rd.items):
            for f, name in (("price", "item price"), ("cnt", "item count"), ("unitprice", "item unit price")):
                if getattr(gi, f) is not None:
                    field_ok[name][1] += 1
                    field_ok[name][0] += getattr(gi, f) == getattr(ri, f)
        rg, ro = RR.check(gt), RR.check(rd)
        if checked(rg):
            n_checked += 1
            gt_flag += failed(rg)
            ocr_flag += failed(ro)
            both += failed(rg) and failed(ro)
        reads.append(dict(id=r.id, words=[[w["text"], text[id(w)]] for w in vw], rules_gt=rg, rules_ocr=ro))
        if k % 20 == 0:
            print(f"{a.split} {k + 1} receipts, {time.time() - t0:.0f}s", flush=True)
    acc = {f: dict(correct=c, of=n, rate=c / n if n else None) for f, (c, n) in field_ok.items()}
    summary = dict(dataset="CORD v2", split=a.split, receipts=len(reads), ocr="docTR crnn_vgg16_bn on dataset word boxes",
                   word_exact=word_ok / max(words, 1), words=words, field_accuracy=acc, receipts_checked=n_checked,
                   flagged_true_values=gt_flag, flagged_as_read=ocr_flag, flagged_both=both,
                   false_alarm_true_values=gt_flag / max(n_checked, 1), false_alarm_as_read=ocr_flag / max(n_checked, 1))
    out = os.path.join(paths.BASE, "results", "real")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"cord_ocr_{a.split}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(out, f"cord_ocr_reads_{a.split}.json"), "w") as fh:
        json.dump(reads, fh)
    print(f"words read exactly: {100 * summary['word_exact']:.1f}% of {words}")
    for f, v in acc.items():
        if v["of"]:
            print(f"  {f:16s} {100 * v['rate']:5.1f}% of {v['of']}")
    print(f"genuine receipts flagged: true values {gt_flag}/{n_checked} = {100 * gt_flag / max(n_checked, 1):.1f}%, "
          f"as read by OCR {ocr_flag}/{n_checked} = {100 * ocr_flag / max(n_checked, 1):.1f}%  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
