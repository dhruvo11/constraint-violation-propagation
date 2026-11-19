"""How well the app's experimental reader (cvp/real/upload.py) reads real receipt photos that it
is given without their labels: CORD v2 test receipts, genuine and with our smooth single-field
edit (T2, the evaluation's seeds). For each photo: which summary values it found and read right
(against CORD's labels), whether a bill-level rule breaks (an alarm: false on a genuine photo, a
catch on a forged one) or only an item-level rule (a warning), and, on forgeries, whether the edited field is among the fields the broken rules fit.

  real_upload_check.py [--limit N]   -> results/real/upload_check.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import paths  # noqa: E402
from cvp.real import cord, forge, upload  # noqa: E402
from cvp.real import receipt_schema as RS  # noqa: E402
from cvp.real.receipt_rules import close  # noqa: E402

FIELDS = ("subtotal", "tax", "service", "discount", "total", "cash", "change")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from doctr.models import ocr_predictor
    predictor = ocr_predictor(pretrained=True)
    out, t0 = [], time.time()
    for k, r in enumerate(cord.receipts("test", images=True)):
        if a.limit and k >= a.limit:
            break
        truth = RS.values(cord.from_lines(r.lines, rid=r.id))
        for cat in ("genuine", "T2"):
            img, info = forge.forge(r, cat, np.random.default_rng(1000 * k + forge.CATEGORIES.index(cat)))
            if cat != "genuine" and not info["edited"]:
                continue
            true_now = dict(truth, **info["new"])
            rd = upload.read(img, predictor)
            got = RS.values(rd.receipt)
            res = upload.analyse(rd, np.zeros((8, 8), np.float32), 1.0)
            fields = {f: ("missing" if got.get(f) is None else
                          "right" if true_now.get(f) is not None and close(got[f], true_now[f]) else "wrong")
                      for f in FIELDS if true_now.get(f) is not None}
            n_items = sum(1 for f in got if f.startswith("price") and got[f] is not None)
            # a forgery is caught by the rules when one breaks; its edit is located when the edited
            # summary field is among the fields the broken rules fit (item prices are numbered differently)
            edited = info["edited"][0] if info["edited"] else None
            out.append(dict(k=k, category=cat, edited=edited, fields=fields, n_items=n_items,
                            true_items=len(r.items), violated=res["violated"], alarm=res["tampered"],
                            warning=res["warning"], checked=res["checked"], checked_hard=res["checked_hard"],
                            located=bool(edited in res["consistent"]) if edited in FIELDS else None,
                            notes=rd.notes))
        if k % 10 == 0:
            print(f"{k + 1} receipts, {len(out)} photos, {time.time() - t0:.0f}s", flush=True)
    gen = [d for d in out if d["category"] == "genuine"]
    fake = [d for d in out if d["category"] != "genuine"]
    per_field = {f: {s: sum(d["fields"].get(f) == s for d in gen) for s in ("right", "wrong", "missing")}
                 for f in FIELDS}
    summary = dict(photos=len(out), genuine=len(gen), forged=len(fake),
                   false_alarm=float(np.mean([d["alarm"] for d in gen])),
                   warning_genuine=float(np.mean([d["warning"] for d in gen])),
                   no_rule_checked=float(np.mean([d["checked"] == 0 for d in gen])),
                   no_bill_rule_checked=float(np.mean([d["checked_hard"] == 0 for d in gen])),
                   caught=float(np.mean([d["alarm"] for d in fake])),
                   caught_or_warned=float(np.mean([d["alarm"] or d["warning"] for d in fake])),
                   located=float(np.mean([d["located"] for d in fake if d["located"] is not None])),
                   located_n=sum(d["located"] is not None for d in fake), per_field_genuine=per_field)
    os.makedirs(os.path.join(paths.BASE, "results", "real"), exist_ok=True)
    with open(os.path.join(paths.BASE, "results", "real", "upload_check.json"), "w") as fh:
        json.dump(dict(summary=summary, photos=out), fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
