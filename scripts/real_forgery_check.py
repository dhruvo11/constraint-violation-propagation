"""Are the real-receipt forgeries faithful? For every CORD receipt of a split, each category
of cvp/real/forge.py is rendered and read back with the shared reader (cvp/real/read.py):

- how often an edited value is read back exactly as the forger printed it;
- how often a coherent forgery (T3/T4) keeps every rule as read (by construction it keeps
  them on its true values: the rest is reading noise);
- how often a single-field forgery (T1/T2) breaks a rule as read;
- how often genuine receipts, re-encoded the same way, break a rule as read.

Writes results/real/cord_forgery_check_<split>.json.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import paths  # noqa: E402
from cvp.real import cord, forge, read  # noqa: E402
from cvp.real import receipt_schema as RS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=cord.SPLITS)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from doctr.models import recognition_predictor
    reco = recognition_predictor("crnn_vgg16_bn", pretrained=True)
    t0 = time.time()
    made, flagged, read_ok, read_n = Counter(), Counter(), 0, 0
    for k, r in enumerate(cord.receipts(a.split, images=True)):
        if a.limit and k >= a.limit:
            break
        S = RS.schema(cord.from_lines(r.lines))
        for c, cat in enumerate(forge.CATEGORIES):
            img, info = forge.forge(r, cat, np.random.default_rng(1000 * k + c))
            if cat != "genuine" and not info["edited"]:
                continue  # nothing this category could edit on this receipt
            made[cat] += 1
            text = read.read_values(img, r.lines, reco)
            vals = RS.values(cord.from_lines(r.lines, read=lambda w: text[id(w)]))
            for f in info["edited"]:
                read_n += 1
                read_ok += vals[f] == info["new"][f]
            w, _ = S.evaluate(vals)
            flagged[cat] += bool(S.violated(w))
        if k % 20 == 0:
            print(f"{a.split} {k + 1} receipts, {time.time() - t0:.0f}s", flush=True)
    rate = {c: flagged[c] / made[c] for c in made}
    out = dict(dataset="CORD v2", split=a.split, made=dict(made), edited_read_back=read_ok / max(1, read_n),
               edited_values=read_n, flagged_as_read=rate,
               coherent_keep_rules=1 - (flagged["T3"] + flagged["T4"]) / max(1, made["T3"] + made["T4"]))
    os.makedirs(os.path.join(paths.BASE, "results", "real"), exist_ok=True)
    with open(os.path.join(paths.BASE, "results", "real", f"cord_forgery_check_{a.split}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"made {dict(made)}; edited values read back exactly {100 * out['edited_read_back']:.1f}% of {read_n}")
    print("break a rule as read: " + ", ".join(f"{c} {100 * v:.1f}%" for c, v in rate.items()))
    print(f"coherent forgeries keep every rule as read: {100 * out['coherent_keep_rules']:.1f}%  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
