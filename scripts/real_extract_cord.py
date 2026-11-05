"""Run both CVP branches over genuine and forged real receipts (CORD v2) and cache what the
evaluation needs.

For every receipt of a split, each category of cvp/real/forge.py (genuine, T1-T4) is
rendered with a fixed seed per (receipt, category), then:
- the constraint branch reads it (cvp/real/read.py) and scores it with the receipt's own
  schema: violations as read, the consistent fields and the signature class's candidates;
- the visual branch computes its heatmap on the photo's visual view (cvp/real/view.py: scaled
  to a longer side of at most 1600 px, encoded once as JPEG, as in training), pooled over
  the printed words of every field.
The true violation pattern (of the forger's values) is kept for the ambiguity split.

Writes features/real/cord_<split><tag>.pkl.

  real_extract_cord.py --split test
  real_extract_cord.py --split test --visual models/visual_real.pt --tag _real
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import paths, visual  # noqa: E402
from cvp.real import cord, forge, read, view  # noqa: E402
from cvp.real import receipt_schema as RS  # noqa: E402


def pooled(H, boxes, fields, s, pad=3):
    out = np.zeros((len(fields), 3), np.float32)
    h, w = H.shape
    for i, f in enumerate(fields):
        if f not in boxes:
            continue
        x0, y0, x1, y1 = (int(round(c * s)) for c in boxes[f])
        reg = H[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
        if reg.size:
            out[i] = [reg.max(), reg.mean(), np.percentile(reg, 90)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=cord.SPLITS)
    ap.add_argument("--visual", default=visual.MODEL_PATH)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from doctr.models import recognition_predictor
    reco = recognition_predictor("crnn_vgg16_bn", pretrained=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = visual.ResUNet(pretrained=False)
    m.load_state_dict(torch.load(a.visual, map_location="cpu"))
    m = m.eval().to(dev)
    if dev == "mps":
        m = m.half()
    rows, t0 = [], time.time()
    for k, r in enumerate(cord.receipts(a.split, images=True)):
        if a.limit and k >= a.limit:
            break
        rg = cord.from_lines(r.lines, rid=r.id)
        S = RS.schema(rg)
        if not S.constraints:
            continue  # nothing CVP's constraint branch could say about this receipt
        words = forge.field_words(r)
        truth0 = RS.values(rg)
        for c, cat in enumerate(forge.CATEGORIES):
            img, info = forge.forge(r, cat, np.random.default_rng(1000 * k + c))
            if cat != "genuine" and not info["edited"]:
                continue
            text = read.read_values(img, r.lines, reco)
            vals = RS.values(cord.from_lines(r.lines, read=lambda wd: text[id(wd)]))
            w, res = S.evaluate(vals)
            w_true, _ = S.evaluate(dict(truth0, **info["new"]))
            small, s = view.visual_view(img)
            H = visual.heatmap(m, [small], dev)[0][0]
            boxes = {f: forge.bbox(words[f], (img.shape[1], img.shape[0])) for f in S.field_names if f in words}
            rows.append(dict(id=r.id, category=cat, fields=list(S.field_names), constraints=list(S.constraint_names),
                             w=w, r=res, w_true=w_true, edited=info["edited"],
                             consistent=S.consistent_fields(w), consistent_true=S.consistent_fields(w_true),
                             candidates=S.candidate_fields(S.signature_class(w), w),
                             pooled=pooled(H, boxes, S.field_names, s), hmax=float(H.max()),
                             classes={f: S.field_class[f] for f in S.field_names}))
        if k % 10 == 0:
            print(f"{a.split} {k + 1} receipts, {len(rows)} documents, {time.time() - t0:.0f}s", flush=True)
    out = os.path.join(paths.BASE, "features", "real")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"cord_{a.split}{a.tag}.pkl"), "wb") as fh:
        pickle.dump(rows, fh)
    print(f"done: {len(rows)} documents from {a.split} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
