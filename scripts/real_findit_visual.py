"""Can the visual branch see real human forgeries? Find it again! val and test scans.

For each visual model (synthetic-only; fine-tuned on real CORD photos, crude edits only and
every category), every scan's heatmap is computed on its visual view (cvp/real/view.py).
- detection: AUC of the page's maximum heat, forged vs genuine (test);
- localisation: on forged scans, how often the hottest pixel lies in an annotated edited
  region (the "pointing game"; chance = the regions' share of the page);
- CVP on real forgeries: flag a scan when a receipt rule breaks as read (the cached OCR of
  scripts/real_findit.py) or its heat reaches a threshold chosen on val by Youden's J.
Writes results/real/findit_visual.json.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import paths, visual  # noqa: E402
from cvp.real import findit, rows, view  # noqa: E402

MODELS = ["visual_resunet.pt", "visual_cord_crude.pt", "visual_cord_all.pt"]
CACHE = os.path.join(paths.BASE, "features", "real", "findit_ocr.pkl")


def heat_stats(m, dev, s):
    img = np.asarray(Image.open(s.image).convert("RGB"))
    v, sc = view.visual_view(img)
    H = visual.heatmap(m, [v], dev)[0][0]
    y, x = np.unravel_index(int(np.argmax(H)), H.shape)
    mask = np.zeros(H.shape, bool)
    for r in s.edited:
        x0, y0, x1, y1 = view.scale_box([r["x0"], r["y0"], r["x1"], r["y1"]], sc)
        mask[max(0, y0):y1, max(0, x0):x1] = True
    return dict(hmax=float(H.max()), hit=bool(mask[y, x]) if s.edited else None,
                area=float(mask.mean()) if s.edited else None,
                inside=float(H[mask].max()) if mask.any() else None)


def youden_threshold(y, rule, h):
    grid = np.unique(np.concatenate([h, [h.max() + 1]]))
    def j(t):
        p = rule | (h >= t)
        return p[y].mean() - p[~y].mean()
    return max(grid, key=lambda t: (round(j(t), 9), t))


def main():
    ocr = pickle.load(open(CACHE, "rb"))
    scans = {sp: list(findit.scans(sp)) for sp in ("val", "test")}
    rule = {sp: np.array([any(x is False for x in findit.check(rows.summary(rows.group_rows(ocr[s.id]))).values())
                          if s.id in ocr else False for s in scans[sp]]) for sp in scans}
    y = {sp: np.array([s.forged for s in scans[sp]]) for sp in scans}
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out = {}
    for name in MODELS:
        path = os.path.join(paths.MODELS, name)
        if not os.path.exists(path):
            continue
        m = visual.ResUNet(pretrained=False)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m = m.eval().to(dev)
        if dev == "mps":
            m = m.half()
        t0 = time.time()
        st = {sp: [heat_stats(m, dev, s) for s in scans[sp]] for sp in scans}
        h = {sp: np.array([d["hmax"] for d in st[sp]]) for sp in scans}
        hits = [d for d in st["test"] if d["hit"] is not None]
        t = youden_threshold(y["val"], rule["val"], h["val"])
        flag = rule["test"] | (h["test"] >= t)
        out[name] = dict(AUC_heat=float(roc_auc_score(y["test"], h["test"])),
                         pointing=float(np.mean([d["hit"] for d in hits])), pointing_chance=float(np.mean([d["area"] for d in hits])),
                         threshold=float(t), flagged_genuine=float(flag[~y["test"]].mean()),
                         flagged_forged=float(flag[y["test"]].mean()),
                         rules_only=dict(genuine=float(rule["test"][~y["test"]].mean()), forged=float(rule["test"][y["test"]].mean())))
        r = out[name]
        print(f"{name}: heat AUC {r['AUC_heat']:.3f} | pointing {100 * r['pointing']:.1f}% (chance {100 * r['pointing_chance']:.1f}%) | "
              f"CVP at val threshold {t:.3f}: genuine {100 * r['flagged_genuine']:.1f}%, forged {100 * r['flagged_forged']:.1f}% "
              f"(rules alone: {100 * r['rules_only']['genuine']:.1f}% / {100 * r['rules_only']['forged']:.1f}%) [{time.time() - t0:.0f}s]", flush=True)
    out["test_scans"] = dict(genuine=int((~y["test"]).sum()), forged=int(y["test"].sum()))
    with open(os.path.join(paths.BASE, "results", "real", "findit_visual.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
