"""Choose CVP's visual-heat threshold for real receipts on validation documents (never test).

CVP flags a receipt when a rule is violated as read, or else when the visual branch's
maximum heat reaches the threshold. The threshold maximising Youden's J (detection rate
minus false-alarm rate) on CORD validation documents (scripts/real_extract_cord.py --split
validation) is stored per visual model in results/real/cord_threshold.json, which
cvp/real/system.py reads. J, unlike F1, cannot be won by flagging everything when most
documents are forged (four in five here); ties go to the higher threshold (fewer alarms).

  real_threshold.py --tag _crude --model visual_cord_crude.pt
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import paths  # noqa: E402


def decisions(rows, t):
    return np.array([bool(np.asarray(d["w"]).sum()) or d["hmax"] >= t for d in rows])


def f1(y, p):
    tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
    return 2 * tp / max(1, 2 * tp + fp + fn)


def youden(y, p):
    return p[y].mean() - p[~y].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", required=True, help="basename of the visual model these features came from")
    a = ap.parse_args()
    rows = pickle.load(open(os.path.join(paths.BASE, "features", "real", f"cord_validation{a.tag}.pkl"), "rb"))
    y = np.array([d["category"] != "genuine" for d in rows])
    grid = np.unique(np.concatenate([[d["hmax"] for d in rows], [1.01]]))
    best = max(grid, key=lambda t: (round(youden(y, decisions(rows, t)), 9), t))
    p = decisions(rows, best)
    path = os.path.join(paths.BASE, "results", "real", "cord_threshold.json")
    table = json.load(open(path)) if os.path.exists(path) else {}
    table[a.model] = float(best)
    with open(path, "w") as fh:
        json.dump(table, fh, indent=1)
    print(f"{a.model}: threshold {best:.4f} on {len(rows)} validation documents: F1 {f1(y, p):.3f}, "
          f"genuine flagged {100 * p[~y].mean():.1f}%, tampered flagged {100 * p[y].mean():.1f}%")


if __name__ == "__main__":
    main()
