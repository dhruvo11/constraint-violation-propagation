"""Fit g_phi and the CVP heads and save models/heads_<doctype>.pkl (what the app loads).

evaluate.py does this too as part of the full evaluation; this script only needs the
training and validation features, so the app can be brought up before the test-set
features exist.
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E  # noqa: E402
from cvp import doctypes, pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctype", default="transcript", choices=doctypes.names())
    ap.add_argument("--train", default="train")
    ap.add_argument("--val", default="val")
    ap.add_argument("--tag", default="", help="feature tag of the validation split, e.g. _blind")
    a = ap.parse_args()
    E.set_doctype(a.doctype)
    tr = E.without_blended(E.load(a.train, tag=a.tag if a.train != "train" else ""))
    va = E.without_blended(E.load(a.val, tag=a.tag))
    heads = E.Heads().fit(tr, va)
    th = E.best_thresholds(heads, va)
    with open(pipeline.heads_path(a.doctype), "wb") as fh:
        pickle.dump(dict(g_phi=heads.g, g_scaler=heads.scaler, det=heads.det, attr_w=heads.attr_w,
                         det_threshold=th["CVP"]), fh)
    print("saved", pipeline.heads_path(a.doctype), "theta", heads.attr_w.round(3), "threshold", round(th["CVP"], 3))


if __name__ == "__main__":
    main()
