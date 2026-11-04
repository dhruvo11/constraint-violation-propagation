"""Build the synthetic transcript corpus: images + per-field ground truth, no annotation."""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import render, tamper  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SPLITS = {"train": (2000, 0), "val": (500, 1_000_000), "test": (1500, 2_000_000),
          "train_g1": (1000, 3_000_000)}  # blended edits from the older generator only
CAT_PROBS = [0.30, 0.20, 0.20, 0.15, 0.15]  # genuine, T1, T2, T3, T4


def make(args):
    split, i, seed = args
    rng = np.random.default_rng(seed)
    doc = render.generate(rng)
    if split == "train_g1":
        cat = ["T2", "T4"][rng.integers(2)]
        _, info = tamper.apply(doc, cat, rng, generator="g1")
    else:
        cat = tamper.CATEGORIES[rng.choice(5, p=CAT_PROBS)]
        _, info = tamper.apply(doc, cat, rng)
    name = f"{split}_{i:05d}.jpg"
    with open(os.path.join(ROOT, split, name), "wb") as fh:
        fh.write(info.pop("jpeg"))
    info["values"] = render.to_json(info["values"])
    return dict(id=name, split=split, orig_values=render.to_json(doc.values), orig_boxes=doc.boxes,
                style=render.style_dict(doc.style), content=doc.content.__dict__, **info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--splits", default=",".join(SPLITS))
    a = ap.parse_args()
    for split in a.splits.split(","):
        n, seed0 = SPLITS[split]
        os.makedirs(os.path.join(ROOT, split), exist_ok=True)
        jobs = [(split, i, seed0 + i) for i in range(int(n * a.scale))]
        with Pool(4) as pool:
            rows = pool.map(make, jobs, chunksize=25)
        with open(os.path.join(ROOT, f"{split}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        cats = {c: sum(r["category"] == c for r in rows) for c in tamper.CATEGORIES}
        print(split, len(rows), cats, flush=True)


if __name__ == "__main__":
    main()
