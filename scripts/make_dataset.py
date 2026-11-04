"""Build a synthetic corpus for one document type: images + per-field ground truth,
obtained without annotation."""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import doctypes, paths  # noqa: E402
from cvp.core import render, tamper  # noqa: E402

SPLITS = {"train": (2000, 0), "val": (500, 1_000_000), "test": (1500, 2_000_000),
          "train_g1": (1000, 3_000_000)}  # blended edits from the older generator only
CAT_PROBS = [0.30, 0.20, 0.20, 0.15, 0.15]  # genuine, T1, T2, T3, T4


def make(job):
    name, split, i, seed = job
    dt = doctypes.get(name)
    rng = np.random.default_rng(seed)
    doc = render.generate(dt, rng)
    if split == "train_g1":
        cat = ["T2", "T4"][rng.integers(2)]
        _, info = tamper.apply(doc, dt, cat, rng, generator="g1")
    else:
        cat = tamper.CATEGORIES[rng.choice(5, p=CAT_PROBS)]
        _, info = tamper.apply(doc, dt, cat, rng)
    file = f"{split}_{i:05d}.jpg"
    with open(os.path.join(paths.data_dir(name), split, file), "wb") as fh:
        fh.write(info.pop("jpeg"))
    info["values"] = render.to_json(info["values"])
    return dict(id=file, split=split, orig_values=render.to_json(doc.values), orig_boxes=doc.boxes,
                style=render.style_dict(doc.style), content=doc.content, **info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctype", default="transcript", choices=doctypes.names())
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    root = paths.data_dir(a.doctype)
    for split in a.splits.split(","):
        n, seed0 = SPLITS[split]
        os.makedirs(os.path.join(root, split), exist_ok=True)
        jobs = [(a.doctype, split, i, seed0 + i) for i in range(int(n * a.scale))]
        with Pool(a.workers) as pool:
            rows = pool.map(make, jobs, chunksize=25)
        with open(os.path.join(root, f"{split}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        cats = {c: sum(r["category"] == c for r in rows) for c in tamper.CATEGORIES}
        print(a.doctype, split, len(rows), cats, flush=True)


if __name__ == "__main__":
    main()
