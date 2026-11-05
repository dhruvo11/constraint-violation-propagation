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
          "train_g1": (1000, 3_000_000),  # blended edits from the older generator only
          "val_g1": (500, 4_000_000)}  # the same, unseen in training: selects the attribution head
CAT_PROBS = [0.30, 0.20, 0.20, 0.15, 0.15]  # genuine, T1, T2, T3, T4


def make(job):
    name, split, i, seed = job
    dt = doctypes.get(name)
    rng = np.random.default_rng(seed)
    doc = render.generate(dt, rng)
    if split.endswith("_g1"):
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
    ap.add_argument("--doctypes", default=",".join(doctypes.names()), help="comma-separated document types")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="rebuild splits that already exist")
    a = ap.parse_args()
    for name in a.doctypes.split(","):
        root = paths.data_dir(name)
        for split in a.splits.split(","):
            n, seed0 = SPLITS[split]
            index = os.path.join(root, f"{split}.jsonl")
            if not a.force and os.path.exists(index) and sum(1 for _ in open(index)) == int(n * a.scale):
                print(name, split, "exists, skipped (use --force to rebuild)", flush=True)
                continue
            os.makedirs(os.path.join(root, split), exist_ok=True)
            jobs = [(name, split, i, seed0 + i) for i in range(int(n * a.scale))]
            with Pool(a.workers) as pool:
                rows = pool.map(make, jobs, chunksize=25)
            with open(os.path.join(root, f"{split}.jsonl"), "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            cats = {c: sum(r["category"] == c for r in rows) for c in tamper.CATEGORIES}
            print(name, split, len(rows), cats, flush=True)


if __name__ == "__main__":
    main()
