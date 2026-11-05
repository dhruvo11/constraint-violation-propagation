"""Training data for the visual branch on real photos: every CORD v2 receipt of a split is
forged in each category of cvp/real/forge.py (genuine, T1-T4) and saved as the visual
branch's view of it (cvp/real/view.py), with the edited regions and the field boxes scaled
to that view. Rows follow the synthetic corpus's format, so scripts/train_visual.py reads
them as a document type named "cord" (data/cord/<split>.jsonl, images in data/cord/<split>/).

Seeds differ from the test extraction's (scripts/real_extract_cord.py), and only train and
validation receipts are used, so no test receipt is ever seen in training.

  make_real_dataset.py                  train and validation
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
from cvp.real import cord, forge, view  # noqa: E402

OUT = paths.data_dir("cord")
PATCH = 256  # train_visual.py cuts 256x256 patches: narrower views are padded to that (view.py)
SEED = {"train": 5_000_000, "validation": 6_000_000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,validation")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for split in a.splits.split(","):
        os.makedirs(os.path.join(OUT, split), exist_ok=True)
        rows, cats, t0 = [], Counter(), time.time()
        for k, r in enumerate(cord.receipts(split, images=True)):
            if a.limit and k >= a.limit:
                break
            words = forge.field_words(r)
            for c, cat in enumerate(forge.CATEGORIES):
                img, info = forge.forge(r, cat, np.random.default_rng(SEED[split] + 1000 * k + c))
                if cat != "genuine" and not info["edited"]:
                    continue
                _, s, data = view.visual_view(img, with_bytes=True, min_side=PATCH)
                name = f"{r.id}_{cat}.jpg"
                with open(os.path.join(OUT, split, name), "wb") as fh:
                    fh.write(data)
                boxes = {f: view.scale_box(forge.bbox(ws, (img.shape[1], img.shape[0])), s) for f, ws in words.items()}
                rows.append(dict(id=name, split=split, category=info["category"], edited=info["edited"],
                                 rects=[view.scale_box(b, s) for b in info["rects"]], boxes=boxes))
                cats[info["category"]] += 1
            if k % 50 == 0:
                print(f"{split} {k + 1} receipts, {len(rows)} images, {time.time() - t0:.0f}s", flush=True)
        with open(os.path.join(OUT, f"{split}.jsonl"), "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"{split}: {len(rows)} images {dict(cats)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
