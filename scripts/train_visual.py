"""Train the visual branch on 256x256 patches with BCE + Dice against the edit masks.

Patches are cut on the 16-pixel JPEG MCU grid, so ELA computed on a patch equals ELA
computed on the full page; inference is fully convolutional on the whole page.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import schema, visual  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
P = 256


def load_rows(split):
    with open(os.path.join(ROOT, f"{split}.jsonl")) as fh:
        return [json.loads(l) for l in fh]


class PatchSet(torch.utils.data.Dataset):
    def __init__(self, rows, split, n):
        self.rows, self.split, self.n = rows, split, n

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,))))
        row = self.rows[rng.integers(len(self.rows))]
        arr = np.asarray(Image.open(os.path.join(ROOT, row["split"], row["id"])).convert("RGB"))
        h, w = arr.shape[:2]
        u = rng.random()
        if row["rects"] and u < 0.5:
            b = row["rects"][rng.integers(len(row["rects"]))]
        elif u < 0.85:
            b = row["boxes"][schema.FIELDS[rng.integers(len(schema.FIELDS))]]
        else:
            b = [rng.integers(w)] * 2 + [rng.integers(h)] * 2
            b = [b[0], b[2], b[1], b[3]]
        cx = (b[0] + b[2]) / 2 + rng.integers(-96, 97)
        cy = (b[1] + b[3]) / 2 + rng.integers(-96, 97)
        x0 = int(np.clip(cx - P / 2, 0, w - P)) // 16 * 16
        y0 = int(np.clip(cy - P / 2, 0, h - P)) // 16 * 16
        crop = np.ascontiguousarray(arr[y0:y0 + P, x0:x0 + P])
        mask = np.zeros((P, P), np.float32)
        for r in row["rects"]:
            mask[max(0, r[1] - y0):max(0, r[3] - y0), max(0, r[0] - x0):max(0, r[2] - x0)] = 1
        return visual.to_input(crop), torch.from_numpy(mask)[None]


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + 1
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1
    return (1 - num / den).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--exclude", default="", help="comma-separated categories to hold out from training")
    ap.add_argument("--extra", default="", help="additional training split, e.g. train_g1")
    ap.add_argument("--out", default=visual.MODEL_PATH)
    args = ap.parse_args()
    rows = [r for r in load_rows("train") if r["category"] not in args.exclude.split(",")]
    if args.extra:
        rows += load_rows(args.extra)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = visual.ResUNet(pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    loader = torch.utils.data.DataLoader(PatchSet(rows, "train", args.steps * args.batch), batch_size=args.batch,
                                         num_workers=args.workers, persistent_workers=True)
    model.train()
    t0 = time.time()
    for step, (x, m) in enumerate(loader):
        x, m = x.to(dev), m.to(dev)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, m) + dice_loss(logits, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.cpu().state_dict(), args.out)
    print("saved", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
