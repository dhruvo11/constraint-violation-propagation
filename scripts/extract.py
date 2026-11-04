"""Run OCR, constraint evaluation and the visual branch over a split; cache every
per-document quantity the heads, baselines and metrics need."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import ocr, pipeline, render, schema, tamper, visual  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE, "data")
FEAT = os.path.join(BASE, "features")


def degrade(arr, cond):
    if cond == "jpeg60":
        return render.jpeg(arr, 60)
    if cond == "jpeg75":
        return render.jpeg(arr, 75)
    if cond == "resize60":
        im = Image.fromarray(arr)
        return np.asarray(im.resize((384, 499), Image.BILINEAR).resize((640, 832), Image.BILINEAR))
    return arr


class Docs(torch.utils.data.Dataset):
    def __init__(self, rows, split, cond):
        self.rows, self.split, self.cond = rows, split, cond

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        arr = np.asarray(Image.open(os.path.join(ROOT, self.split, self.rows[i]["id"])).convert("RGB"))
        arr = np.ascontiguousarray(degrade(arr, self.cond))
        return i, arr, visual.ela(arr)


def collate(b):
    return b


def gt_mask(row, shape):
    m = np.zeros(shape, bool)
    for r in row["rects"]:
        m[r[1]:r[3], r[0]:r[2]] = True
    return m


def run(split, cond, reader, m, dev, tag="", limit=0, embed_only=False):
    with open(os.path.join(ROOT, f"{split}.jsonl")) as fh:
        rows = [json.loads(l) for l in fh]
    if limit:
        rows = rows[:limit]
    n, F = len(rows), len(schema.FIELDS)
    out = dict(cfeat=np.zeros((n, 40), np.float32), w=np.zeros((n, 10), np.float32),
               w_true=np.zeros((n, 10), np.float32), pooled=np.zeros((n, F, 3), np.float32),
               ela=np.zeros((n, F, 2), np.float32), emb=np.zeros((n, 512), np.float32),
               boxes=np.zeros((n, F, 4), np.int32), ocr_ok=np.zeros((n, F), bool),
               vis_iou=np.full(n, np.nan, np.float32), vis_f1=np.full(n, np.nan, np.float32))
    loader = torch.utils.data.DataLoader(Docs(rows, split, cond), batch_size=4, num_workers=2,
                                         collate_fn=collate)
    t0 = time.time()
    for batch in loader:
        idx = [b[0] for b in batch]
        arrs = [b[1] for b in batch]
        if embed_only:  # training split: g_phi and the concatenation baseline need no heatmap
            emb = visual.embed(m, arrs, dev)
            H = np.zeros((len(arrs),) + arrs[0].shape[:2], np.float32)
        else:
            H, emb = visual.heatmap(m, arrs, dev)
        for j, i in enumerate(idx):
            row, E = rows[i], batch[j][2]
            o = reader.read(arrs[j])
            feats, w, _ = pipeline.constraint_features(o)
            truth = render.from_json(row["values"])
            out["cfeat"][i], out["w"][i] = feats, w
            out["w_true"][i] = schema.evaluate(truth)[0]
            out["pooled"][i] = pipeline.pool(H[j], o["boxes"])
            out["ela"][i] = pipeline.pool(E, o["boxes"])[:, [1, 2]]
            out["emb"][i] = emb[j]
            out["boxes"][i] = [o["boxes"][f] for f in schema.FIELDS]
            out["ocr_ok"][i] = [o["text"][f] == schema.fmt(f, truth[f]) for f in schema.FIELDS]
            if row["rects"] and not embed_only:
                g, p = gt_mask(row, H[j].shape), H[j] > 0.5
                inter, union = (g & p).sum(), (g | p).sum()
                out["vis_iou"][i] = inter / max(union, 1)
                out["vis_f1"][i] = 2 * inter / max(g.sum() + p.sum(), 1)
        if idx[0] % 400 < 4:
            print(f"{split}/{cond} {idx[-1] + 1}/{n} {time.time() - t0:.0f}s", flush=True)
    meta = dict(category=np.array([tamper.CATEGORIES.index(r["category"]) for r in rows]),
                primary=np.array([schema.FIELDS.index(r["primary"]) if r["primary"] else -1 for r in rows]),
                edited=np.array([[f in r["edited"] for f in schema.FIELDS] for r in rows]),
                oor=np.array([r["mark_out_of_range"] for r in rows]),
                rect=np.array([r["rects"][0] if len(r["rects"]) == 1 else [0, 0, 0, 0] for r in rows]),
                gt_box=np.array([r["boxes"][r["primary"]] if r["primary"] else [0, 0, 0, 0] for r in rows]))
    os.makedirs(FEAT, exist_ok=True)
    np.savez_compressed(os.path.join(FEAT, f"{split}_{cond}{tag}.npz"), **out, **meta)
    print(f"done {split}/{cond}: ocr field acc {out['ocr_ok'].mean():.4f}, {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="train:clean:embed,val:clean,test:clean,test:jpeg60:750,test:resize60:750",
                    help="split:condition[:limit][:embed], comma-separated")
    ap.add_argument("--visual", default=visual.MODEL_PATH)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    reader = ocr.Reader(device="cpu")
    m = visual.ResUNet(pretrained=False)
    m.load_state_dict(torch.load(a.visual, map_location="cpu"))
    m = m.eval().to(dev)
    if dev == "mps":
        m = m.half()  # half precision: ~2x faster and half the memory on a memory-bound laptop
    for job in a.jobs.split(","):
        split, cond, *opt = job.split(":")
        limit = int(opt[0]) if opt and opt[0].isdigit() else 0
        run(split, cond, reader, m, dev, a.tag, limit=limit, embed_only="embed" in opt)


if __name__ == "__main__":
    main()
