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
from cvp import doctypes, ocr, paths, pipeline, visual  # noqa: E402
from cvp.core import render, tamper  # noqa: E402


def degrade(arr, cond):
    h, w = arr.shape[:2]
    if cond == "jpeg60":
        return render.jpeg(arr, 60)
    if cond == "jpeg75":
        return render.jpeg(arr, 75)
    if cond == "resize60":
        im = Image.fromarray(arr)
        return np.asarray(im.resize((round(w * 0.6), round(h * 0.6)), Image.BILINEAR).resize((w, h), Image.BILINEAR))
    return arr


class Docs(torch.utils.data.Dataset):
    def __init__(self, rows, root, cond):
        self.rows, self.root, self.cond = rows, root, cond

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        arr = np.asarray(Image.open(os.path.join(self.root, row["split"], row["id"])).convert("RGB"))
        arr = np.ascontiguousarray(degrade(arr, self.cond))
        return i, arr, visual.ela(arr)


def collate(b):
    return b


def gt_mask(row, shape):
    m = np.zeros(shape, bool)
    for r in row["rects"]:
        m[r[1]:r[3], r[0]:r[2]] = True
    return m


def run(dt, split, cond, reader, m, dev, tag="", limit=0, embed_only=False):
    s = dt.schema
    root = paths.data_dir(dt.name)
    with open(os.path.join(root, f"{split}.jsonl")) as fh:
        rows = [json.loads(l) for l in fh]
    if limit:
        rows = rows[:limit]
    n, F, C = len(rows), len(s.field_names), len(s.constraints)
    out = dict(cfeat=np.zeros((n, 2 * C + 2 * F), np.float32), w=np.zeros((n, C), np.float32),
               w_true=np.zeros((n, C), np.float32), pooled=np.zeros((n, F, 3), np.float32),
               ela=np.zeros((n, F, 2), np.float32), emb=np.zeros((n, 512), np.float32),
               boxes=np.zeros((n, F, 4), np.int32), ocr_ok=np.zeros((n, F), bool),
               vis_iou=np.full(n, np.nan, np.float32), vis_f1=np.full(n, np.nan, np.float32))
    loader = torch.utils.data.DataLoader(Docs(rows, root, cond), batch_size=4, num_workers=2, collate_fn=collate)
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
            o = reader.read(arrs[j], dt)
            feats, w, _ = pipeline.constraint_features(o, s)
            truth = dt.from_json(row["values"])
            out["cfeat"][i], out["w"][i] = feats, w
            out["w_true"][i] = s.evaluate(truth)[0]
            out["pooled"][i] = pipeline.pool(H[j], o["boxes"], s)
            out["ela"][i] = pipeline.pool(E, o["boxes"], s)[:, [1, 2]]
            out["emb"][i] = emb[j]
            out["boxes"][i] = [o["boxes"][f] for f in s.field_names]
            out["ocr_ok"][i] = [o["text"][f.name] == f.fmt(truth[f.name]) for f in s.fields]
            if row["rects"] and not embed_only:
                g, p = gt_mask(row, H[j].shape), H[j] > 0.5
                inter, union = (g & p).sum(), (g | p).sum()
                out["vis_iou"][i] = inter / max(union, 1)
                out["vis_f1"][i] = 2 * inter / max(g.sum() + p.sum(), 1)
        if idx[0] % 400 < 4:
            print(f"{dt.name}/{split}/{cond} {idx[-1] + 1}/{n} {time.time() - t0:.0f}s", flush=True)
    meta = dict(category=np.array([tamper.CATEGORIES.index(r["category"]) for r in rows]),
                primary=np.array([s.field_index[r["primary"]] if r["primary"] else -1 for r in rows]),
                edited=np.array([[f in r["edited"] for f in s.field_names] for r in rows]),
                rect=np.array([r["rects"][0] if len(r["rects"]) == 1 else [0, 0, 0, 0] for r in rows]),
                gt_box=np.array([r["boxes"][r["primary"]] if r["primary"] else [0, 0, 0, 0] for r in rows]))
    os.makedirs(paths.feature_dir(dt.name), exist_ok=True)
    np.savez_compressed(os.path.join(paths.feature_dir(dt.name), f"{split}_{cond}{tag}.npz"), **out, **meta)
    print(f"done {dt.name}/{split}/{cond}: ocr field acc {out['ocr_ok'].mean():.4f}, {time.time() - t0:.0f}s",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctype", default="transcript", choices=doctypes.names())
    ap.add_argument("--jobs", default="train:clean:embed,val:clean,val_g1:clean,test:clean,test:jpeg60:750,test:resize60:750",
                    help="split:condition[:limit][:embed], comma-separated")
    ap.add_argument("--visual", default=visual.MODEL_PATH)
    ap.add_argument("--ocr", default=ocr.MODEL_PATH)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dt = doctypes.get(a.doctype)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    reader = ocr.Reader(a.ocr, device="cpu")
    m = visual.ResUNet(pretrained=False)
    m.load_state_dict(torch.load(a.visual, map_location="cpu"))
    m = m.eval().to(dev)
    if dev == "mps":
        m = m.half()  # half precision: ~2x faster and half the memory on a memory-bound laptop
    for job in a.jobs.split(","):
        split, cond, *opt = job.split(":")
        limit = int(opt[0]) if opt and opt[0].isdigit() else 0
        run(dt, split, cond, reader, m, dev, a.tag, limit=limit, embed_only="embed" in opt)


if __name__ == "__main__":
    main()
