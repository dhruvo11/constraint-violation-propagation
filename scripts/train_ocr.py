"""Train the slot OCR (CRNN + CTC) on crops cut from synthetic documents.

Field values are randomised beyond the valid range (so the recogniser learns no
language model that would 'correct' a forged value), documents pass through the
tampering renderers (so it sees every font in the taxonomy) and a third are degraded.
One recogniser serves every document type: crops from all requested types are pooled.
Long lines (machine-readable zones, amounts in words) are read from wider crops; the
network is the same (fully convolutional + BiLSTM), only the sequence is longer, so
normal and wide crops are trained in separate batches of the same pixel count.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import doctypes, ocr  # noqa: E402
from cvp.core import render, tamper  # noqa: E402

EXTRA_DOCS = {"passport": 2.0}  # two 44-character MRZ lines per page need more examples


def degrade(img, rng):
    u = rng.random()
    if u < 0.2:
        img = render.jpeg(img, int(rng.integers(40, 75)))
    elif u < 0.4:
        s = float(rng.uniform(0.5, 0.85))
        h, w = img.shape[:2]
        im = Image.fromarray(img)
        img = np.asarray(im.resize((int(w * s), int(h * s)), Image.BILINEAR).resize((w, h), Image.BILINEAR))
    return img


def make(job):
    name, seed = job
    dt = doctypes.get(name)
    rng = np.random.default_rng(seed)
    values, content, style = dt.sample_spec(rng)
    doc = render.render(dt, dt.random_values(rng, values), content, style)
    img, info = tamper.apply(doc, dt, tamper.CATEGORIES[rng.integers(5)], rng)
    img = degrade(img, rng)
    gray = img.astype(np.float32).mean(2)
    ox, oy = ocr.register(gray, dt.header_rule, dt.rule_min)
    out = []
    for f in dt.schema.fields:
        c, _ = ocr.crop_slot(gray, dt, f.name, ox, oy)
        out.append(((c * 255).astype(np.uint8), f.fmt(info["values"][f.name]), f.charset, name))
    return out


def encode(texts):
    tgt = torch.tensor([ocr.CHAR_INDEX[c] for t in texts for c in t], dtype=torch.long)
    return tgt, torch.tensor([len(t) for t in texts], dtype=torch.long)


class Pool_:
    """Crops of one width: images (uint8, the machine is memory-bound), labels, charsets."""

    def __init__(self, rows, n_val_frac=0.05):
        self.X = torch.from_numpy(np.stack([r[0] for r in rows]))[:, None]
        self.texts = [r[1] for r in rows]
        self.charsets = [r[2] for r in rows]
        self.types = [r[3] for r in rows]
        self.n_val = max(1, int(len(rows) * n_val_frac))

    def __len__(self):
        return len(self.texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctypes", default=",".join(doctypes.names()))
    ap.add_argument("--docs", type=int, default=1500, help="documents per type")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--wide-share", type=float, default=0.3, help="fraction of steps spent on long lines")
    ap.add_argument("--out", default=ocr.MODEL_PATH)
    args = ap.parse_args()
    t0 = time.time()
    names = args.doctypes.split(",")
    jobs = [(name, 10_000_000 + k * 1_000_000 + i) for k, name in enumerate(names)
            for i in range(int(args.docs * EXTRA_DOCS.get(name, 1.0)))]
    rows = {ocr.CROP_W: [], ocr.WIDE_W: []}
    with Pool(4) as pool:
        for doc in pool.imap_unordered(make, jobs, chunksize=50):
            for r in doc:
                rows[r[0].shape[1]].append(r)
    rng = np.random.default_rng(0)
    pools = {}
    for width, rs in rows.items():
        if rs:
            order = rng.permutation(len(rs))  # validation crops are a random 5% of each width
            pools[width] = Pool_([rs[i] for i in order])
    del rows
    print(" ".join(f"{len(p)} crops of width {w}" for w, p in pools.items()), f"in {time.time() - t0:.0f}s", flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ocr.CRNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.steps, pct_start=0.1)
    ctc = nn.CTCLoss(zero_infinity=True)
    batch = {ocr.CROP_W: 128, ocr.WIDE_W: 32}
    for step in range(args.steps):
        width = ocr.WIDE_W if ocr.WIDE_W in pools and rng.random() < args.wide_share else ocr.CROP_W
        p = pools[width]
        idx = rng.integers(p.n_val, len(p), batch[width])
        x = p.X[idx].to(dev).float().div(255)
        tgt, lens = encode([p.texts[i] for i in idx])
        logp = model(x).log_softmax(-1).transpose(0, 1).cpu()  # T,B,C
        loss = ctc(logp, tgt, torch.full((len(idx),), logp.shape[0], dtype=torch.long), lens)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0:
            print(f"step {step} width {width} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    model.eval()
    for width, p in pools.items():
        with torch.no_grad():
            logits = torch.cat([model(p.X[i:min(i + 256, p.n_val)].to(dev).float().div(255)).float().cpu()
                                for i in range(0, p.n_val, 256)])
        ok = np.array([ocr.decode(logits[i], p.charsets[i])[0] == p.texts[i] for i in range(p.n_val)])
        types = np.array(p.types[:p.n_val])
        per = ", ".join(f"{t} {ok[types == t].mean():.4f}" for t in names if (types == t).any())
        print(f"held-out accuracy, width {width}: {ok.mean():.4f} ({p.n_val} crops; {per})", flush=True)
    print(f"total {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.cpu().state_dict(), args.out)


if __name__ == "__main__":
    main()
