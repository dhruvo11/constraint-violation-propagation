"""Train the slot OCR (CRNN + CTC) on crops cut from synthetic documents.

Field values are randomised beyond the valid range (so the recogniser learns no
language model that would 'correct' a forged value), documents pass through the
tampering renderers (so it sees every font in the taxonomy) and a third are degraded.
One recogniser serves every document type: crops from all requested types are pooled.
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
    ox, oy = ocr.register(gray, dt.header_rule)
    out = []
    for f in dt.schema.fields:
        c, _ = ocr.crop_slot(gray, dt, f.name, ox, oy)
        out.append(((c * 255).astype(np.uint8), f.fmt(info["values"][f.name]), f.charset))
    return out


def encode(texts):
    tgt = torch.tensor([ocr.CHAR_INDEX[c] for t in texts for c in t], dtype=torch.long)
    return tgt, torch.tensor([len(t) for t in texts], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctypes", default=",".join(doctypes.names()))
    ap.add_argument("--docs", type=int, default=4000, help="documents per type")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--out", default=ocr.MODEL_PATH)
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(name, 10_000_000 + k * 1_000_000 + i) for k, name in enumerate(args.doctypes.split(","))
            for i in range(args.docs)]
    with Pool(4) as pool:
        rows = [r for doc in pool.map(make, jobs, chunksize=50) for r in doc]
    X = torch.from_numpy(np.stack([r[0] for r in rows]))[:, None]  # uint8: the machine is memory-bound
    texts = [r[1] for r in rows]
    charsets = [r[2] for r in rows]
    del rows
    n_val = len(texts) // 20
    print(f"{len(texts)} crops in {time.time() - t0:.0f}s", flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ocr.CRNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.steps, pct_start=0.1)
    ctc = nn.CTCLoss(zero_infinity=True)
    rng = np.random.default_rng(0)
    for step in range(args.steps):
        idx = rng.integers(n_val, len(texts), 128)
        x = X[idx].to(dev).float().div(255)
        tgt, lens = encode([texts[i] for i in idx])
        logp = model(x).log_softmax(-1).transpose(0, 1).cpu()  # T,B,C
        loss = ctc(logp, tgt, torch.full((len(idx),), logp.shape[0], dtype=torch.long), lens)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)

    model.eval()
    correct = 0
    with torch.no_grad():
        logits = torch.cat([model(X[i:i + 512].to(dev).float().div(255)).float().cpu()
                            for i in range(0, n_val, 512)])[:n_val]
    for i in range(n_val):
        correct += ocr.decode(logits[i], charsets[i])[0] == texts[i]
    print(f"held-out slot accuracy {correct / n_val:.4f} ({n_val} crops), total {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.cpu().state_dict(), args.out)


if __name__ == "__main__":
    main()
