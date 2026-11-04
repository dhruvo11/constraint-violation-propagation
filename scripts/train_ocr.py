"""Train the slot OCR (CRNN + CTC) on crops cut from synthetic documents.

Field values are randomised beyond the valid range (so the recogniser learns no
language model that would 'correct' a forged value), documents pass through the
tampering renderers (so it sees every font in the taxonomy) and a third are degraded.
"""
import argparse
import datetime as dt
import os
import string
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import ocr, render, schema, tamper  # noqa: E402


def degrade(img, rng):
    u = rng.random()
    if u < 0.2:
        img = render.jpeg(img, int(rng.integers(40, 75)))
    elif u < 0.4:
        s = float(rng.uniform(0.5, 0.85))
        im = Image.fromarray(img)
        img = np.asarray(im.resize((int(640 * s), int(832 * s)), Image.BILINEAR).resize((640, 832), Image.BILINEAR))
    return img


def random_values(rng, v):
    v = dict(v)
    for m in schema.MARK_FIELDS:
        if rng.random() < 0.5:
            v[m] = int(rng.integers(0, 200))
    if rng.random() < 0.5:
        v["total"] = int(rng.integers(10, 1000))
    if rng.random() < 0.5:
        v["pct"] = round(float(rng.uniform(0, 100)), 2)
    if rng.random() < 0.5:
        v["grade"] = schema.GRADES[rng.integers(len(schema.GRADES))]
    if rng.random() < 0.5:
        v["date"] = dt.date(int(rng.integers(2005, 2031)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
    if rng.random() < 0.4:
        n = int(rng.integers(9, 15))
        v["roll"] = "".join(rng.choice(list(string.digits + string.ascii_uppercase), n))
    return v


def make(seed):
    rng = np.random.default_rng(seed)
    values, content, style = render.sample_document_spec(rng)
    doc = render.render(random_values(rng, values), content, style)
    img, info = tamper.apply(doc, tamper.CATEGORIES[rng.integers(5)], rng)
    img = degrade(img, rng)
    gray = img.astype(np.float32).mean(2)
    ox, oy = ocr.register(gray)
    out = []
    for f in schema.FIELDS:
        c, _ = ocr.crop_slot(gray, f, ox, oy)
        out.append(((c * 255).astype(np.uint8), schema.fmt(f, info["values"][f]), f))
    return out


def encode(texts):
    tgt = torch.tensor([ocr.CHAR_INDEX[c] for t in texts for c in t], dtype=torch.long)
    return tgt, torch.tensor([len(t) for t in texts], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    t0 = time.time()
    with Pool(4) as pool:
        rows = [r for doc in pool.map(make, range(10_000_000, 10_000_000 + args.docs), chunksize=50) for r in doc]
    X = torch.from_numpy(np.stack([r[0] for r in rows]))[:, None]  # uint8: the machine is memory-bound
    texts = [r[1] for r in rows]
    fields = [r[2] for r in rows]
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
        correct += ocr.decode(logits[i], fields[i])[0] == texts[i]
    print(f"held-out slot accuracy {correct / n_val:.4f} ({n_val} crops), total {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(ocr.MODEL_PATH), exist_ok=True)
    torch.save(model.cpu().state_dict(), ocr.MODEL_PATH)


if __name__ == "__main__":
    main()
