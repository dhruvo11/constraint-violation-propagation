"""Real AI forgeries: CVP on CORD receipts edited by commercial image models (AIForge-Doc).

AIForge-Doc (Wu et al., 2026) changes one field of a document with an AI image editor and
pastes only that field's box back into the photo: v1 used Gemini 2.5 Flash Image and Ideogram
v2 Edit, v2 uses GPT-Image-2 on the same specifications. Every forgery records the field, the
old and new text, the edited box and a pixel mask. For the 983-1,000 forgeries made from CORD
v2 receipts we already have the receipt's labelled words, so CVP runs on them exactly as on
our own forgeries (cvp/real/system.py): docTR reads each value at its labelled box, the
receipt's own rules are checked, and the heatmap picks among the fields the broken rules fit.

The authentic image of each forgery is the CORD photo itself, decoded to RGB; AIForge-Doc's
authentic images are that same decode and pixel-identical to the forgery outside the mask
(checked here for every forgery). No model is trained or tuned on AIForge-Doc.

  real_aiforge.py --version v2 --download     metadata, then the CORD-sourced images and masks
  real_aiforge.py --version v2                extract: OCR, rules and the heatmaps of the three
                                              visual branches -> features/real/aiforge_v2.pkl
  real_aiforge.py --version v2 --evaluate     metrics -> results/real/aiforge_v2.json
"""
import argparse
import io
import json
import os
import pickle
import re
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvp import paths  # noqa: E402
from cvp.real import cord, forge, money, read, view  # noqa: E402
from cvp.real import receipt_schema as RS  # noqa: E402

REPOS = {"v1": "Scam-AI/AIForge-Doc-v1", "v2": "Scam-AI/AIForge-Doc-v2"}
ROOT = paths.real_dir("aiforge")
FEATURES = os.path.join(paths.BASE, "features", "real")
RESULTS = os.path.join(paths.BASE, "results", "real")
THRESHOLDS = os.path.join(RESULTS, "cord_threshold.json")  # chosen on our CORD validation forgeries
# visual branches: synthetic only (zero-shot), and fine-tuned on our forgeries of CORD photos
MODELS = {"zero-shot": "visual_resunet.pt", "real-crude": "visual_cord_crude.pt", "real-all": "visual_cord_all.pt"}
FINE_TUNED = ("real-crude", "real-all")  # trained on CORD train photos: evaluated on CORD test receipts only


# ------------------------------------------------------------------ data
def local(version):
    return os.path.join(ROOT, version)


def metadata(version):
    rows = [json.loads(line) for line in open(os.path.join(local(version), "metadata.jsonl"))]
    return [r for r in rows if r.get("source_dataset") == "cord"]


def cord_index(row):
    """(split, k) of the CORD receipt a forgery was made from: AIForge-Doc numbers CORD v2
    train 0-799, validation 800-899, test 900-999."""
    for key in ("image_id", "source_index", "image", "authentic"):
        m = re.search(r"cord[_-]?(\d+)", str(row.get(key, "")))
        if m:
            n = int(m.group(1))
            break
    else:
        raise ValueError(f"no CORD index in {row}")
    return ("train", n) if n < 800 else ("validation", n - 800) if n < 900 else ("test", n - 900)


def download(version):
    from huggingface_hub import hf_hub_download, snapshot_download
    os.makedirs(local(version), exist_ok=True)
    hf_hub_download(REPOS[version], "metadata.jsonl", repo_type="dataset", local_dir=local(version))
    rows = metadata(version)
    want = sorted({r[k] for r in rows for k in ("image", "mask") if r.get(k)})
    print(f"{version}: {len(rows)} CORD forgeries; fetching {len(want)} files", flush=True)
    snapshot_download(REPOS[version], repo_type="dataset", local_dir=local(version), allow_patterns=want,
                      max_workers=4)
    missing = [p for p in want if not os.path.exists(os.path.join(local(version), p))]
    print(f"done; {len(missing)} missing")


def rgb(path_or_bytes):
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    return np.asarray(Image.open(src).convert("RGB"))


def edited_field(words, mask, size, fields):
    """The schema field whose printed words the edit mask covers most (None: a field no rule reads)."""
    best, best_px = None, 0
    for f in fields:
        if f not in words:
            continue
        x0, y0, x1, y1 = (int(round(c)) for c in forge.bbox(words[f], size, pad=0))
        px = int(mask[max(0, y0):y1 + 1, max(0, x0):x1 + 1].sum())
        if px > best_px:
            best, best_px = f, px
    return best


def new_value(field, text):
    return money.count(text) if field == "qty" or field.startswith("cnt") else money.amount(text)


# ------------------------------------------------------------------ extraction
def pooled(H, boxes, fields, s, pad=3):
    out = np.zeros((len(fields), 3), np.float32)
    h, w = H.shape
    for i, f in enumerate(fields):
        if f not in boxes:
            continue
        x0, y0, x1, y1 = (int(round(c * s)) for c in boxes[f])
        reg = H[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
        if reg.size:
            out[i] = [reg.max(), reg.mean(), np.percentile(reg, 90)]
    return out


def extract(version, limit=0):
    import torch
    from doctr.models import recognition_predictor
    from cvp import visual
    reco = recognition_predictor("crnn_vgg16_bn", pretrained=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    nets = {}
    for tag, name in MODELS.items():
        m = visual.ResUNet(pretrained=False)
        m.load_state_dict(torch.load(os.path.join(paths.MODELS, name), map_location="cpu"))
        m = m.eval().to(dev)
        nets[tag] = m.half() if dev == "mps" else m
    rows, done_genuine, t0 = [], set(), time.time()
    meta = metadata(version)
    for n, row in enumerate(meta):
        if limit and n >= limit:
            break
        split, k = cord_index(row)
        r = cord.receipt(split, k)
        rg = cord.from_lines(r.lines, rid=r.id)
        S = RS.schema(rg)
        words = forge.field_words(r)
        auth = rgb(r.image)
        fake = rgb(os.path.join(local(version), row["image"]))
        mask = rgb(os.path.join(local(version), row["mask"]))[..., 0] > 127
        aligned = fake.shape == auth.shape
        same_outside = float(np.mean(np.all(fake[~mask] == auth[~mask], axis=-1))) if aligned else 0.0
        size = (auth.shape[1], auth.shape[0])
        field = edited_field(words, mask, size, S.field_names) if aligned else None
        truth0 = RS.values(rg)
        nv = new_value(field, row.get("forged_value", "")) if field else None
        w_true, _ = S.evaluate(dict(truth0, **({field: nv} if field and nv is not None else {})))
        boxes = {f: forge.bbox(words[f], size) for f in S.field_names if f in words}
        docs = [("ai", fake)] + ([] if r.id in done_genuine else [("genuine", auth)])
        done_genuine.add(r.id)
        for cat, img in docs:
            text = read.read_values(img, r.lines, reco)
            vals = RS.values(cord.from_lines(r.lines, read=lambda wd: text[id(wd)]))
            w, res = S.evaluate(vals)
            small, s = view.visual_view(img)
            heat = {}
            for tag, net in nets.items():
                if tag in FINE_TUNED and split != "test":
                    continue
                H = visual.heatmap(net, [small], dev)[0][0]
                heat[tag] = dict(pooled=pooled(H, boxes, S.field_names, s), hmax=float(H.max()))
            rows.append(dict(id=r.id, split=split, spec=row.get("spec_id"), category=cat,
                             tool=row.get("assigned_tool"), cord_field=row.get("field_name"),
                             original=row.get("original_value"), forged=row.get("forged_value"),
                             edited=[field] if (cat == "ai" and field) else [], unconstrained=(cat == "ai" and not field),
                             aligned=aligned, same_outside=same_outside, fields=list(S.field_names),
                             n_rules=len(S.constraints), w=w, w_true=w_true if cat == "ai" else w,
                             consistent=S.consistent_fields(w), consistent_true=S.consistent_fields(w_true),
                             candidates=S.candidate_fields(S.signature_class(w), w),
                             read_value=vals.get(field) if (cat == "ai" and field) else None,
                             new_value=nv if cat == "ai" else None, heat=heat))
        if n % 25 == 0:
            print(f"{version}: {n + 1}/{len(meta)} forgeries, {len(rows)} documents, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(FEATURES, exist_ok=True)
    with open(os.path.join(FEATURES, f"aiforge_{version}.pkl"), "wb") as fh:
        pickle.dump(rows, fh)
    print(f"done: {len(rows)} documents in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ evaluation
def evaluate(version):
    from sklearn.metrics import roc_auc_score
    from real_evaluate_cord import correct
    rows = pickle.load(open(os.path.join(FEATURES, f"aiforge_{version}.pkl"), "rb"))
    thresholds = json.load(open(THRESHOLDS))
    ai = [d for d in rows if d["category"] == "ai"]
    out = dict(version=version, forgeries=len(ai), genuine=sum(d["category"] == "genuine" for d in rows),
               tools=sorted({d["tool"] for d in ai if d["tool"]}),
               aligned=float(np.mean([d["aligned"] for d in ai])),
               identical_outside_mask=float(np.mean([d["same_outside"] for d in ai])),
               rule_visible=float(np.mean([not d["unconstrained"] for d in ai])),
               breaks_rule_true=float(np.mean([bool(np.asarray(d["w_true"]).sum()) for d in ai if not d["unconstrained"]])),
               read_back=float(np.mean([d["read_value"] is not None and d["new_value"] is not None and
                                        abs(d["read_value"] - d["new_value"]) < 0.5 for d in ai if d["edited"]])),
               flagged_by_rules={c: float(np.mean([bool(np.asarray(d["w"]).sum()) for d in rows if d["category"] == c]))
                                 for c in ("genuine", "ai")},
               cord_fields={}, results={})
    for d in ai:
        out["cord_fields"][d["cord_field"]] = out["cord_fields"].get(d["cord_field"], 0) + 1
    for tag, name in MODELS.items():
        sel = [d for d in rows if tag in d["heat"]]
        if not sel:
            continue
        y = np.array([d["category"] == "ai" for d in sel])
        res = {}
        for method in ("Visual-only", "Constraint-only", "CVP"):
            det, acc = [], []
            for d in sel:
                heat = d["heat"][tag]["pooled"][:, 0]
                hmax = d["heat"][tag]["hmax"]
                violated = bool(np.asarray(d["w"]).sum())
                cons = d["consistent"] if violated else d["fields"]
                if method == "Visual-only":
                    det.append(hmax)
                    scores = list(heat)
                elif method == "Constraint-only":
                    det.append(float(np.asarray(d["w"]).sum()))
                    scores = [1.0 if f in cons else 0.0 for f in d["fields"]]
                else:
                    det.append(1.0 + hmax if violated else hmax)
                    scores = [heat[i] if f in cons else -1.0 for i, f in enumerate(d["fields"])] if violated else list(heat)
                acc.append(correct(scores, d["fields"], d["edited"]) if d["edited"] else np.nan)
            det, acc = np.array(det), np.array(acc)
            amb = np.array([d["category"] == "ai" and bool(d["edited"]) and len(d["consistent_true"]) > 1 for d in sel])
            vis = np.array([d["category"] == "genuine" or not d["unconstrained"] for d in sel])
            t = thresholds.get(name)
            r = dict(AUC=float(roc_auc_score(y, det)),
                     AUC_rule_fields=float(roc_auc_score(y[vis], det[vis])),
                     attr=float(np.nanmean(acc[y & ~np.isnan(acc)])) if (y & ~np.isnan(acc)).any() else float("nan"),
                     attr_amb=float(np.nanmean(acc[amb])) if amb.any() else float("nan"), n_amb=int(amb.sum()),
                     chance_amb=float(np.mean([1 / len(d["consistent_true"]) for d, a in zip(sel, amb) if a])) if amb.any() else float("nan"))
            if method == "CVP" and t is not None:
                flag = det >= t
                r["flagged_genuine"] = float(flag[~y].mean())
                r["flagged_ai"] = float(flag[y].mean())
            res[method] = r
        out["results"][tag] = dict(documents=len(sel), forgeries=int(y.sum()), methods=res)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"aiforge_{version}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))
    for tag, r in out["results"].items():
        print(f"{tag:10s} n={r['documents']:4d}", " | ".join(
            f"{m}: AUC {v['AUC']:.3f} attr {100 * v['attr']:.1f} amb {100 * v['attr_amb']:.1f}" for m, v in r["methods"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2", choices=sorted(REPOS))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.download:
        download(a.version)
    elif a.evaluate:
        evaluate(a.version)
    else:
        extract(a.version, a.limit)


if __name__ == "__main__":
    main()
