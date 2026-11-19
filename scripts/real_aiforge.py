"""Real AI forgeries: CVP on CORD receipts edited by commercial image models (AIForge-Doc).

AIForge-Doc (Wu et al., 2026) changes one field of a document with an AI image editor and
pastes only that field's box back into the photo: v1 used Gemini 2.5 Flash Image and Ideogram
v2 Edit, v2 uses GPT-Image-2 on the same specifications. Every forgery records the field, the
old and new text, the edited box and a pixel mask. For the 983-1,000 forgeries made from CORD
v2 receipts we already have the receipt's labelled words, so CVP runs on them exactly as on
our own forgeries (cvp/real/system.py): docTR reads each value at its labelled box, the
receipt's own rules are checked, and the heatmap picks among the fields the broken rules fit.

The authentic image of each forgery is the forgery with its masked region restored from the
CORD photo, so the pair differs only where the AI edited it (AIForge-Doc's own authentic images
are the CORD photo's decode, pixel-identical to the forgery outside the mask; the share of
identical pixels is recorded). No model is trained or tuned on AIForge-Doc.

  real_aiforge.py --version v2 --download     metadata, then the CORD-sourced images and masks
  real_aiforge.py --version v2                extract: OCR, rules and the heatmaps of the three
                                              visual branches -> features/real/aiforge_v2.pkl
  real_aiforge.py --version v2 --evaluate     metrics -> results/real/aiforge_v2.json
  real_aiforge.py --version v2 --jpeg --split test [--evaluate]
                                              the same with both images first saved once more as
                                              JPEG (quality 90), as every document of our own CORD
                                              forgeries was: the fine-tuned branches and the alarm
                                              thresholds were built on that compression history
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
from cvp.core import render  # noqa: E402
from cvp.real import cord, forge, money, read, view  # noqa: E402
from cvp.real import receipt_schema as RS  # noqa: E402
from cvp.real.aiforge import (REPOS, ROOT, TEXT_FIELDS, cord_index, edited_field, local,  # noqa: E402,F401
                              metadata, new_value, rgb)

FEATURES = os.path.join(paths.BASE, "features", "real")
RESULTS = os.path.join(paths.BASE, "results", "real")
THRESHOLDS = os.path.join(RESULTS, "cord_threshold.json")  # chosen on our CORD validation forgeries
# visual branches: synthetic only (zero-shot), and fine-tuned on our forgeries of CORD photos
MODELS = {"zero-shot": "visual_resunet.pt", "real-crude": "visual_cord_crude.pt", "real-all": "visual_cord_all.pt"}
FINE_TUNED = ("real-crude", "real-all")  # trained on CORD train photos: evaluated on CORD test receipts only


# ------------------------------------------------------------------ data
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


def extract(version, limit=0, jpeg=False, split=None):
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
    if split:
        meta = [r for r in meta if cord_index(r)[0] == split]
    failures = []
    for n, row in enumerate(meta):
        if limit and n >= limit:
            break
        try:
            _extract_one(row, n, len(meta), version, reco, nets, dev, rows, done_genuine, t0, jpeg)
        except Exception as e:  # a corrupt or unexpected file must not stop the run: count it
            failures.append((row.get("spec_id"), f"{type(e).__name__}: {e}"))
    os.makedirs(FEATURES, exist_ok=True)
    with open(os.path.join(FEATURES, f"aiforge_{version}{suffix(jpeg)}.pkl"), "wb") as fh:
        pickle.dump(rows, fh)
    print(f"done: {len(rows)} documents in {time.time() - t0:.0f}s; {len(failures)} forgeries skipped")
    for f in failures[:20]:
        print("  skipped", f)


def suffix(jpeg):
    return "_jpeg" if jpeg else ""


def _extract_one(row, n, total, version, reco, nets, dev, rows, done_genuine, t0, jpeg=False):
    from cvp import visual
    if True:
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
        if aligned:
            # the paired authentic image: the forgery with the edited region restored from the CORD
            # photo, so the two differ only inside the mask (even if two JPEG decoders differ by 1)
            auth = fake.copy()
            auth[mask] = rgb(r.image)[mask]
        if jpeg:  # one more JPEG save of both, as forge.forge gives every document of our own forgeries
            fake, auth = render.jpeg(fake, forge.QUALITY), render.jpeg(auth, forge.QUALITY)
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
            print(f"{version}: {n + 1}/{total} forgeries, {len(rows)} documents, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ evaluation
def scores(d, tag, method):
    """(detection score, field scores) of one document, as in real_evaluate_cord.score_row."""
    heat = d["heat"][tag]["pooled"][:, 0]
    hmax = d["heat"][tag]["hmax"]
    violated = bool(np.asarray(d["w"]).sum())
    cons = d["consistent"] if violated else d["fields"]
    if method == "Visual-only":
        return hmax, list(heat)
    if method == "Constraint-only":
        return float(np.asarray(d["w"]).sum()), [1.0 if f in cons else 0.0 for f in d["fields"]]
    # CVP: noisy-OR detection (a violation outranks any heat); the heat picks among the consistent fields
    return (1.0 + hmax if violated else hmax), \
        ([heat[i] if f in cons else -1.0 for i, f in enumerate(d["fields"])] if violated else list(heat))


def alarm(d, tag, t):
    """CVP's alarm at threshold t, as in real_evaluate_cord: a broken rule, or else heat of at least t
    (a threshold above 1 means that only the rules raise alarms)."""
    return bool(np.asarray(d["w"]).sum()) or d["heat"][tag]["hmax"] >= t


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def evaluate(version, jpeg=False, boot=1000):
    from sklearn.metrics import roc_auc_score
    from real_evaluate_cord import correct
    rows = pickle.load(open(os.path.join(FEATURES, f"aiforge_{version}{suffix(jpeg)}.pkl"), "rb"))
    for d in rows:
        if d["category"] == "ai" and d["cord_field"] in TEXT_FIELDS:
            d.update(edited=[], unconstrained=True, read_value=None, new_value=None)
    thresholds = json.load(open(THRESHOLDS))
    ai = [d for d in rows if d["category"] == "ai"]
    edited = [d for d in ai if d["edited"]]
    read_back = lambda ds: float(np.mean([d["read_value"] is not None and d["new_value"] is not None and  # noqa: E731
                                          abs(d["read_value"] - d["new_value"]) < 0.5 for d in ds]))
    out = dict(version=version, jpeg=jpeg, forgeries=len(ai), genuine=sum(d["category"] == "genuine" for d in rows),
               receipts=len({d["id"] for d in rows}),
               tools={t: sum(d["tool"] == t for d in ai) for t in sorted({d["tool"] for d in ai if d["tool"]})},
               aligned=float(np.mean([d["aligned"] for d in ai])),
               identical_outside_mask=float(np.mean([d["same_outside"] for d in ai])),
               rule_visible=float(np.mean([not d["unconstrained"] for d in ai])),
               breaks_rule_true=float(np.mean([bool(np.asarray(d["w_true"]).sum()) for d in ai if not d["unconstrained"]])),
               read_back=read_back(edited),
               read_back_by_tool={t: read_back([d for d in edited if d["tool"] == t]) for t in {d["tool"] for d in edited}},
               flagged_by_rules={c: float(np.mean([bool(np.asarray(d["w"]).sum()) for d in rows if d["category"] == c]))
                                 for c in ("genuine", "ai")},
               cord_fields={}, results={})
    for d in ai:
        out["cord_fields"][d["cord_field"]] = out["cord_fields"].get(d["cord_field"], 0) + 1
    rng = np.random.default_rng(0)
    for tag, name in MODELS.items():
        sel = [d for d in rows if tag in d["heat"]]
        if not sel:
            continue
        y = np.array([d["category"] == "ai" for d in sel])
        amb = np.array([d["category"] == "ai" and bool(d["edited"]) and len(d["consistent_true"]) > 1 for d in sel])
        vis = np.array([d["category"] == "genuine" or not d["unconstrained"] for d in sel])
        tools = np.array([d["tool"] or "" for d in sel])
        # bootstrap over receipts: each receipt gives one forgery and its authentic image
        ids = np.array([d["id"] for d in sel])
        groups = [np.flatnonzero(ids == u) for u in np.unique(ids)]
        boots = [np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))]) for _ in range(boot)]
        t = thresholds.get(name)
        res, per = {}, {}
        for method in ("Visual-only", "Constraint-only", "CVP"):
            det, acc = [], []
            for d in sel:
                s_det, s_fields = scores(d, tag, method)
                det.append(s_det)
                acc.append(correct(s_fields, d["fields"], d["edited"]) if d["edited"] else np.nan)
            det, acc = np.array(det), np.array(acc)
            has = y & ~np.isnan(acc)
            auc_b = np.array([roc_auc_score(y[b], det[b]) for b in boots])
            attr_b = np.array([np.mean(acc[b][has[b]]) for b in boots])
            amb_b = np.array([np.mean(acc[b][amb[b]]) if amb[b].any() else np.nan for b in boots])
            per[method] = (auc_b, attr_b, amb_b)
            r = dict(AUC=float(roc_auc_score(y, det)), AUC_ci=ci(auc_b),
                     AUC_rule_fields=float(roc_auc_score(y[vis], det[vis])),
                     attr=float(np.mean(acc[has])), attr_ci=ci(attr_b),
                     attr_amb=float(np.mean(acc[amb])) if amb.any() else float("nan"),
                     attr_amb_ci=ci(amb_b[~np.isnan(amb_b)]) if amb.any() else None, n_amb=int(amb.sum()),
                     chance_amb=float(np.mean([1 / len(d["consistent_true"]) for d, a in zip(sel, amb) if a])) if amb.any() else float("nan"))
            if len(out["tools"]) > 1:
                r["by_tool"] = {tl: dict(AUC=float(roc_auc_score(y[tools == tl], det[tools == tl])),
                                         attr=float(np.mean(acc[has & (tools == tl)])), forgeries=int((y & (tools == tl)).sum()))
                                for tl in out["tools"]}
            if method == "CVP" and t is not None:
                flag = np.array([alarm(d, tag, t) for d in sel])
                r["threshold"] = t
                r["flagged_genuine"] = float(flag[~y].mean())
                r["flagged_ai"] = float(flag[y].mean())
            res[method] = r
        for key, k in (("AUC", 0), ("attr", 1), ("attr_amb", 2)):
            diff = per["CVP"][k] - per["Constraint-only"][k]
            res["CVP"][f"{key}_gain_over_constraints_ci"] = ci(diff[~np.isnan(diff)])
        out["results"][tag] = dict(documents=len(sel), forgeries=int(y.sum()), methods=res)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"aiforge_{version}{suffix(jpeg)}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("results", "cord_fields")}, indent=1))
    for tag, r in out["results"].items():
        print(f"{tag:10s} n={r['documents']:4d}", " | ".join(
            f"{m}: AUC {v['AUC']:.3f} [{v['AUC_ci'][0]:.3f}, {v['AUC_ci'][1]:.3f}] attr {100 * v['attr']:.1f} "
            f"amb {100 * v['attr_amb']:.1f}" for m, v in r["methods"].items()))
        c = r["methods"]["CVP"]
        if "flagged_ai" in c:
            print(f"{'':10s} CVP alarms at the validation threshold: genuine {100 * c['flagged_genuine']:.1f}%, "
                  f"AI {100 * c['flagged_ai']:.1f}%")


def table(versions=("v2", "v1")):
    """report/tables/real/aiforge.tex from results/real/aiforge_<version>.json: one block of rows per
    visual branch, one group of columns per AIForge-Doc version."""
    R = {v: json.load(open(os.path.join(RESULTS, f"aiforge_{v}.json"))) for v in versions
         if os.path.exists(os.path.join(RESULTS, f"aiforge_{v}.json"))}
    gen = {"v2": "GPT-Image-2", "v1": "Gemini / Ideogram"}
    branch = {"zero-shot": "synthetic only", "real-crude": "CORD, crude edits", "real-all": "CORD, every category"}
    vs = [v for v in versions if v in R]
    cols = "l" + "ccc" * len(vs)
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{gen[v]}}}" for v in vs) + r"\\",
             "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(vs))),
             "Method & " + " & ".join(["AUC & Attr. & Amb."] * len(vs)) + r"\\"]
    for tag in MODELS:
        if not all(tag in R[v]["results"] for v in vs):
            continue
        n = R[vs[0]]["results"][tag]["forgeries"]
        lines += [r"\midrule", rf"\multicolumn{{{len(cols)}}}{{l}}{{\emph{{Visual branch: {branch[tag]} ({n} receipts)}}}}\\"]
        for m in ("Visual-only", "Constraint-only", "CVP"):
            cells = []
            for v in vs:
                x = R[v]["results"][tag]["methods"][m]
                cells += [f"{x['AUC']:.3f}", f"{100 * x['attr']:.1f}", f"{100 * x['attr_amb']:.1f}"]
            lines.append(f"{m} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = os.path.join(paths.REPORT, "tables", "real", "aiforge.tex")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2", choices=sorted(REPOS))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--table", action="store_true", help="write the LaTeX table from the saved results")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jpeg", action="store_true", help="save both images once more as JPEG first (see above)")
    ap.add_argument("--split", choices=["train", "validation", "test"], help="only forgeries of this CORD split")
    a = ap.parse_args()
    if a.table:
        table()
    elif a.download:
        download(a.version)
    elif a.evaluate:
        evaluate(a.version, a.jpeg)
    else:
        extract(a.version, a.limit, a.jpeg, a.split)


if __name__ == "__main__":
    main()
