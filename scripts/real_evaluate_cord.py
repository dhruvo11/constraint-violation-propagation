"""Evaluate CVP on real receipts from features/real/cord_<split><tag>.pkl
(scripts/real_extract_cord.py).

Methods, as in the synthetic evaluation:
- Visual-only: the field with the most heat; detection score = the page's maximum heat.
- Constraint-only: a uniform guess among the fields the violations are consistent with (all
  fields when none is violated); detection score = the number of violated rules.
- CVP: Algorithm 1 with the receipt's own signature lookup. The violated rules nominate the
  consistent fields and the heatmap picks among them (all fields when none is violated).
  Detection is the noisy-OR of the two branches: a violation, or else the visual score.
- CVP (one class): the stricter form that keeps only the signature class's candidates.
Only exact ties count as ties. The ambiguity split holds the single-field edits (T1/T2)
whose true violation pattern fits more than one field; chance there is the mean 1/|C(V)|.

Writes results/real/cord_cvp_<split><tag>.json and report/tables/real/cord_cvp<tag>.tex.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import paths  # noqa: E402

CATS = ["genuine", "T1", "T2", "T3", "T4"]
METHODS = ["Visual-only", "Constraint-only", "CVP (one class)", "CVP"]


def correct(scores, fields, edited):
    """Probability that the top-scoring field is an edited one; exact ties share the credit."""
    s = np.asarray(scores, float)
    top = s == s.max()
    hits = sum(1 for i, f in enumerate(fields) if top[i] and f in edited)
    return hits / top.sum()


def score_row(d):
    fields, heat = d["fields"], d["pooled"][:, 0]
    violated = bool(np.asarray(d["w"]).sum())

    def within(cands):
        return [heat[i] if f in cands else -1.0 for i, f in enumerate(fields)]
    det = {"Visual-only": d["hmax"], "Constraint-only": float(np.asarray(d["w"]).sum()),
           "CVP": 1.0 + d["hmax"] if violated else d["hmax"]}  # noisy-OR: a violation outranks any heat
    det["CVP (one class)"] = det["CVP"]
    cons = d["consistent"] if violated else fields
    attr = {"Visual-only": list(heat), "Constraint-only": [1.0 if f in cons else 0.0 for f in fields],
            "CVP": within(cons) if violated else list(heat),
            "CVP (one class)": within(d["candidates"]) if violated else list(heat)}
    return det, attr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default="", help="visual model of these features: reports CVP's flag rates at its "
                                                 "validation threshold (results/real/cord_threshold.json)")
    a = ap.parse_args()
    rows = pickle.load(open(os.path.join(paths.BASE, "features", "real", f"cord_{a.split}{a.tag}.pkl"), "rb"))
    y = np.array([d["category"] != "genuine" for d in rows])
    cat = np.array([d["category"] for d in rows])
    amb = np.array([d["category"] in ("T1", "T2") and len(d["consistent_true"]) > 1 for d in rows])
    chance = float(np.mean([1.0 / len(d["consistent_true"]) for d, m in zip(rows, amb) if m])) if amb.any() else float("nan")
    scored = [score_row(d) for d in rows]
    res = {}
    for m in METHODS:
        s = np.array([det[m] for det, _ in scored])
        acc = np.array([correct(attr[m], d["fields"], d["edited"]) if d["edited"] else np.nan
                        for d, (_, attr) in zip(rows, scored)])
        r = dict(AUC=float(roc_auc_score(y, s)), attr_all=float(np.nanmean(acc[y])),
                 attr_amb=float(np.nanmean(acc[amb])) if amb.any() else float("nan"))
        for c in CATS[1:]:
            sel = (cat == "genuine") | (cat == c)
            r[f"AUC_{c}"] = float(roc_auc_score(y[sel], s[sel])) if (cat == c).any() else float("nan")
            r[f"attr_{c}"] = float(np.nanmean(acc[cat == c])) if (cat == c).any() else float("nan")
        res[m] = r
    summary = dict(dataset="CORD v2", split=a.split, tag=a.tag, documents=len(rows),
                   per_category={c: int((cat == c).sum()) for c in CATS}, n_ambiguous=int(amb.sum()),
                   chance_ambiguous=chance, results=res)
    tfile = os.path.join(paths.BASE, "results", "real", "cord_threshold.json")
    if a.model and os.path.exists(tfile) and a.model in json.load(open(tfile)):
        t = json.load(open(tfile))[a.model]
        flag = np.array([bool(np.asarray(d["w"]).sum()) or d["hmax"] >= t for d in rows])
        summary["threshold"] = dict(model=a.model, value=t,
                                    flagged={c: float(flag[cat == c].mean()) for c in CATS if (cat == c).any()})
    os.makedirs(os.path.join(paths.BASE, "results", "real"), exist_ok=True)
    with open(os.path.join(paths.BASE, "results", "real", f"cord_cvp_{a.split}{a.tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    pct = lambda x: "--" if x != x else f"{100 * x:.1f}"  # noqa: E731
    tex = ["\\begin{tabular}{lccccccc}", "\\toprule",
           "Method & AUC & AUC T2 & AUC T4 & Attr. all & Attr. T2 & Attr. T4 & Ambiguity split\\\\", "\\midrule"]
    for m in METHODS:
        r = res[m]
        tex.append(f"{m} & {r['AUC']:.3f} & {r['AUC_T2']:.3f} & {r['AUC_T4']:.3f} & {pct(r['attr_all'])} & "
                   f"{pct(r['attr_T2'])} & {pct(r['attr_T4'])} & {pct(r['attr_amb'])}\\\\")
    tex += ["\\midrule", f"Chance & & & & & & & {pct(chance)}\\\\", "\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.join(paths.REPORT, "tables", "real"), exist_ok=True)
    with open(os.path.join(paths.REPORT, "tables", "real", f"cord_cvp{a.tag}.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"CORD v2 {a.split}{a.tag}: {len(rows)} documents {summary['per_category']}, "
          f"{int(amb.sum())} ambiguous (chance {pct(chance)}%)")
    for m in METHODS:
        r = res[m]
        print(f"  {m:16s} AUC {r['AUC']:.3f} (T1 {r['AUC_T1']:.3f} T2 {r['AUC_T2']:.3f} T3 {r['AUC_T3']:.3f} T4 {r['AUC_T4']:.3f}) | "
              f"attr all {pct(r['attr_all'])} T1 {pct(r['attr_T1'])} T2 {pct(r['attr_T2'])} T3 {pct(r['attr_T3'])} "
              f"T4 {pct(r['attr_T4'])} | ambiguity split {pct(r['attr_amb'])}")
    if "threshold" in summary:
        th = summary["threshold"]
        print(f"  CVP flags at the validation threshold {th['value']:.4f}: " +
              ", ".join(f"{c} {pct(v)}%" for c, v in th["flagged"].items()))


if __name__ == "__main__":
    main()
