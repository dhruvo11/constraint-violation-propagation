"""Choose the design of CVP's learned attribution head on validation data made with the
older generator (val_g1: blended T2/T4 edits that no model was trained or tuned on), never
on the test set.

Candidates: field prior {class, shared} x heat {prob, log, logmax} (pipeline.field_prior,
pipeline.attribution_features), plus Algorithm 1 as stated (the rule) for reference. Each
learned head is fitted exactly as in evaluate.py (train/val without blended edits). The
criterion, fixed before looking at any number: top-1 attribution accuracy over every
tampered val_g1 document, averaged over document types, so one design serves every type.
Writes results/head_selection.json and report/tables/head_selection.tex.

  select_head.py                       every type with val_g1 features
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E  # noqa: E402
from cvp import doctypes, paths, pipeline  # noqa: E402

CANDIDATES = [(prior, heat) for prior in ("class", "shared") for heat in pipeline.HEAT]
RULE = "rule"


def name_of(prior, heat):
    return f"{prior}/{heat}"


def refit(h, va, prior, heat):
    """Refit only the attribution head of fitted Heads `h` for one design, exactly as
    Heads.fit does (g_phi, the detector and the baselines do not depend on the design).
    Leaves evaluate.py's design globals set to this design."""
    E.SHARED, E.HEAT = prior == "shared", heat
    t = va["category"] > 0
    h.attr_w = E.fit_softmax(h.attr_features(va, h.p_cls(va))[t], va["primary"][t])
    return h


def scores(acc, d):
    t = d["category"] > 0
    out = dict(all=acc[t].mean(), T2=acc[d["category"] == 2].mean(), T4=acc[d["category"] == 4].mean())
    amb = E.ambiguous(d)
    out["amb"] = acc[amb].mean() if amb.any() else float("nan")
    return {k: float(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctypes", default=",".join(doctypes.names()))
    a = ap.parse_args()
    names = [n for n in a.doctypes.split(",")
             if os.path.exists(os.path.join(paths.feature_dir(n), "val_g1_clean.npz"))]
    res = {}  # variant -> type -> {all, T2, T4, amb}
    for name in names:
        E.set_doctype(name)
        tr, va = E.without_blended(E.load("train")), E.without_blended(E.load("val"))
        g1 = E.load("val_g1")
        h = E.Heads().fit(tr, va)
        for prior, heat in CANDIDATES:
            refit(h, va, prior, heat)
            acc = E.attribution_correct(h.attr_features(g1, h.p_cls(g1)) @ h.attr_w, g1)
            res.setdefault(name_of(prior, heat), {})[name] = scores(acc, g1)
        p = h.p_cls(g1)  # g_phi does not depend on the head design
        rule = np.zeros((len(p), len(E.S.field_names)))
        for i in range(len(p)):
            rule[i, E.S.field_index[pipeline.cvp_rule(p[i], g1["w"][i], g1["pooled"][i], E.S)]] = 1
        res.setdefault(RULE, {})[name] = scores(E.attribution_correct(rule, g1), g1)
        print(name, " ".join(f"{v} {res[v][name]['all']:.3f}" for v in res), flush=True)

    mean = {v: float(np.mean([res[v][n]["all"] for n in names])) for v in res}
    learned = [name_of(*c) for c in CANDIDATES]
    best = max(learned, key=lambda v: mean[v])
    prior, heat = best.split("/")
    out = dict(types=names, criterion="mean over types of top-1 attribution on tampered val_g1 documents",
               mean=mean, per_type=res, chosen=dict(prior=prior, heat=heat))
    os.makedirs(os.path.join(paths.BASE, "results"), exist_ok=True)
    with open(os.path.join(paths.BASE, "results", "head_selection.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    short = {n: n[:4].capitalize() for n in names}
    tex = ["\\begin{tabular}{l" + "c" * (len(names) + 1) + "}", "\\toprule",
           "Head (prior / heat) & " + " & ".join(short[n] for n in names) + " & Mean\\\\", "\\midrule"]
    for v in learned + [RULE]:
        label = "Algorithm 1 (rule)" if v == RULE else v.replace("/", " / ")
        cells = [f"{100 * res[v][n]['all']:.1f}" for n in names] + [f"{100 * mean[v]:.1f}"]
        if v == best:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        tex.append(f"{label} & " + " & ".join(cells) + "\\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.join(paths.REPORT, "tables"), exist_ok=True)
    with open(os.path.join(paths.REPORT, "tables", "head_selection.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print("\nmean over types (attribution on tampered val_g1):")
    for v in sorted(mean, key=mean.get, reverse=True):
        print(f"  {v:16s} {mean[v]:.4f}" + ("   <- chosen" if v == best else ""))


if __name__ == "__main__":
    main()
