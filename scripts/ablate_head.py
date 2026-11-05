"""Test-set ablation of the attribution head, run only after select_head.py has fixed the
design on validation data (results/head_selection.json). Every candidate design is fitted
as in evaluate.py and scored on the test set, so the report can show what each choice
does; the choice itself never sees these numbers. Writes results/head_ablation.json and
report/tables/head_ablation.tex.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E  # noqa: E402
import select_head as SH  # noqa: E402
from cvp import doctypes, paths, pipeline  # noqa: E402

COLS = ["all", "amb", "T1", "T2", "T3", "T4"]


def test_scores(acc, d):
    out = {"all": acc[d["category"] > 0].mean()}
    amb = E.ambiguous(d)
    out["amb"] = acc[amb].mean() if amb.any() else float("nan")
    for i, c in enumerate(E.CATS):
        if i:
            out[c] = acc[d["category"] == i].mean()
    return {k: float(v) for k, v in out.items()}


def main():
    path = os.path.join(paths.BASE, "results", "head_selection.json")
    if not os.path.exists(path):
        sys.exit("run scripts/select_head.py first: the design is chosen on validation data")
    sel = json.load(open(path))
    chosen = SH.name_of(sel["chosen"]["prior"], sel["chosen"]["heat"])
    names = [n for n in doctypes.names() if os.path.exists(os.path.join(paths.feature_dir(n), "test_clean.npz"))]
    res = {}
    for name in names:
        E.set_doctype(name)
        tr, va, te = E.without_blended(E.load("train")), E.without_blended(E.load("val")), E.load("test")
        h = E.Heads().fit(tr, va)
        for prior, heat in SH.CANDIDATES:
            SH.refit(h, va, prior, heat)
            acc = E.attribution_correct(h.attr_features(te, h.p_cls(te)) @ h.attr_w, te)
            res.setdefault(SH.name_of(prior, heat), {})[name] = test_scores(acc, te)
        p = h.p_cls(te)
        rule = np.zeros((len(p), len(E.S.field_names)))
        for i in range(len(p)):
            rule[i, E.S.field_index[pipeline.cvp_rule(p[i], te["w"][i], te["pooled"][i], E.S)]] = 1
        res.setdefault(SH.RULE, {})[name] = test_scores(E.attribution_correct(rule, te), te)
        print(name, "done", flush=True)

    mean = {v: {c: float(np.nanmean([res[v][n][c] for n in names])) for c in COLS} for v in res}
    with open(os.path.join(paths.BASE, "results", "head_ablation.json"), "w") as fh:
        json.dump(dict(types=names, chosen=chosen, per_type=res, mean=mean), fh, indent=1)

    short = {n: n[:4].capitalize() for n in names}
    tex = ["\\begin{tabular}{l" + "c" * (len(names) + 4) + "}", "\\toprule",
           " & \\multicolumn{%d}{c}{Attribution, all tampered} & \\multicolumn{4}{c}{Mean over types}\\\\"
           % len(names),
           "Head (prior / heat) & " + " & ".join(short[n] for n in names) + " & All & Amb. & T2 & T4\\\\",
           "\\midrule"]
    for v in [SH.name_of(*c) for c in SH.CANDIDATES] + [SH.RULE]:
        label = "Algorithm 1 (rule)" if v == SH.RULE else v.replace("/", " / ") + (" (chosen)" if v == chosen else "")
        cells = [f"{100 * res[v][n]['all']:.1f}" for n in names] + [f"{100 * mean[v][c]:.1f}" for c in
                                                                     ("all", "amb", "T2", "T4")]
        tex.append(f"{label} & " + " & ".join(cells) + "\\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(paths.REPORT, "tables", "head_ablation.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"\n{'head':16s} " + " ".join(f"{c:>6s}" for c in COLS) + "   (test, mean over types)")
    for v in mean:
        print(f"{v:16s} " + " ".join(f"{mean[v][c]:6.3f}" for c in COLS) + ("   <- chosen on val_g1" if v == chosen else ""))


if __name__ == "__main__":
    main()
