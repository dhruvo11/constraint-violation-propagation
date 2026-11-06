"""One table across every document type: constraint structure, OCR accuracy, and how each
method detects and attributes forgeries. Writes results/summary.md and
report/tables/cross_type.tex from results/<type>/results.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import doctypes, paths  # noqa: E402

METHODS = ["Visual-only", "Constraint-only", "Concat fusion", "CVP"]


def pct(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def main():
    rows = []
    for name in doctypes.names():
        path = os.path.join(paths.results_dir(name), "results.json")
        if not os.path.exists(path):
            continue
        s = json.load(open(path))
        T = doctypes.get(name)
        S = T.schema
        rows.append(dict(name=name, title=T.title, fields=len(S.field_names), rules=len(S.constraints),
                         rho=s["rho_theory"], amb=s["n_ambiguous"], chance=s.get("chance_ambiguous"),
                         ocr=s["ocr_acc"].get("Clean"), res=s["results"],
                         t2=(s.get("heldout") or {}).get("blind"), head=(s.get("prior", "class"), s.get("heat", "prob"))))
    if not rows:
        print("no results yet")
        return
    heads = {r["head"] for r in rows}
    head_note = ("CVP attribution head: " + "; ".join(f"{p} prior, {h} heat" for p, h in sorted(heads)) +
                 (" (chosen on g1 validation data, results/head_selection.json)."
                  if os.path.exists(os.path.join(paths.BASE, "results", "head_selection.json")) else "."))
    if len(heads) > 1:
        head_note += " WARNING: the types were evaluated with different head designs."

    md = ["| Type | Fields / rules | rho | OCR | Ambig. edits (chance) | " +
          " | ".join(f"{m} AUC / attr. / ambig." for m in METHODS) + " |",
          "|---|---|---|---|---|" + "---|" * len(METHODS)]
    for r in rows:
        cells = [f"{r['res'][m]['AUC']:.3f} / {pct(r['res'][m]['attr_all'])} / {pct(r['res'][m]['attr_amb'])}"
                 for m in METHODS]
        md.append(f"| {r['title']} | {r['fields']} / {r['rules']} | {r['rho']:.2f} | {pct(r['ocr'])} | "
                  f"{r['amb']} ({pct(r['chance'])}) | " + " | ".join(cells) + " |")
    os.makedirs(paths.BASE + "/results", exist_ok=True)
    with open(os.path.join(paths.BASE, "results", "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n\n" + head_note + "\n")

    tex = ["\\begin{tabular}{lcccc" + "cc" * len(METHODS) + "}", "\\toprule",
           " & & & & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in METHODS) + "\\\\",
           "Type & $|F|/|C|$ & $\\rho$ & OCR & $n_{\\mathrm{amb}}$ & " + " & ".join(["Attr. & Amb."] * len(METHODS)) + "\\\\",
           "\\midrule"]
    for r in rows:
        best = max(r["res"][m]["attr_all"] for m in METHODS)
        cells = []
        for m in METHODS:
            a = pct(r["res"][m]["attr_all"])
            cells.append(f"\\textbf{{{a}}}" if np.isclose(r["res"][m]["attr_all"], best) else a)
            cells.append(pct(r["res"][m]["attr_amb"]))
        tex.append(f"{r['title']} & {r['fields']}/{r['rules']} & {r['rho']:.2f} & {pct(r['ocr'])} & {r['amb']} & "
                   + " & ".join(cells) + "\\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.join(paths.REPORT, "tables"), exist_ok=True)
    with open(os.path.join(paths.REPORT, "tables", "cross_type.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print("\n".join(md) + "\n\n" + head_note)


if __name__ == "__main__":
    main()
