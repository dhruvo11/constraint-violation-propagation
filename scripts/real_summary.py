"""One page for every real-data result: results/real/summary.md and report/tables/real/cord_regimes.tex.
Reads the JSON files the real_*.py scripts write."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import paths  # noqa: E402

R = os.path.join(paths.BASE, "results", "real")
REGIMES = [("", "synthetic documents only (zero-shot)"), ("_crude", "real photos, crude edits only"),
           ("_all", "real photos, every category (has seen the blended renderer)")]


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x):
    return "--" if x is None or x != x else f"{100 * x:.1f}"


def main():
    md = ["# Real-data results (Phase 2)", ""]
    rules = load("cord_rules.json")
    if rules:
        md += ["## Do real receipts satisfy the rules? (CORD v2, 1,000 genuine receipts, true values)", "",
               "| Rule | Applies | Holds |", "|---|---|---|"]
        for k, v in rules["rules"].items():
            md.append(f"| {v['description']} | {v['applies']} | {pct(v['holds'] / max(1, v['applies']))}% |")
        md += ["", f"Genuine receipts that break some rule: **{pct(rules['false_alarm_rate'])}%** "
                   f"({rules['genuine_flagged']}/{rules['receipts_checked']}).", ""]
    ocr = [load(f"cord_ocr_{s}.json") for s in ("train", "validation", "test")]
    if all(ocr):
        n = sum(o["receipts_checked"] for o in ocr)
        md += [f"With a free local OCR (docTR) reading each value at its layout label, genuine receipts breaking a "
               f"rule rise to **{pct(sum(o['flagged_as_read'] for o in ocr) / n)}%** (from "
               f"{pct(sum(o['flagged_true_values'] for o in ocr) / n)}% on true values).", ""]
    fc = load("cord_forgery_check_test.json")
    if fc:
        md += [f"Forgeries of real photos read back exactly {pct(fc['edited_read_back'])}% of the time; as read, "
               f"single-field forgeries break a rule {pct(fc['flagged_as_read']['T1'])}% (T1) and "
               f"{pct(fc['flagged_as_read']['T2'])}% (T2), coherent ones keep every rule {pct(fc['coherent_keep_rules'])}%.", ""]
    md += ["## CVP on real receipt photos (CORD v2 test: 98 genuine, 98 T1, 98 T2, 90 T3, 90 T4)", "",
           "| Visual branch trained on | Method | AUC | Attr. all | Attr. T2 | Attr. T4 | Ambiguity split | "
           "Flagged at validation threshold: genuine / T1 / T2 / T3 / T4 |", "|---|---|---|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{llcccccc}", "\\toprule",
           "Visual branch & Method & AUC & AUC T2 & Attr. & Attr. T2 & Attr. T4 & Amb.\\\\", "\\midrule"]
    for tag, label in REGIMES:
        s = load(f"cord_cvp_test{tag}.json")
        if not s:
            continue
        for m in ("Visual-only", "Constraint-only", "CVP"):
            r = s["results"][m]
            flags = ""
            if m == "CVP" and "threshold" in s:
                f = s["threshold"]["flagged"]
                flags = " / ".join(pct(f.get(c)) for c in ("genuine", "T1", "T2", "T3", "T4"))
            md.append(f"| {label if m == 'Visual-only' else ''} | {m} | {r['AUC']:.3f} | {pct(r['attr_all'])} | "
                      f"{pct(r['attr_T2'])} | {pct(r['attr_T4'])} | {pct(r['attr_amb'])} | {flags} |")
            tex.append(f"{label.split(' (')[0] if m == 'Visual-only' else ''} & {m} & {r['AUC']:.3f} & "
                       f"{r['AUC_T2']:.3f} & {pct(r['attr_all'])} & {pct(r['attr_T2'])} & {pct(r['attr_T4'])} & "
                       f"{pct(r['attr_amb'])}\\\\")
        tex.append("\\midrule")
        chance = s["chance_ambiguous"]
    tex[-1] = "\\bottomrule"
    tex.append("\\end{tabular}")
    md += ["", f"Chance on the ambiguity split: {pct(chance)}%. The last regime has seen the blended renderer that "
               "made the test forgeries, so it is an upper bound, not a test of unseen blending.", ""]
    fi = load("findit_rules.json")
    if fi:
        md += ["## Real human forgeries (Find it again!, 987 scans)", "", "| Reading | Receipts | Checkable | Flagged by a rule |",
               "|---|---|---|---|"]
        for source in ("ocr", "transcription"):
            for kind in ("genuine", "forged"):
                c = fi[source][kind]
                md.append(f"| {source} ({kind}) | {c.get('receipts', 0)} | {c.get('checkable', 0)} | {pct(c['flag_rate'])}% |")
    fv = load("findit_visual.json")
    if fv:
        md += ["", "Visual branch on the real forgeries (test scans): heat AUC, how often the hottest pixel is in an "
               "edited region (chance in brackets), and CVP's flags at a validation threshold.", "",
               "| Visual branch | Heat AUC | Pointing | CVP flags genuine / forged |", "|---|---|---|---|"]
        for m, r in fv.items():
            if isinstance(r, dict) and "AUC_heat" in r:
                md.append(f"| {m} | {r['AUC_heat']:.3f} | {pct(r['pointing'])}% ({pct(r['pointing_chance'])}%) | "
                          f"{pct(r['flagged_genuine'])}% / {pct(r['flagged_forged'])}% |")
    os.makedirs(R, exist_ok=True)
    with open(os.path.join(R, "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    os.makedirs(os.path.join(paths.REPORT, "tables", "real"), exist_ok=True)
    with open(os.path.join(paths.REPORT, "tables", "real", "cord_regimes.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
