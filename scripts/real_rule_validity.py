"""Do receipt rules hold on genuine, real receipts? Checked on CORD v2's ground-truth values,
so reading errors play no part: what remains is how faithfully real shops follow the
arithmetic, i.e. the constraint branch's false-alarm floor on real documents.

Writes results/real/cord_rules.json and report/tables/real/cord_rules.tex.

  real_rule_validity.py                every CORD split that has been downloaded
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import paths  # noqa: E402
from cvp.real import cord  # noqa: E402
from cvp.real import receipt_rules as RR  # noqa: E402


def main():
    splits = [s for s in cord.SPLITS if cord.files(s)]
    if not splits:
        sys.exit(f"CORD v2 not found under {cord.ROOT}")
    rules = {k: dict(applies=0, holds=0, failed=[]) for k in RR.RULES}
    n = checked = flagged = 0
    for split in splits:
        for r in cord.receipts(split):
            n += 1
            res = RR.check(r)
            any_rule, all_ok = False, True
            for k, v in res.items():
                if v is None:
                    continue
                vals = v if isinstance(v, list) else [v]
                rules[k]["applies"] += len(vals)
                rules[k]["holds"] += sum(vals)
                any_rule = True
                if not all(vals):
                    all_ok = False
                    rules[k]["failed"].append(r.id)
            checked += any_rule
            flagged += any_rule and not all_ok
    summary = dict(dataset="CORD v2", splits=splits, receipts=n, receipts_checked=checked,
                   genuine_flagged=flagged, false_alarm_rate=flagged / max(checked, 1),
                   tolerance=RR.TOL, rules={k: dict(description=RR.RULES[k], **v) for k, v in rules.items()})
    out = os.path.join(paths.BASE, "results", "real")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cord_rules.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    tex = ["\\begin{tabular}{llrr}", "\\toprule", "Rule & Checks & Applies & Holds\\\\", "\\midrule"]
    for k, v in rules.items():
        rate = f"{100 * v['holds'] / v['applies']:.1f}\\%" if v["applies"] else "--"
        tex.append(f"{k} & {RR.RULES[k].replace('%', chr(92) + '%')} & {v['applies']} & {rate}\\\\")
    tex += ["\\midrule", f"\\multicolumn{{2}}{{l}}{{Genuine receipts flagged (any rule fails)}} & {checked} & "
            f"{100 * flagged / max(checked, 1):.1f}\\%\\\\", "\\bottomrule", "\\end{tabular}"]
    tab = os.path.join(paths.REPORT, "tables", "real")
    os.makedirs(tab, exist_ok=True)
    with open(os.path.join(tab, "cord_rules.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")

    print(f"CORD v2 ({', '.join(splits)}): {n} genuine receipts, {checked} with at least one checkable rule")
    for k, v in rules.items():
        rate = f"{100 * v['holds'] / v['applies']:.1f}%" if v["applies"] else "--"
        print(f"  {k:7s} applies {v['applies']:5d}  holds {rate:>6s}   {RR.RULES[k]}")
    print(f"  genuine receipts flagged by some rule: {flagged}/{checked} = {100 * flagged / max(checked, 1):.1f}%")


if __name__ == "__main__":
    main()
