"""Figures the paper needs beyond make_figures.py, from the results already computed.

Across the six synthetic types (report/figures/):
  cross_flags       how often each method flags genuine, T2 and T4 documents
  cross_regimes     T2 attribution with the visual branch zero-shot, main and oracle
  cross_robustness  attribution on clean, JPEG q=60 and 0.6x resized pages
Real documents (report/figures/real/):
  cord_rules        do genuine CORD receipts satisfy the rules, and what OCR adds
  cord_regimes      CVP on forged real photos, three visual branches
  cord_flags        what CVP flags on real photos at the validation threshold
  findit_rules      which rules real human forgeries break (Find it again!)
  cord_example      one real receipt, forged, and CVP's evidence (needs docTR and models)

  make_paper_figures.py               everything
  make_paper_figures.py --no-example  skip the panel that runs the models
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import COLORS, FIG, GRID, INK, INK2, SHORT, grid, save  # noqa: E402
from cvp import doctypes, paths  # noqa: E402

REAL = os.path.join(paths.BASE, "results", "real")
OUT_REAL = os.path.join(FIG, "real")
METHODS = ["Visual-only", "Constraint-only", "CVP"]
# ordinal: how much the visual branch has seen of real photos (blue ramp, steps 250 / 450 / 650)
REGIME_BLUES = ["#86b6ef", "#2a78d6", "#104281"]
GENUINE, FORGED = "#898781", "#e34948"  # neutral for genuine, categorical red for forged


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes in n, as (low, high) proportions."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def bars(ax, x, groups, values, colors, labels, bw):
    n = len(groups)
    for j, (vals, col, lab) in enumerate(zip(values, colors, labels)):
        ax.bar(x + (j - (n - 1) / 2) * bw, vals, bw, color=col, edgecolor="white", linewidth=0.6, label=lab)


def load_types():
    out = {}
    for name in doctypes.names():
        p = os.path.join(paths.results_dir(name), "results.json")
        if os.path.exists(p):
            out[name] = json.load(open(p))
    return out


# ------------------------------------------------------------------ across synthetic types
def cross_flags(S):
    names = list(S)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0))
    for ax, (cat, title, top) in zip(axes, [("genuine", "Genuine documents flagged (%)", 12),
                                            ("T2", "T2 blended edits flagged (%)", 105),
                                            ("T4", "T4 complete forgeries flagged (%)", 105)]):
        vals = [[100 * S[t]["results"][m][f"rate_{cat}"] for t in names] for m in METHODS]
        bars(ax, x, METHODS, vals, [COLORS[m] for m in METHODS], METHODS, 0.26)
        ax.set_ylim(0, top)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xticks(x, [SHORT[t] for t in names], rotation=35, ha="right", fontsize=6.5)
        grid(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, 1.08), handlelength=1)
    fig.subplots_adjust(wspace=0.28)
    save(fig, FIG, "cross_flags")


def cross_regimes(S):
    names = [t for t in S if all(r in (S[t].get("heldout") or {}) for r in ("blind", "main", "oracle"))]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0), sharey=True)
    for ax, (reg, title) in zip(axes, [("blind", "Zero-shot: crude edits only"),
                                       ("main", "Main: crude + older generator g1"),
                                       ("oracle", "Oracle: the test generator g2")]):
        vals = [[100 * S[t]["heldout"][reg][m]["attr_T2"] for t in names] for m in METHODS]
        bars(ax, x, METHODS, vals, [COLORS[m] for m in METHODS], METHODS, 0.26)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xticks(x, [SHORT[t] for t in names], rotation=35, ha="right", fontsize=6.5)
        grid(ax)
    axes[0].set_ylabel("T2 attribution (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, 1.08), handlelength=1)
    fig.subplots_adjust(wspace=0.12)
    save(fig, FIG, "cross_regimes")


def cross_robustness(S):
    names = list(S)
    x = np.arange(len(names))
    conds = list(next(iter(S.values()))["robustness"])
    titles = {c: c.replace("$q{=}60$", "q=60").replace("$0.6\\times$", "0.6x") for c in conds}
    fig, axes = plt.subplots(1, len(conds), figsize=(7.16, 2.0), sharey=True)
    for ax, c in zip(axes, conds):
        vals = [[100 * S[t]["robustness"][c][m]["attr_all"] for t in names] for m in METHODS]
        bars(ax, x, METHODS, vals, [COLORS[m] for m in METHODS], METHODS, 0.26)
        ax.set_ylim(0, 105)
        ax.set_title(titles[c] + " pages", fontsize=7.5, color=INK)
        ax.set_xticks(x, [SHORT[t] for t in names], rotation=35, ha="right", fontsize=6.5)
        grid(ax)
    axes[0].set_ylabel("Attribution, all tampered (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, 1.08), handlelength=1)
    fig.subplots_adjust(wspace=0.12)
    save(fig, FIG, "cross_robustness")


# ------------------------------------------------------------------ real receipts
def cord_rules():
    R = json.load(open(os.path.join(REAL, "cord_rules.json")))
    ocr = [json.load(open(os.path.join(REAL, f"cord_ocr_{s}.json"))) for s in ("train", "validation", "test")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 1.9), gridspec_kw={"width_ratios": [1.35, 1]})
    names = list(R["rules"])
    for i, r in enumerate(names):
        k, n = R["rules"][r]["holds"], R["rules"][r]["applies"]
        lo, hi = wilson(k, n)
        a.plot([100 * lo, 100 * hi], [i, i], color=INK2, lw=1.2, solid_capstyle="round")
        a.plot(100 * k / n, i, "o", ms=5, color=COLORS["Constraint-only"], markeredgecolor="white", markeredgewidth=0.6)
        a.text(100.4, i, f"{100 * k / n:.1f}%  ({k}/{n})", va="center", fontsize=6.3, color=INK)
    a.set_yticks(range(len(names)), names, fontsize=7)
    a.invert_yaxis()
    a.set_xlim(90, 100)
    a.set_xlabel("Genuine receipts on which the rule holds (%), 95% CI", fontsize=7)
    a.set_title("Rule validity on 1,000 genuine CORD receipts", fontsize=7.5, color=INK)
    a.xaxis.grid(True, color=GRID, linewidth=0.6)
    a.set_axisbelow(True)
    n = sum(o["receipts_checked"] for o in ocr)
    rows = [("all printed words,\ntrue values", R["genuine_flagged"], R["receipts_checked"]),
            ("fields CVP reads,\ntrue values", sum(o["flagged_true_values"] for o in ocr), n),
            ("fields CVP reads,\nread by docTR", sum(o["flagged_as_read"] for o in ocr), n)]
    for i, (lab, k, m) in enumerate(rows):
        lo, hi = wilson(k, m)
        b.barh(i, 100 * k / m, 0.55, color=[GENUINE, GENUINE, INK2][i], edgecolor="white")
        b.plot([100 * lo, 100 * hi], [i, i], color=INK, lw=0.9)
        b.text(100 * hi + 0.3, i, f"{100 * k / m:.1f}%", va="center", fontsize=6.5, color=INK)
    b.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6.5)
    b.invert_yaxis()
    b.set_xlim(0, 16)
    b.set_xlabel("Genuine receipts that break a rule (%)", fontsize=7)
    b.set_title("False alarms from the rules alone", fontsize=7.5, color=INK)
    b.xaxis.grid(True, color=GRID, linewidth=0.6)
    b.set_axisbelow(True)
    fig.subplots_adjust(wspace=0.9)
    save(fig, OUT_REAL, "cord_rules")


def real_results():
    tags = [("", "zero-\nshot"), ("_crude", "real,\ncrude"), ("_all", "real,\nall")]
    return [(lab, json.load(open(os.path.join(REAL, f"cord_cvp_test{t}.json")))) for t, lab in tags]


def cord_regimes():
    regs = real_results()
    x = np.arange(len(regs))
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.05))
    panels = [("AUC", "Detection AUC", 1), ("attr_all", "Attribution, all tampered (%)", 100),
              ("attr_T2", "Attribution, T2 blended (%)", 100), ("attr_amb", "Ambiguity split (%)", 100)]
    for ax, (key, title, scale) in zip(axes, panels):
        vals = [[scale * R["results"][m][key] for _, R in regs] for m in METHODS]
        if key == "AUC":
            for j, m in enumerate(METHODS):
                ax.plot(x + (j - 1) * 0.22, vals[j], "o", ms=4.5, color=COLORS[m], markeredgecolor="white",
                        markeredgewidth=0.6, label=m)
            ax.axhline(0.5, color=INK2, ls="--", lw=0.8)
            ax.set_ylim(0.4, 1.0)
        else:
            bars(ax, x, METHODS, vals, [COLORS[m] for m in METHODS], METHODS, 0.26)
            ax.set_ylim(0, 100)
        if key == "attr_amb":
            ax.axhline(100 * regs[0][1]["chance_ambiguous"], color=INK2, ls="--", lw=0.8)
        ax.set_xticks(x, [lab for lab, _ in regs], fontsize=6.3)
        ax.set_title(title, fontsize=7.2, color=INK)
        grid(ax)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, 1.08), handlelength=1)
    fig.subplots_adjust(wspace=0.38)
    save(fig, OUT_REAL, "cord_regimes")


def cord_flags():
    regs = real_results()
    cats = ["genuine", "T1", "T2", "T3", "T4"]
    names = {"genuine": "Genuine", "T1": "T1", "T2": "T2", "T3": "T3", "T4": "T4"}
    legend = {"zero-\nshot": "visual branch trained on synthetic documents (zero-shot)",
              "real,\ncrude": "fine-tuned on real photos, crude edits only",
              "real,\nall": "fine-tuned on real photos, every category"}
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(3.5, 1.95))
    vals = [[100 * R["threshold"]["flagged"][c] for c in cats] for _, R in regs]
    bars(ax, x, regs, vals, REGIME_BLUES, [legend[lab] for lab, _ in regs], 0.26)
    ax.set_xticks(x, [names[c] for c in cats], fontsize=6.3)
    ax.set_ylabel("Flagged by CVP (%)", fontsize=7)
    ax.set_ylim(0, 105)
    grid(ax)
    ax.legend(ncol=1, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 1.33), handlelength=1)
    save(fig, OUT_REAL, "cord_flags")


def findit_rules():
    F = json.load(open(os.path.join(REAL, "findit_rules.json")))["ocr"]
    rules = ["change", "tender", "rounding", "total"]
    desc = {"change": "change\ncash \u2212 total = change", "tender": "tender\ncash \u2265 total",
            "rounding": "rounding\n|adjustment| \u2264 0.02", "total": "total\nsubtotal + tax = total",
            "any": "any rule\n(checkable receipts)"}
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    rows = rules + ["any"]
    for j, (who, col) in enumerate([("genuine", GENUINE), ("forged", FORGED)]):
        for i, r in enumerate(rows):
            if r == "any":
                k, n = F[who]["flagged"], F[who]["checkable"]
            else:
                n = F[who][f"{r} applies"]
                k = n - F[who][f"{r} holds"]
            lo, hi = wilson(k, n)
            y = i + (j - 0.5) * 0.36
            ax.barh(y, 100 * k / n, 0.34, color=col, edgecolor="white", linewidth=0.6,
                    label=f"{who} scans" if i == 0 else None)
            ax.plot([100 * lo, 100 * hi], [y, y], color=INK, lw=0.8)
            ax.text(100 * hi + 0.8, y, f"{100 * k / n:.1f}% of {n}", va="center", fontsize=5.6, color=INK)
    ax.set_yticks(range(len(rows)), [desc[r] for r in rows], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_xlabel("Scans on which the rule fails (%), 95% CI", fontsize=7)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.45, 1.14), handlelength=1)
    save(fig, OUT_REAL, "findit_rules")


# ------------------------------------------------------------------ one real receipt
def cord_example(k, field_label, cat):
    """Receipt k of the CORD test split, genuine and forged (category cat, at field_label), through
    the system the app runs, with the evaluation's seeds (scripts/real_extract_cord.py)."""
    from cvp.real import cord, demo, forge, view
    k = demo.index(k)
    r = cord.receipt(demo.SPLIT, k)
    lab = demo.labels(r)
    field = {v: f for f, v in lab.items()}[field_label]
    sysm = demo.system()
    rng = lambda cat: np.random.default_rng(1000 * k + forge.CATEGORIES.index(cat))  # noqa: E731
    img_g, _ = forge.forge(r, "genuine", rng("genuine"))
    img_f, info = forge.forge(r, cat, rng(cat), field=field)
    res = sysm(r, img_f)
    v_g, s = view.visual_view(img_g)
    v_f, _ = view.visual_view(img_f)
    ev = np.asarray(demo.overlay(v_f, res, info["rects"], s))
    # zoom on the fields the broken rules read, with their printed labels to the left
    shown = [f for f in res["consistent"] + ["total"] if f in res["boxes"]]
    bx = np.array([view.scale_box(res["boxes"][f], s) for f in shown], float)
    x0, y0, x1, y1 = bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()
    h, w = ev.shape[:2]
    rows = y1 - y0
    x0, x1 = max(0, int(x1 - 0.82 * w)), min(w, int(x1 + 0.04 * w))  # amounts sit right, labels left
    y0, y1 = max(0, int(y0 - 0.35 * rows - 8)), min(h, int(y1 + 0.2 * rows + 8))
    fig = plt.figure(figsize=(3.5, 2.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.45], wspace=0.04, hspace=0.28)
    axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    old, new = demo.money(info["old"][field]), demo.money(info["new"][field])
    panels = [(ev, "(a) CVP evidence"),
              (v_g[y0:y1, x0:x1], "(b) genuine, zoomed"),
              (ev[y0:y1, x0:x1], f"(c) {cat}: {field_label.lower()} {old} → {new}")]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=6.8, color=INK)
        ax.axis("off")
    axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#0b0b0b", lw=0.8, ls="--"))
    save(fig, OUT_REAL, "cord_example")
    print(f"receipt {k}: {field} {old} -> {new}; violated {sorted(res['violated'])}; consistent "
          f"{res['consistent']}; heat " + ", ".join(f"{f} {res['heat'].get(f, 0):.3f}" for f in res["consistent"]) +
          f"; attributed {res['field']} (tampered {res['tampered']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-example", action="store_true")
    ap.add_argument("--receipt", type=int, default=2, help="CORD test receipt for the example")
    ap.add_argument("--field", default="Change", help="label of the field the example's edit changes")
    ap.add_argument("--category", default="T1", help="forgery category of the example")
    a = ap.parse_args()
    S = load_types()
    cross_flags(S)
    cross_regimes(S)
    cross_robustness(S)
    cord_rules()
    cord_regimes()
    cord_flags()
    findit_rules()
    if not a.no_example:
        cord_example(a.receipt, a.field, a.category)
    print("figures written to", FIG, "and", OUT_REAL)


if __name__ == "__main__":
    main()
