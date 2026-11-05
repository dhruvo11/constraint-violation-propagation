"""Figures for the report (PDF for LaTeX, PNG for the app) from results/<type>/results.json.

Per document type, in report/figures/<type>/: attribution, detection, robustness, regimes and
signatures (from the evaluation summary), examples (one T1 and one T2 edit of the same
field, run through the trained system) and page.png (a genuine page). Across types, in
report/figures/: cross_type (detection and attribution for every type) and doctypes.png
(one genuine page of each type).

  make_figures.py                       every type that has results
  make_figures.py --doctypes invoice    some types only
  make_figures.py --no-examples         skip the panels that need the trained models
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E  # noqa: E402
from cvp import doctypes, paths, pipeline  # noqa: E402
from cvp.core import render, tamper  # noqa: E402

FIG = os.path.join(paths.REPORT, "figures")
# Fixed categorical order (reference palette, light mode); colour follows the method.
COLORS = {"CVP": "#2a78d6", "Visual-only": "#eb6834", "Constraint-only": "#1baf7a",
          "Concat fusion": "#eda100", "ELA": "#e87ba4", "CVP (rule)": "#6250d6"}
ORDER = ["ELA", "Visual-only", "Constraint-only", "Concat fusion", "CVP (rule)", "CVP"]
MAIN = ["Visual-only", "Constraint-only", "Concat fusion", "CVP"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95"])
SHORT = {"transcript": "Transcript", "invoice": "GST invoice", "receipt": "Receipt",
         "certificate": "Certificate", "passport": "Passport", "idcard": "Student ID"}
# The field each type's qualitative example edits, whether the example should show fields the
# constraints cannot tell apart (a Proposition 1 class or a shared signature), and the first
# seed tried. The figure uses the first seed from there on where CVP attributes both edits
# correctly: it illustrates the mechanism, the measured rates are in the other figures.
EXAMPLE = {"transcript": ("m3", True, 11), "invoice": ("qty2", True, 0), "receipt": ("change", True, 0),
           "certificate": ("grade", True, 0), "passport": ("viz_dob", True, 0), "idcard": ("valid", False, 0)}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def finite(x):
    return x is not None and bool(np.isfinite(x))


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=160)
    plt.close(fig)


def grid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ per type, from the summary
def attribution(R, chance, out):
    groups = [("attr_all", "All tampered\n(T1-T4)"), ("attr_unamb", "Unambiguous\n(T1/T2)"),
              ("attr_amb", "Ambiguity split\n(T1/T2)")]
    groups = [(k, g) for k, g in groups if any(finite(R[m][k]) for m in ORDER)]  # a type may have no ambiguity
    keys = [k for k, _ in groups]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    n, bw = len(ORDER), 0.13
    for j, m in enumerate(ORDER):
        xs = np.arange(len(groups)) + (j - (n - 1) / 2) * bw
        ax.bar(xs, [100 * R[m][k] if finite(R[m][k]) else 0 for k in keys], bw, color=COLORS[m],
               edgecolor="white", linewidth=0.8, label=m)
    if "attr_amb" in keys:
        g = keys.index("attr_amb")
        if finite(chance):
            ax.hlines(100 * chance, g - 0.45, g + 0.45, colors=INK2, linestyles="--", linewidth=0.8)
            ax.text(g + 0.47, 100 * chance, "chance\n1/|C(V)|", va="center", ha="left", fontsize=6.5, color=INK2)
        v = R["CVP"]["attr_amb"]
        if finite(v):
            ax.text(g + (n - 1) / 2 * bw, 100 * v + 2, f"{100 * v:.0f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(groups)), [g for _, g in groups])
    ax.set_ylabel("Top-1 field attribution (%)")
    ax.set_ylim(0, 105)
    grid(ax)
    ax.legend(ncol=3, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, 1.25), handlelength=1)
    save(fig, out, "attribution")


def detection(R, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    cats = ["T1", "T2", "T3", "T4"]
    for j, m in enumerate([m for m in ORDER if m != "CVP (rule)"]):
        xs = np.arange(4) + (j - 2) * 0.09
        ax.plot(xs, [R[m][f"AUC_{c}"] for c in cats], "o", ms=4.5, color=COLORS[m], label=m,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0.5, color=INK2, linestyle="--", linewidth=0.8)
    ax.text(3.45, 0.505, "chance", fontsize=6.5, color=INK2, ha="right", va="bottom")
    ax.set_xticks(range(4), ["T1 crude", "T2 blended", "T3 coherent", "T4 complete"])
    ax.set_ylabel("Detection AUC (genuine vs. Tk)")
    ax.set_ylim(0.4, 1.02)
    grid(ax)
    ax.legend(ncol=3, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, 1.22), handlelength=1)
    save(fig, out, "detection")


def robustness(rob, out):
    conds = list(rob)
    labels = [c.replace("$q{=}60$", "q=60").replace("$0.6\\times$", "0.6x") for c in conds]
    amb = any(finite(rob[c][m]["attr_amb"]) for c in conds for m in MAIN)
    key2, title2 = ("attr_amb", "Ambiguity-split attribution (%)") if amb else ("attr_all", "Attribution, all tampered (%)")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.8))
    for ax, (key, title) in zip(axes, [("AUC", "Detection AUC"), (key2, title2)]):
        for m in MAIN:
            scale = 1 if key == "AUC" else 100
            ax.plot(range(len(conds)), [scale * rob[c][m][key] for c in conds], "-o", lw=1.5, ms=4,
                    color=COLORS[m], label=m, markeredgecolor="white", markeredgewidth=0.6)
        ax.set_xticks(range(len(conds)), labels, fontsize=6.5)
        ax.set_title(title, fontsize=7.5, color=INK)
        grid(ax)
    axes[0].set_ylim(0.5, 1.01)
    axes[1].set_ylim(0, 100)
    axes[0].legend(ncol=4, fontsize=6.3, loc="upper center", bbox_to_anchor=(1.15, 1.42), handlelength=1)
    fig.subplots_adjust(wspace=0.35)
    save(fig, out, "robustness")


def regimes(G, chance, out):
    tags = [t for t in ["blind", "main", "oracle"] if t in G]
    if len(tags) < 2:
        return  # nothing to compare yet
    labels = {"blind": "Zero-shot\n(crude only)", "main": "Older generator\n(main)", "oracle": "Same generator\n(oracle)"}
    methods = ["Visual-only", "Constraint-only", "CVP"]
    amb = any(finite(G[t][m]["attr_amb_T2"]) for t in tags for m in methods)
    key2, title2 = ("attr_amb_T2", "Ambiguity split, T2 (%)") if amb else ("attr_T2", "Attribution, T2 (%)")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
    for ax, (key, title, scale) in zip(axes, [("AUC_T2", "Detection AUC on T2", 1), (key2, title2, 100)]):
        for j, m in enumerate(methods):
            xs = np.arange(len(tags)) + (j - 1) * 0.26
            ax.bar(xs, [scale * G[t][m][key] if finite(G[t][m][key]) else 0 for t in tags], 0.26, color=COLORS[m],
                   edgecolor="white", linewidth=0.8, label=m)
        ax.set_xticks(range(len(tags)), [labels[t] for t in tags], fontsize=5.8)
        ax.set_title(title, fontsize=7.5, color=INK)
        grid(ax)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 105)
    if amb and finite(chance):
        axes[1].axhline(100 * chance, color=INK2, linestyle="--", linewidth=0.8)
    axes[0].legend(ncol=3, fontsize=6.3, loc="upper center", bbox_to_anchor=(1.15, 1.4), handlelength=1)
    fig.subplots_adjust(wspace=0.3)
    save(fig, out, "regimes")


def family_label(S, fam):
    if fam[:1] == "c" and fam[1:].isdigit():
        return E.tex_family(fam)
    copies = sum((c.family or c.name) == fam for c in S.constraints)
    return fam + ("(·)" if copies > 1 else "")


def signatures(sig, N, name, out):
    E.set_doctype(name)
    S = E.S
    rows = [label for _, _, label in E.signature_rows() if label in sig]
    if not rows:
        return
    M = np.array([sig[label] for label in rows])
    nr, nc = M.shape
    fs = 6.5 if nr <= 10 and nc <= 8 else 5.5
    fig, ax = plt.subplots(figsize=(max(3.4, 0.3 * nc + 1.9), max(2.2, 0.19 * nr + 0.9)))
    ax.imshow(M, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(nr):
        for j in range(nc):
            if M[i, j] > 0:  # zero cells stay blank: the pattern is the point
                ax.text(j, i, f"{M[i, j]:.2f}".rstrip("0").rstrip("."), ha="center", va="center", fontsize=fs,
                        color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(nc), [family_label(S, f) for f in S.families], fontsize=fs + 0.5,
                  rotation=0 if nc <= 8 else 90)
    ax.set_yticks(range(nr), [f"{r} (n={N[r]})" for r in rows], fontsize=fs + 0.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Empirical violation frequency, single-field edits", fontsize=7.5)
    save(fig, out, "signatures")


# ------------------------------------------------------------------ per type, from the models
def overlay(img, res):
    """The app's evidence view: red heat, thin orange = fields sharing the violations,
    orange = the nominated class, dark red = the attributed field."""
    a = (np.clip(res["heatmap"], 0, 1) ** 1.5 * 0.7)[..., None]
    red = np.zeros_like(img, np.float32)
    red[..., 0] = 230
    im = Image.fromarray((img * (1 - a) + red * a).astype(np.uint8))
    d = ImageDraw.Draw(im)
    boxes = res["ocr"]["boxes"]
    if res["cls"] != "coherent":
        for f in res["consistent"]:
            if f not in res["candidates"]:
                x0, y0, x1, y1 = boxes[f]
                d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=(230, 150, 0), width=1)
        for f in res["candidates"]:
            x0, y0, x1, y1 = boxes[f]
            d.rectangle([x0 - 5, y0 - 5, x1 + 5, y1 + 5], outline=(230, 150, 0), width=2)
    if res["tampered"]:
        x0, y0, x1, y1 = boxes[res["field"]]
        d.rectangle([x0 - 9, y0 - 9, x1 + 9, y1 + 9], outline=(200, 0, 0), width=3)
    return im


def example_panels(system, DT, field, ambiguous, start, tries):
    """First seed from `start` where a T2 and a T1 edit of `field` are both attributed
    correctly (and, if asked, the T2 one leaves several suspects)."""
    for seed in range(start, start + tries):
        panels = []
        for cat in ("T2", "T1"):
            rng = np.random.default_rng(seed)
            doc = render.generate(DT, rng)
            img, info = tamper.apply(doc, DT, cat, rng, field=field)
            res = system(img)
            if not (res["tampered"] and res["field"] in info["edited"]):
                break
            if cat == "T2" and ambiguous and len(res["consistent"]) < 2:
                break
            panels.append((cat, img, info, res))
        else:
            return seed, panels[::-1]
    return None, []


def crop_box(panels, field, page, margin=30):
    """The full page width (labels sit left of their values) and the rows around every
    field the two panels point to, at least a quarter of the page high."""
    W, H = page
    spans = []
    for _, _, _, res in panels:
        names = {field, res["field"], *res["candidates"]}
        if len(res["consistent"]) <= 6:
            names |= set(res["consistent"])
        spans += [res["ocr"]["boxes"][f][1::2] for f in names]
    s = np.array(spans)
    y0, y1 = s[:, 0].min() - margin, s[:, 1].max() + margin
    need = 0.25 * H - (y1 - y0)
    if need > 0:
        y0, y1 = y0 - need / 2, y1 + need / 2
    shift = max(0, -y0) - max(0, y1 - H)
    return 0, int(max(0, y0 + shift)), W, int(min(H, y1 + shift))


def short_list(names, k=5):
    return ", ".join(names[:k]) + (", ..." if len(names) > k else "")


def suspects(res, label):
    if res["cls"] == "coherent":
        return "no rule broken: every field is a suspect"
    names = [label[f] for f in res["consistent"]]
    return f"suspects: {', '.join(names)}" if len(names) <= 3 else f"{len(names)} suspect fields (orange)"


def examples(name, out, tries=100):
    DT = doctypes.get(name)
    S = DT.schema
    if name not in EXAMPLE:
        return
    field, ambiguous, start = EXAMPLE[name]
    seed, panels = example_panels(pipeline.CVP(name), DT, field, ambiguous, start, tries)
    if seed is None:
        print(f"{name}: no example among seeds {start}..{start + tries - 1}; skipped", flush=True)
        return
    label = {f.name: f.label for f in S.fields}
    fmt = {f.name: f.fmt for f in S.fields}
    box = crop_box(panels, field, DT.page)
    w, h = box[2] - box[0], box[3] - box[1]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 2 * 3.45 * h / w + 1.1))
    kind = {"T1": "crude", "T2": "blended"}
    for r, (cat, img, info, res) in enumerate(panels):
        axes[r, 0].imshow(Image.fromarray(img).crop(box))
        axes[r, 1].imshow(overlay(img, res).crop(box))
        violated = sorted(S.violated(res["w"]), key=S.c_index.get)
        axes[r, 0].set_title(f"{cat} {kind[cat]} edit: {label[field]} changed to "
                             f"'{fmt[field](info['values'][field])}'\ndocument as received (seed {seed})", fontsize=7)
        axes[r, 1].set_title(f"violated {{{short_list(violated)}}}; {suspects(res, label)}\n"
                             f"attributed: {label[res['field']]} (red)", fontsize=7)
        for ax in axes[r]:
            ax.axis("off")
    fig.subplots_adjust(wspace=0.03, hspace=0.35)
    save(fig, out, "examples")


def page_example(name, out):
    doc = render.generate(doctypes.get(name), np.random.default_rng(3))
    os.makedirs(out, exist_ok=True)
    Image.fromarray(doc.image).save(os.path.join(out, "page.png"))


# ------------------------------------------------------------------ across types
def cross_type(summaries):
    names = list(summaries)
    x = np.arange(len(names))
    n, bw = len(MAIN), 0.19
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
    # Detection AUC as dots (an AUC axis does not start at zero, so no bars)
    ax = axes[0]
    for j, m in enumerate(MAIN):
        ax.plot(x + (j - (n - 1) / 2) * bw, [summaries[t]["results"][m]["AUC"] for t in names], "o", ms=4,
                color=COLORS[m], markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0.5, color=INK2, linestyle="--", linewidth=0.8)
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Detection AUC, all tampered", fontsize=7.5, color=INK)
    for ax, key, title in [(axes[1], "attr_all", "Attribution, all tampered (%)"),
                           (axes[2], "attr_amb", "Ambiguity split, T1/T2 (%)")]:
        for j, m in enumerate(MAIN):
            vals = [summaries[t]["results"][m][key] for t in names]
            ax.bar(x + (j - (n - 1) / 2) * bw, [100 * v if finite(v) else 0 for v in vals], bw, color=COLORS[m],
                   edgecolor="white", linewidth=0.6, label=m)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=7.5, color=INK)
    for i, t in enumerate(names):
        c = summaries[t].get("chance_ambiguous")
        if finite(c):
            axes[2].hlines(100 * c, i - 0.45, i + 0.45, colors=INK2, linestyles="--", linewidth=0.8,
                           label="chance 1/|C(V)|" if i == 0 else None)
        else:
            axes[2].text(i, 3, "no ambiguous\nedits", ha="center", va="bottom", fontsize=5.5, color=INK2, rotation=90)
    for ax in axes:
        ax.set_xticks(x, [SHORT.get(t, t) for t in names], rotation=35, ha="right", fontsize=6.5)
        grid(ax)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, 1.1), handlelength=1)
    fig.subplots_adjust(wspace=0.3)
    save(fig, FIG, "cross_type")


def gallery(names):
    cols = 3
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.5 * rows))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, name in zip(np.atleast_1d(axes).flat, names):
        ax.imshow(render.generate(doctypes.get(name), np.random.default_rng(3)).image)
        ax.set_anchor("N")  # pages differ in shape: align their tops so the titles line up
        ax.set_title(doctypes.get(name).title, fontsize=7.5, color=INK)
    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    save(fig, FIG, "doctypes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctypes", default="", help="comma-separated; default every type with results")
    ap.add_argument("--no-examples", action="store_true", help="skip the panels that run the trained models")
    ap.add_argument("--tries", type=int, default=100, help="seeds scanned for each qualitative example")
    a = ap.parse_args()
    names = a.doctypes.split(",") if a.doctypes else doctypes.names()
    summaries = {}
    for name in names:
        path = os.path.join(paths.results_dir(name), "results.json")
        if not os.path.exists(path):
            print(f"{name}: no results yet, skipped", flush=True)
            continue
        S = json.load(open(path))
        summaries[name] = S
        out = os.path.join(FIG, name)
        chance = S.get("chance_ambiguous")
        attribution(S["results"], chance, out)
        detection(S["results"], out)
        robustness(S["robustness"], out)
        signatures(S["signatures"], S["signature_n"], name, out)
        regimes(S.get("heldout") or {}, chance, out)
        page_example(name, out)
        if not a.no_examples and os.path.exists(pipeline.heads_path(name)):
            examples(name, out, a.tries)
        print(f"{name}: figures in {out}", flush=True)
    if len(summaries) > 1:
        cross_type(summaries)
    gallery(doctypes.names())  # always every type: it shows what the system covers
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
