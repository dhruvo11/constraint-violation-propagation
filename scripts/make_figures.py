"""Figures for the report (PDF for LaTeX, PNG for the demo) from results/results.json."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvp import doctypes, paths, pipeline  # noqa: E402
from cvp.core import render, tamper  # noqa: E402

FIG = os.path.join(paths.REPORT, "figures")
DT = doctypes.get("transcript")
SCHEMA = DT.schema
# Fixed categorical order (reference palette, light mode); colour follows the method.
COLORS = {"CVP": "#2a78d6", "Visual-only": "#eb6834", "Constraint-only": "#1baf7a",
          "Concat fusion": "#eda100", "ELA": "#e87ba4", "CVP (rule)": "#6250d6"}
ORDER = ["ELA", "Visual-only", "Constraint-only", "Concat fusion", "CVP (rule)", "CVP"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=160)
    plt.close(fig)


def grid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def attribution(R):
    groups = [("attr_all", "All tampered\n(T1-T4)"), ("attr_unamb", "Unambiguous\n(T1/T2)"),
              ("attr_amb", "Ambiguity split\n(T1/T2)")]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    n, bw = len(ORDER), 0.13
    for j, m in enumerate(ORDER):
        xs = np.arange(len(groups)) + (j - (n - 1) / 2) * bw
        ax.bar(xs, [100 * R[m][k] for k, _ in groups], bw, color=COLORS[m], edgecolor="white",
               linewidth=0.8, label=m)
    ax.hlines(20, 2 - 0.45, 2 + 0.45, colors=INK2, linestyles="--", linewidth=0.8)
    ax.text(2 + 0.47, 20, "chance\n1/|K|", va="center", ha="left", fontsize=6.5, color=INK2)
    v = 100 * R["CVP"]["attr_amb"]
    ax.text(2 + (n - 1) / 2 * bw, v + 2, f"{v:.0f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(groups)), [g for _, g in groups])
    ax.set_ylabel("Top-1 field attribution (%)")
    ax.set_ylim(0, 105)
    grid(ax)
    ax.legend(ncol=3, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, 1.25), handlelength=1)
    save(fig, "attribution")


def detection(R):
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
    save(fig, "detection")


def robustness(rob):
    conds = list(rob)
    labels = [c.replace("$q{=}60$", "q=60").replace("$0.6\\times$", "0.6x") for c in conds]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.8))
    for ax, (key, title) in zip(axes, [("AUC", "Detection AUC"), ("attr_amb", "Ambiguity-split attribution (%)")]):
        for m in ["Visual-only", "Constraint-only", "Concat fusion", "CVP"]:
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
    save(fig, "robustness")


def regimes(G):
    tags = [t for t in ["blind", "main", "oracle"] if t in G]
    labels = {"blind": "Zero-shot\n(crude only)", "main": "Older generator\n(main)", "oracle": "Same generator\n(oracle)"}
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
    methods = ["Visual-only", "Constraint-only", "CVP"]
    for ax, (key, title, scale) in zip(axes, [("AUC_T2", "Detection AUC on T2", 1), ("attr_amb_T2", "Ambiguity split, T2 (%)", 100)]):
        for j, m in enumerate(methods):
            xs = np.arange(len(tags)) + (j - 1) * 0.26
            ax.bar(xs, [scale * G[t][m][key] for t in tags], 0.26, color=COLORS[m], edgecolor="white", linewidth=0.8,
                   label=m)
        ax.set_xticks(range(len(tags)), [labels[t] for t in tags], fontsize=5.8)
        ax.set_title(title, fontsize=7.5, color=INK)
        grid(ax)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 105)
    axes[1].axhline(20, color=INK2, linestyle="--", linewidth=0.8)
    axes[0].legend(ncol=3, fontsize=6.3, loc="upper center", bbox_to_anchor=(1.15, 1.4), handlelength=1)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "regimes")


def signatures(S, N):
    import evaluate as E
    E.set_doctype(DT.name)
    rows = [(label, u) for _, u, label in E.signature_rows() if label in S]
    order = [label for label, _ in rows]
    M = np.array([S[label] for label, _ in rows])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95"])
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}".rstrip("0").rstrip(".") if M[i, j] > 0 else "0",
                    ha="center", va="center", fontsize=6.5, color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(SCHEMA.families)), [E.tex_family(f) for f in SCHEMA.families])
    ax.set_yticks(range(len(order)), [f"{o} (n={N[o]})" for o in order], fontsize=6.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Empirical violation frequency, single-field edits", fontsize=7.5)
    save(fig, "signatures")


def examples():
    """Qualitative: the same transcript with a T1 and a T2 edit of subject mark 3."""
    system = pipeline.CVP(DT)
    panels = []
    for cat in ["T1", "T2"]:
        rng = np.random.default_rng(11)
        doc = render.generate(DT, rng)
        img, info = tamper.apply(doc, DT, cat, rng, field="m3")
        res = system(img)
        H = res["heatmap"]
        a = (np.clip(H, 0, 1) ** 1.5 * 0.7)[..., None]
        red = np.zeros_like(img, np.float32)
        red[..., 0] = 230
        im = Image.fromarray((img * (1 - a) + red * a).astype(np.uint8))
        d = ImageDraw.Draw(im)
        for f in res["candidates"]:
            x0, y0, x1, y1 = res["ocr"]["boxes"][f]
            d.rectangle([x0 - 5, y0 - 5, x1 + 5, y1 + 5], outline=(230, 150, 0), width=2)
        x0, y0, x1, y1 = res["ocr"]["boxes"][res["field"]]
        d.rectangle([x0 - 9, y0 - 9, x1 + 9, y1 + 9], outline=(200, 0, 0), width=3)
        ox, oy = doc.style.ox, doc.style.oy
        crop = im.crop((30 + ox, 262 + oy, 610 + ox, 640 + oy))
        orig = Image.fromarray(img).crop((30 + ox, 262 + oy, 610 + ox, 640 + oy))
        panels.append((cat, orig, crop, res, info))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    for r, (cat, orig, crop, res, info) in enumerate(panels):
        axes[r, 0].imshow(orig)
        axes[r, 1].imshow(crop)
        sig = ", ".join(sorted(SCHEMA.violated(res["w"]), key=SCHEMA.c_index.get))
        axes[r, 0].set_title(f"{cat}: mark 3 edited to {info['values']['m3']} (document as received)", fontsize=7.5)
        axes[r, 1].set_title(f"violated {{{sig}}} -> class '{res['cls']}' (orange); heatmap; "
                             f"attributed: {res['field']}", fontsize=7.5)
        for ax in axes[r]:
            ax.axis("off")
    fig.subplots_adjust(wspace=0.03, hspace=0.12)
    save(fig, "examples")


def page_example():
    rng = np.random.default_rng(3)
    doc = render.generate(DT, rng)
    Image.fromarray(doc.image).save(os.path.join(FIG, "transcript.png"))


def main():
    S = json.load(open(os.path.join(paths.results_dir(DT.name), "results.json")))
    attribution(S["results"])
    detection(S["results"])
    robustness(S["robustness"])
    signatures(S["signatures"], S["signature_n"])
    if S.get("heldout"):
        regimes(S["heldout"])
    examples()
    page_example()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
