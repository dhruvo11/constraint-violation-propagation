"""Interactive CVP demo: forge a document of any supported type (or upload one) and watch
the two branches work."""
import json
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # works offline; nothing is reported anywhere

import gradio as gr  # noqa: E402
import numpy as np
from PIL import Image, ImageDraw

from cvp import doctypes, paths, pipeline
from cvp.core import render, tamper
from cvp.real import demo as real_demo

BASE = os.path.dirname(os.path.abspath(__file__))
TYPES = {doctypes.get(n).title: n for n in doctypes.names()}
CAT_LABELS = {"genuine": "Genuine (no edit)", "T1": "T1 - crude edit", "T2": "T2 - blended (inpainting-like)",
              "T3": "T3 - coherent crude edit", "T4": "T4 - coherent blended edit"}
RANDOM = "random"
# One illustrative case per type: (type, seed, forgery, field to edit)
# The scenarios of DEMO.md, in its order; each was checked against the trained system.
EXAMPLES = [("transcript", 2, "T2", "m3"), ("transcript", 3, "T1", "m2"), ("transcript", 5, "T2", "total"),
            ("transcript", 4, "T3", RANDOM), ("transcript", 8, "T4", RANDOM), ("transcript", 9, "genuine", RANDOM),
            ("invoice", 3, "T2", "qty2"), ("invoice", 4, "T2", "total"), ("invoice", 5, "T3", RANDOM),
            ("receipt", 4, "T2", "change"), ("receipt", 5, "T1", "total"),
            ("certificate", 6, "T2", "grade"), ("certificate", 8, "T4", RANDOM),
            ("passport", 6, "T2", "viz_dob"), ("passport", 7, "T1", "mrz2"),
            ("idcard", 7, "T2", "valid"), ("idcard", 9, "genuine", RANDOM)]
_SYSTEMS = {}


def system(name):
    if name not in _SYSTEMS:
        _SYSTEMS[name] = pipeline.CVP(name)
    return _SYSTEMS[name]


def labels(name):
    return {f.name: f.label for f in doctypes.get(name).schema.fields}


def editable(name):
    """Fields a single-field (T1/T2) edit can target: those at least one rule reads."""
    S = doctypes.get(name).schema
    return [f for f in S.field_names if S.constrained(f)]


def overlay(arr, res, truth=None):
    H = res["heatmap"]
    heat = np.zeros_like(arr, np.float32)
    heat[..., 0] = 255
    a = (np.clip(H, 0, 1) ** 1.5 * 0.65)[..., None]
    img = Image.fromarray((arr * (1 - a) + heat * a).astype(np.uint8))
    d = ImageDraw.Draw(img)
    boxes = res["ocr"]["boxes"]
    if res["cls"] != "coherent":
        for f in res["consistent"]:  # fields that share the signature: thin orange
            if f not in res["candidates"]:
                x0, y0, x1, y1 = boxes[f]
                d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=(230, 160, 0), width=1)
        for f in res["candidates"]:
            x0, y0, x1, y1 = boxes[f]
            d.rectangle([x0 - 5, y0 - 5, x1 + 5, y1 + 5], outline=(230, 160, 0), width=2)
    if truth:
        for r in truth:
            d.rectangle(r, outline=(0, 160, 60), width=1)
    if res["tampered"]:
        x0, y0, x1, y1 = boxes[res["field"]]
        d.rectangle([x0 - 8, y0 - 8, x1 + 8, y1 + 8], outline=(220, 0, 0), width=3)
    return img


def report(name, res, info=None):
    S = doctypes.get(name).schema
    names = labels(name)
    n = len(S.field_names)
    w, r = res["w"], res["r"]
    o = res["ocr"]
    lines = []
    verdict = "TAMPERED" if res["tampered"] else "GENUINE"
    lines.append(f"### Verdict: **{verdict}**  (P(tampered) = {res['p_tamper']:.3f})")
    if res["tampered"]:
        lines.append(f"**Attributed field:** {names[res['field']]} (red box)")
    violated = sorted(S.violated(w), key=S.c_index.get)
    if res["cls"] == "coherent":
        lines.append(f"**Constraint branch:** every constraint holds, so the decision rests on the visual "
                     f"branch alone (all {n} fields are candidates).")
    else:
        lines.append(f"**Constraint branch:** violated {{{', '.join(violated)}}} -> class `{res['cls']}`, which "
                     f"narrows {n} fields to {len(res['candidates'])} candidate(s): "
                     f"{', '.join(names[f] for f in res['candidates'])} (orange boxes).")
        if len(res["candidates"]) > 1:
            lines.append("These fields are indistinguishable to constraint reasoning (Proposition 1); the "
                         "visual heatmap ranks them.")
        others = [f for f in res["consistent"] if f not in res["candidates"]]
        if others:
            lines.append(f"The same violations could also come from an edit of "
                         f"{', '.join(names[f] for f in others)} (thin orange boxes): those fields share this "
                         f"signature, so the visual branch has to decide.")
    if info is not None:
        ok = (res["field"] in info["edited"]) if (res["tampered"] and info["edited"]) else \
            (not info["edited"] and not res["tampered"])
        lines.append(f"**Ground truth:** {CAT_LABELS[info['category']]}; edited "
                     f"{', '.join(names[f] for f in info['edited']) or 'nothing'} (green boxes) - "
                     f"{'CORRECT' if ok else 'MISSED'}")
    lines = ["\n\n".join(lines)]
    lines.append("\n| Constraint | Predicate | Status | Residual |\n|---|---|---|---|")
    for j, c in enumerate(S.constraints):
        lines.append(f"| {c.name} | {c.text} | {'VIOLATED' if w[j] else 'ok'} | {r[j]:+.3f} |")
    lines.append("\n| Field | OCR read | conf. | max heat |\n|---|---|---|---|")
    for i, f in enumerate(S.field_names):
        lines.append(f"| {names[f]} | `{o['text'][f]}` | {o['conf'][f]:.2f} | {res['pooled'][i, 0]:.2f} |")
    return "\n".join(lines)


def class_posterior(name, res):
    return {k: float(p) for k, p in zip(doctypes.get(name).schema.class_names, res["p_cls"])}


def missing(name):
    return not os.path.exists(pipeline.heads_path(name))


NOT_READY = ("The models for **{}** are still being trained (the heads file `models/heads_{}.pkl` does not "
             "exist yet). Run `./scripts/reproduce.sh` or wait for the running pipeline to finish.")


def forge(type_title, seed, category, field):
    name = TYPES[type_title]
    if missing(name):
        return None, None, NOT_READY.format(type_title, name), {}
    DT = doctypes.get(name)
    rng = np.random.default_rng(int(seed))
    doc = render.generate(DT, rng)
    cat = next(k for k, v in CAT_LABELS.items() if v == category)
    by_label = {v: k for k, v in labels(name).items()}
    f = by_label.get(field)
    f = f if (f in editable(name) and cat in ("T1", "T2")) else None
    img, info = tamper.apply(doc, DT, cat, rng, field=f)
    res = system(name)(img)
    return (Image.fromarray(img), overlay(img, res, info["rects"]), report(name, res, info),
            class_posterior(name, res))


def field_choices(type_title):
    name = TYPES[type_title]
    return gr.update(choices=[RANDOM] + [labels(name)[f] for f in editable(name)], value=RANDOM)


def analyse(type_title, upload):
    name = TYPES[type_title]
    if upload is None:
        return None, "Upload an image first.", {}
    if missing(name):
        return None, NOT_READY.format(type_title, name), {}
    DT = doctypes.get(name)
    arr = np.asarray(Image.fromarray(upload).convert("RGB").resize(DT.page))
    res = system(name)(arr)
    return overlay(arr, res), report(name, res), class_posterior(name, res)


def results_md(type_title):
    name = TYPES[type_title]
    path = os.path.join(paths.results_dir(name), "results.json")
    if not os.path.exists(path):
        return f"No results for {type_title} yet: run `scripts/evaluate.py --doctype {name}`."
    summary = json.load(open(path))
    R = summary["results"]

    def pct(x):
        return "--" if x is None or x != x else f"{100 * x:.1f}%"
    rows = [f"**{type_title}**: {summary['n_ambiguous']} ambiguous test edits "
            f"(chance {pct(summary.get('chance_ambiguous'))}); OCR field accuracy "
            f"{pct(summary['ocr_acc'].get('Clean'))}.\n",
            "| Method | AUC | F1 | Attr. all | Unambiguous | Ambiguity split |", "|---|---|---|---|---|---|"]
    for m, r in R.items():
        rows.append(f"| {m} | {r['AUC']:.3f} | {r['F1']:.3f} | {pct(r['attr_all'])} | "
                    f"{pct(r['attr_unamb'])} | {pct(r['attr_amb'])} |")
    return "\n".join(rows)


FIGS = ("attribution", "regimes", "detection", "robustness", "signatures", "examples")


def figures(type_title):
    """The selected type's figures, then the comparison across types (scripts/make_figures.py)."""
    root = os.path.join(BASE, "report", "figures")
    out = [os.path.join(root, TYPES[type_title], f + ".png") for f in FIGS]
    out += [os.path.join(root, f + ".png") for f in ("cross_type", "doctypes")]
    return [f for f in out if os.path.exists(f)]


def results_tab(type_title):
    return results_md(type_title), figures(type_title)


def example_rows():
    out = []
    for name, seed, cat, field in EXAMPLES:
        title = doctypes.get(name).title
        out.append([title, seed, CAT_LABELS[cat], RANDOM if field == RANDOM else labels(name)[field]])
    return out


with gr.Blocks(title="CVP - Constraint Violation Propagation") as demo:
    gr.Markdown("# Constraint Violation Propagation\nAttributing document forgery when visual evidence "
                "fails. The **constraint branch** reads the document (OCR), checks its arithmetic and logic "
                "and narrows the suspect fields; the **visual branch** (ResNet-18 U-Net on RGB + ELA) "
                "produces a tamper heatmap that decides among the fields the rules cannot tell apart.")
    first = doctypes.get("transcript").title
    with gr.Tab("Forge & analyse"):
        with gr.Row():
            with gr.Column(scale=1):
                dtype = gr.Dropdown(list(TYPES), value=first, label="Document type")
                seed = gr.Number(value=2, label="Document seed", precision=0)
                cat = gr.Dropdown(list(CAT_LABELS.values()), value=CAT_LABELS["T2"], label="Forgery type")
                fld = gr.Dropdown([RANDOM] + [labels("transcript")[f] for f in editable("transcript")],
                                  value=labels("transcript")["m3"], label="Field to edit (T1/T2)",
                                  allow_custom_value=True)
                btn = gr.Button("Forge and analyse", variant="primary")
                gr.Examples(examples=example_rows(), inputs=[dtype, seed, cat, fld],
                            label="Scenarios (click one, then Forge and analyse)")
                post = gr.Label(label="Constraint-branch class posterior p(K)", num_top_classes=4)
            with gr.Column(scale=2):
                with gr.Row():
                    raw = gr.Image(label="Document as received", type="pil")
                    out = gr.Image(label="CVP evidence (heatmap, candidates, attribution)", type="pil")
                rep = gr.Markdown()
        dtype.input(field_choices, [dtype], [fld])
        btn.click(forge, [dtype, seed, cat, fld], [raw, out, rep, post])
        demo.load(forge, [dtype, seed, cat, fld], [raw, out, rep, post])
    with gr.Tab("Analyse an image"):
        with gr.Row():
            with gr.Column(scale=1):
                dtype2 = gr.Dropdown(list(TYPES), value=first, label="Document type")
                up = gr.Image(label="Upload a document rendered from this template", type="numpy")
            out2 = gr.Image(label="CVP evidence", type="pil")
        rep2 = gr.Markdown()
        post2 = gr.Label(label="p(K)")
        gr.Button("Analyse", variant="primary").click(analyse, [dtype2, up], [out2, rep2, post2])
    with gr.Tab("Real receipt"):
        gr.Markdown("A real photographed receipt from the CORD v2 test set (Indonesian shops; CC BY 4.0). Forge "
                    "it, or leave it genuine: CVP reads its values with a real OCR (docTR), checks the receipt's "
                    "own arithmetic, and runs the visual branch on the photo.")
        why = real_demo.unavailable()
        if why:
            gr.Markdown(f"**Not available on this machine:** {why}")
        else:
            with gr.Row():
                with gr.Column(scale=1):
                    rk = gr.Number(value=7, label=f"Receipt (0-{real_demo.count() - 1})", precision=0)
                    rcat = gr.Dropdown(list(real_demo.CAT_LABELS.values()), value=real_demo.CAT_LABELS["T2"],
                                       label="Forgery type")
                    rfld = gr.Dropdown(real_demo.field_choices(7), value=real_demo.RANDOM,
                                       label="Field to edit (T1/T2)", allow_custom_value=True)
                    rbtn = gr.Button("Forge and analyse", variant="primary")
                    gr.Markdown("The first analysis loads the OCR and the visual branch (about 15 s).")
                with gr.Column(scale=2):
                    with gr.Row():
                        rraw = gr.Image(label="Photo as received", type="pil")
                        rout = gr.Image(label="CVP evidence (heat; orange: fields the broken rules fit; red: "
                                              "attributed; green: truth)", type="pil")
                    rrep = gr.Markdown()
            rk.change(lambda k: gr.update(choices=real_demo.field_choices(k), value=real_demo.RANDOM), [rk], [rfld])
            rbtn.click(real_demo.analyse, [rk, rcat, rfld], [rraw, rout, rrep])
    with gr.Tab("Results"):
        dtype3 = gr.Dropdown(list(TYPES), value=first, label="Document type")
        res_md = gr.Markdown(results_md(first))
        gal = gr.Gallery(figures(first), columns=3, height=360)
        dtype3.change(results_tab, [dtype3], [res_md, gal])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)))
