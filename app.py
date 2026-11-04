"""Interactive CVP demo: forge a transcript (or upload one) and watch the two branches work."""
import json
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # works offline; nothing is reported anywhere

import gradio as gr  # noqa: E402
import numpy as np
from PIL import Image, ImageDraw

from cvp import doctypes, paths, pipeline
from cvp.core import render, tamper

BASE = os.path.dirname(os.path.abspath(__file__))
DT = doctypes.get("transcript")
SCHEMA = DT.schema
SYSTEM = pipeline.CVP(DT)
FIELD_NAMES = {f.name: f.label for f in SCHEMA.fields}
CAT_LABELS = {"genuine": "Genuine (no edit)", "T1": "T1 - crude edit", "T2": "T2 - blended (inpainting-like)",
              "T3": "T3 - coherent crude edit", "T4": "T4 - coherent blended edit"}


def overlay(arr, res, truth=None):
    H = res["heatmap"]
    heat = np.zeros_like(arr, np.float32)
    heat[..., 0] = 255
    a = (np.clip(H, 0, 1) ** 1.5 * 0.65)[..., None]
    img = Image.fromarray((arr * (1 - a) + heat * a).astype(np.uint8))
    d = ImageDraw.Draw(img)
    boxes = res["ocr"]["boxes"]
    for f in res["candidates"]:
        if res["cls"] != "coherent":
            x0, y0, x1, y1 = boxes[f]
            d.rectangle([x0 - 5, y0 - 5, x1 + 5, y1 + 5], outline=(230, 160, 0), width=2)
    if truth:
        for r in truth:
            d.rectangle(r, outline=(0, 160, 60), width=1)
    if res["tampered"]:
        x0, y0, x1, y1 = boxes[res["field"]]
        d.rectangle([x0 - 8, y0 - 8, x1 + 8, y1 + 8], outline=(220, 0, 0), width=3)
    return img


def report(res, info=None):
    w, r = res["w"], res["r"]
    o = res["ocr"]
    lines = []
    verdict = "TAMPERED" if res["tampered"] else "GENUINE"
    lines.append(f"### Verdict: **{verdict}**  (P(tampered) = {res['p_tamper']:.3f})")
    if res["tampered"]:
        lines.append(f"**Attributed field:** {FIELD_NAMES[res['field']]} (red box)")
    violated = sorted(SCHEMA.violated(w), key=SCHEMA.c_index.get)
    if res["cls"] == "coherent":
        lines.append("**Constraint branch:** every constraint holds, so the decision rests on the visual "
                     "branch alone (all 10 fields are candidates).")
    else:
        lines.append(f"**Constraint branch:** violated {{{', '.join(violated)}}} -> class `{res['cls']}`, which "
                     f"narrows 10 fields to {len(res['candidates'])} candidate(s): "
                     f"{', '.join(FIELD_NAMES[f] for f in res['candidates'])} (orange boxes).")
        if len(res["candidates"]) > 1:
            lines.append("These fields are indistinguishable to constraint reasoning (Proposition 1); the "
                         "visual heatmap ranks them.")
    if info is not None:
        ok = (res["field"] in info["edited"]) if (res["tampered"] and info["edited"]) else (not info["edited"] and not res["tampered"])
        lines.append(f"**Ground truth:** {CAT_LABELS[info['category']]}; edited "
                     f"{', '.join(FIELD_NAMES[f] for f in info['edited']) or 'nothing'} (green boxes) - "
                     f"{'CORRECT' if ok else 'MISSED'}")
    lines = ["\n\n".join(lines)]
    lines.append("\n| Constraint | Predicate | Status | Residual |\n|---|---|---|---|")
    for j, c in enumerate(SCHEMA.constraints):
        lines.append(f"| {c.name} | {c.text} | {'VIOLATED' if w[j] else 'ok'} | {r[j]:+.3f} |")
    lines.append("\n| Field | OCR read | conf. | max heat |\n|---|---|---|---|")
    for i, f in enumerate(SCHEMA.field_names):
        lines.append(f"| {FIELD_NAMES[f]} | `{o['text'][f]}` | {o['conf'][f]:.2f} | {res['pooled'][i, 0]:.2f} |")
    return "\n".join(lines)


def class_posterior(res):
    return {k: float(p) for k, p in zip(SCHEMA.class_names, res["p_cls"])}


def forge(seed, category, field):
    rng = np.random.default_rng(int(seed))
    doc = render.generate(DT, rng)
    cat = next(k for k, v in CAT_LABELS.items() if v == category)
    f = None if field == "random" else next(k for k, v in FIELD_NAMES.items() if v == field)
    img, info = tamper.apply(doc, DT, cat, rng, field=f if cat in ("T1", "T2") else None)
    res = SYSTEM(img)
    return (Image.fromarray(img), overlay(img, res, info["rects"]), report(res, info), class_posterior(res))


def analyse(upload):
    arr = np.asarray(Image.fromarray(upload).convert("RGB").resize(DT.page))
    res = SYSTEM(arr)
    return overlay(arr, res), report(res), class_posterior(res)


def results_md():
    path = os.path.join(paths.results_dir(DT.name), "results.json")
    if not os.path.exists(path):
        return "Run `scripts/evaluate.py` first."
    R = json.load(open(path))["results"]
    rows = ["| Method | AUC | F1 | Attr. all | Unambiguous | Ambiguity split |", "|---|---|---|---|---|---|"]
    for m, r in R.items():
        rows.append(f"| {m} | {r['AUC']:.3f} | {r['F1']:.3f} | {100 * r['attr_all']:.1f}% | "
                    f"{100 * r['attr_unamb']:.1f}% | {100 * r['attr_amb']:.1f}% |")
    return "\n".join(rows)


with gr.Blocks(title="CVP - Constraint Violation Propagation") as demo:
    gr.Markdown("# Constraint Violation Propagation\nAttributing document forgery when visual evidence "
                "fails. The **constraint branch** reads the transcript (OCR), checks its arithmetic and logic "
                "and narrows the suspect fields to an equivalence class; the **visual branch** (ResNet-18 "
                "U-Net on RGB + ELA) produces a tamper heatmap that resolves which field inside the class "
                "was edited.")
    with gr.Tab("Forge & analyse"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Number(value=2, label="Document seed", precision=0)
                cat = gr.Dropdown(list(CAT_LABELS.values()), value=CAT_LABELS["T2"], label="Forgery type")
                fld = gr.Dropdown(["random"] + list(FIELD_NAMES.values()), value=FIELD_NAMES["m3"],
                                  label="Field to edit (T1/T2)")
                btn = gr.Button("Forge and analyse", variant="primary")
                gr.Examples(
                    examples=[[2, CAT_LABELS["T2"], FIELD_NAMES["m3"]], [3, CAT_LABELS["T1"], FIELD_NAMES["m2"]],
                              [5, CAT_LABELS["T2"], FIELD_NAMES["total"]], [4, CAT_LABELS["T3"], "random"],
                              [8, CAT_LABELS["T4"], "random"], [9, CAT_LABELS["genuine"], "random"]],
                    inputs=[seed, cat, fld], label="Scenarios (click, then Forge and analyse)")
                post = gr.Label(label="Constraint-branch class posterior p(K)", num_top_classes=4)
            with gr.Column(scale=2):
                with gr.Row():
                    raw = gr.Image(label="Document as received", type="pil")
                    out = gr.Image(label="CVP evidence (heatmap, candidates, attribution)", type="pil")
                rep = gr.Markdown()
        btn.click(forge, [seed, cat, fld], [raw, out, rep, post])
        demo.load(forge, [seed, cat, fld], [raw, out, rep, post])
    with gr.Tab("Analyse an image"):
        with gr.Row():
            up = gr.Image(label="Upload a transcript rendered from this template", type="numpy")
            out2 = gr.Image(label="CVP evidence", type="pil")
        rep2 = gr.Markdown()
        post2 = gr.Label(label="p(K)")
        gr.Button("Analyse", variant="primary").click(analyse, [up], [out2, rep2, post2])
    with gr.Tab("Results"):
        gr.Markdown(results_md())
        figs = [os.path.join(BASE, "report", "figures", f) for f in
                ("attribution.png", "regimes.png", "detection.png", "robustness.png", "signatures.png", "examples.png")]
        gr.Gallery([f for f in figs if os.path.exists(f)], columns=3, height=360)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)))
