# CVP — Constraint Violation Propagation

Attributing document forgery when visual evidence fails. Full implementation of the
CVP architecture from our research paper (Harshith Rudresh, Dhruv S P): a synthetic
transcript corpus with per-field ground truth, a four-way tampering taxonomy (crude,
inpainting-like blended, coherent, complete), a learned OCR, a symbolic + learned
constraint branch, a ResNet-18 U-Net visual branch over RGB + ELA, class-conditioned
attribution, five baselines, and an interactive demo.

```
cvp/            library
  schema.py     fields F, constraint system C, beta bands, equivalence classes, rho
  render.py     parameterised transcript template + acquisition model (blur, noise, JPEG)
  tamper.py     T1 crude / T2 blended / T3 coherent / T4 complete forgeries
  ocr.py        page registration + CRNN/CTC slot reader + field assignment
  visual.py     ResNet-18 U-Net (RGB + ELA), ELA, full-page inference
  pipeline.py   constraint features, field prior, attribution/detection features, CVP system
scripts/
  train_ocr.py      train the CRNN on synthetic slot crops
  make_dataset.py   build data/{train,val,test} (+ jsonl ground truth)
  train_visual.py   train the visual branch on MCU-aligned patches (BCE + Dice)
  extract.py        run OCR + constraints + visual branch over a split / degradation
  evaluate.py       fit g_phi + heads, run all baselines, write results + LaTeX tables
  make_figures.py   report figures (PDF + PNG)
app.py          Gradio demo
report/         IEEE LaTeX paper (Overleaf-ready)
```

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/train_ocr.py --docs 4000 --steps 1200
.venv/bin/python scripts/make_dataset.py
.venv/bin/python scripts/train_visual.py --steps 1100 --batch 8
.venv/bin/python scripts/extract.py        # train/val/test + JPEG-60 and 0.6x-resize test copies
.venv/bin/python scripts/evaluate.py       # results/results.json, report/tables/*.tex, report/numbers.tex
.venv/bin/python scripts/make_figures.py
.venv/bin/python app.py        # http://127.0.0.1:7860
```

Everything runs on a laptop (Apple-silicon GPU via MPS, or CPU).

## Report (Overleaf)

`report/` is a complete IEEEtran project: `main.tex`, `refs.bib`, generated `numbers.tex`,
`results_text.tex`, `tables/` and `figures/`. Upload `cvp_report_overleaf.zip` via
Overleaf -> New Project -> Upload Project; it compiles with pdfLaTeX (default) or XeLaTeX.
