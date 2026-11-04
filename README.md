# CVP — Constraint Violation Propagation

Attributing document forgery when visual evidence fails. Generative inpainting removes the
pixel traces that forensic detectors look for, but it does not know a document's
arithmetic and logic. CVP reads the document, checks its constraints, uses the pattern of
violated constraints to narrow down which field could have been edited, and uses a visual
tamper heatmap to pick the edited field within that set.

Research project by Harshith Rudresh (1RUA24CSE0157) and Dhruv S P (1RUA24CSE0133).

## Quick start

```bash
./setup.sh                # one-time: Python environment, dependencies, tests
```

Then double-click **Start CVP.command** (or run `.venv/bin/python app.py`). The app opens
in your browser once it is ready; keep the Terminal window open while you use it.

## Layout

```
cvp/                      the library
  core/schema.py          fields, constraints, equivalence classes (Prop. 1), signatures, rho
  core/render.py          fonts, acquisition model (blur, noise, JPEG), Document record
  core/tamper.py          forgery taxonomy T1-T4 and the crude / blended renderers
  doctypes/               one module per document type (fields, rules, template, forger)
    transcript.py         academic transcript: 10 fields, 10 constraints
  ocr.py                  page registration + CRNN/CTC slot reader
  visual.py               ResNet-18 U-Net over RGB + ELA
  pipeline.py             constraint features, field prior, heads, the CVP system
  paths.py                data/, features/, results/ per document type; models/
scripts/
  train_ocr.py            train the OCR on synthetic slot crops
  make_dataset.py         build data/<type>/{train,val,test,train_g1}
  train_visual.py         train the visual branch on MCU-aligned patches (BCE + Dice)
  extract.py              OCR + constraints + visual branch over a split / degradation
  evaluate.py             fit g_phi and heads, run all baselines, write results and tables
  fit_heads.py            fit only the heads the app needs
  make_figures.py         report figures (PDF + PNG)
  reproduce.sh            rebuild everything from scratch
tests/                    pytest suite (theory, rendering, forgeries, models)
app.py                    the app
report/                   IEEE LaTeX paper (draft; on hold)
```

Generated artefacts are not under version control and are rebuilt by
`./scripts/reproduce.sh`: `data/` (synthetic corpus), `features/` (cached model outputs),
`models/` (trained weights), `results/` (metrics) and `logs/`.

## Adding a document type

Create a module in `cvp/doctypes/` that subclasses `DocType`. Declare its fields and
constraints (`Schema`), a template (`sample_spec`, `render_clean`, `anchor`, `slots`) and a
forger (`violating_value`, `coherent_edit`, `random_values`), then register it in
`cvp/doctypes/__init__.py`. Equivalence classes, signatures, candidate sets and the
reduction factor are derived from the declared constraints automatically.

## Tests

```bash
.venv/bin/python -m pytest
```

## Reproducing the results

```bash
./scripts/reproduce.sh            # all steps, a few hours on an 8 GB Apple-silicon laptop
./scripts/reproduce.sh evaluate   # only evaluation and figures, from cached features
```

Heavy steps run one at a time; closing memory-hungry apps (browsers, music players)
during training speeds them up considerably.
