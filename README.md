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
  core/rules.py           check digits (ICAO 9303, GSTIN), money in paise, amounts in words, dates
  doctypes/               one module per document type (fields, rules, template, forger)
    transcript.py         academic transcript: 10 fields, 10 constraints
    invoice.py            GST tax invoice: 24 fields, 19 constraints
    receipt.py            shop receipt: 27 fields, 18 constraints
    certificate.py        certificate of completion: 10 fields, 10 constraints
    passport.py           Utopia specimen passport (ICAO 9303 MRZ): 10 fields, 18 constraints
    idcard.py             student ID card: 7 fields, 8 constraints
    _draw.py              drawing helpers shared by the templates
  ocr.py                  page registration + CRNN/CTC slot reader
  visual.py               ResNet-18 U-Net over RGB + ELA
  pipeline.py             constraint features, field prior, heads, the CVP system
  paths.py                data/, features/, results/ per document type; models/
  real/                   real documents (Phase 2): public datasets and the rules they satisfy
    money.py              amounts and counts as printed on real receipts
    cord.py               CORD v2 receipts: loader, rebuilding a receipt from its labelled words
    receipt_rules.py      six receipt rules, robust to real shops' conventions
    receipt_schema.py     each real receipt's own constraint graph (the core theory applies unchanged)
    read.py               how CVP reads a real receipt (docTR at the layout labels)
    forge.py              the T1-T4 forgeries on real photos, with ground truth
    view.py               what the visual branch sees of a real photo (same in training and testing)
    system.py, demo.py    CVP end to end on one real photo; the app's "Real receipt" tab
scripts/
  train_ocr.py            train the OCR on synthetic slot crops
  make_dataset.py         build data/<type>/{train,val,test,train_g1}
  train_visual.py         train the visual branch on MCU-aligned patches (BCE + Dice)
  extract.py              OCR + constraints + visual branch over a split / degradation
  evaluate.py             fit g_phi and heads, run all baselines, write results and tables
  fit_heads.py            fit only the heads the app needs
  make_figures.py         figures per type and across types (PDF for the report, PNG for the app)
  summarize.py            one table across every document type (results/summary.md)
  select_head.py          choose the attribution head's design on g1 validation data
  ablate_head.py          every head design on the test set, after the choice
  reproduce.sh            rebuild every synthetic-data result from scratch
  real_*.py               real data: rule validity, OCR, forgery check, extraction, evaluation,
                          threshold; make_real_dataset.py builds real-photo training data
  reproduce_real.sh       rebuild every real-data result from the downloaded datasets
tests/                    pytest suite (theory, rules, rendering, forgeries, scoring, models)
app.py                    the app
report/                   IEEE LaTeX paper (draft; on hold)
```

The bulky generated artefacts are not under version control and are rebuilt by
`./scripts/reproduce.sh`: `data/` (synthetic corpus), `features/` (cached model outputs),
`models/` (trained weights) and `logs/`. The small outputs are kept so the numbers can be
read without re-running anything: `results/` (metrics as JSON) and the report's tables,
numbers and figures under `report/`.

## Document types

| Type | What the rules check | Fields / rules | rho |
|---|---|---|---|
| Transcript | marks sum to the total, percentage, grade band, mark ranges, issue date, roll-number format | 10 / 10 | 0.70 |
| GST invoice | qty x rate per line, taxable value, CGST+SGST or IGST by state, round-off, total, amount in words, GSTIN check characters, financial year in the invoice number | 24 / 19 | 0.94 |
| Shop receipt | qty x rate, item count, total quantity, subtotal, 2.5% CGST/SGST, total, cash - total = change, GSTIN, bill number month | 27 / 18 | 0.96 |
| Certificate | duration matches the dates, issue date window, grade matches score, certificate-number check letter, verification code, USN | 10 / 10 | 0.90 |
| Passport | every printed field equals its MRZ copy, ICAO check digits, expiry follows from issue date and age | 10 / 18 | 0.90 |
| Student ID card | USN encodes branch and admission year, four-year batch, validity, age at admission, issue window | 7 / 8 | 0.86 |

All templates are fictional (an Institute of Advanced Technology, Skillnest Academy, the
ICAO's own specimen state Utopia); no real organisation's document is imitated. rho is
the search-space reduction factor of Eq. 1, computed from the rules.

## Results

1,500 synthetic test documents per type, main setting: the visual branch was trained on
crude edits plus blended edits from an older generator (g1); the test's blended edits come
from a generator (g2) that no model saw. Percentages are top-1 field attribution over all
tampered documents; the ambiguity split is the single-field edits whose broken rules fit
more than one field, where constraint reasoning alone is at chance.

| Type | OCR fields read correctly | Detection AUC (CVP / visual) | Attribution (CVP / visual / rules only) | Ambiguity split (CVP / visual / chance) | CVP flags: genuine / blended edit / complete forgery |
|---|---|---|---|---|---|
| Transcript | 99.9% | 0.953 / 0.941 | **89.1** / 87.3 / 48.7 | 87.7 / 84.4 / 20.0 | 2.9% / 97.9% / 45.1% |
| GST invoice | 99.7% | 0.959 / 0.957 | **90.6** / 88.0 / 57.0 | 91.3 / 87.9 / 50.0 | 2.5% / 98.3% / 43.7% |
| Shop receipt | 99.5% | 0.966 / 0.965 | **91.9** / 84.4 / 62.8 | 76.6 / 86.4 / 50.0 | 1.2% / 99.7% / 65.0% |
| Certificate | 99.1% | 0.956 / 0.961 | **92.4** / 89.7 / 43.6 | 91.9 / 89.4 / 44.6 | 2.9% / 97.6% / 65.5% |
| Passport | 99.8% | 0.949 / 0.890 | **88.0** / 83.7 / 48.1 | 87.6 / 83.3 / 50.0 | 5.8% / 97.8% / 48.7% |
| Student ID card | 99.9% | 0.942 / 0.923 | **87.6** / 81.0 / 46.4 | 93.2 / 81.8 / 50.0 | 3.3% / 100.0% / 38.2% |

What this shows, and where it falls short:

- CVP attributes better than the visual branch, the rules alone, ELA and feature
  concatenation on every type. Algorithm 1 as stated (the hard rule) does better still on the
  transcript (92.0) and the ID card (88.9) but worse on the other four, where the learned head
  is best. Blended edits that the visual branch alone flags only 13-67% of the time are caught
  97.6-100% of the time, because they still break the rules.
- On the ambiguity split CVP beats the visual branch on five types. The exception is the
  receipt, where editing the cash or the change breaks the same single rule: the learned
  class posterior leans to one of the two and can overrule the heatmap.
- Complete forgeries (T4: every rule kept, blended rendering) are caught only when the visual
  branch sees them, 38-66% of the time. Unconstrained fields such as names are protected by
  the visual branch alone.
- Genuine documents are flagged 1.2-5.8% of the time, mostly through OCR misreads that break a
  rule (the passport's check digits make a single misread digit visible to several rules).

**How the attribution head was chosen.** Six designs (field prior: per class, or shared by
fields with the same signature; heatmap evidence: as probabilities, in log space, or log
max only) and Algorithm 1 as stated were compared on a separate validation split of g1
blended edits (`scripts/select_head.py`), never on the test set, with the criterion fixed
beforehand: mean attribution across types. The per-class prior with probability-space
heat won (97.0% against 91.1% for Algorithm 1) and is the CVP above.
`scripts/ablate_head.py` then reports every design on the test set: the alternatives trade
blended single-field edits against complete forgeries and none is better overall
(`report/tables/head_ablation.tex`). Scoring counts only exact ties as ties, so a method
that ranks fields by tiny heatmap differences gets credit for its arg-max like the app does.

Per-type tables are in `report/tables/<type>/`, figures in `report/figures/<type>/` (also in
the app's Results tab), and the cross-type summary in `results/summary.md`.

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

The real-data results need the public datasets in `data/real/` (not under version
control) and the synthetic models above:

```bash
.venv/bin/hf download naver-clova-ix/cord-v2 --repo-type dataset --local-dir data/real/cord-v2
./scripts/reproduce_real.sh       # rule validity, OCR, forgeries, fine-tuning, evaluation
```
