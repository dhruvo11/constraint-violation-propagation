# Real-data results (Phases 2 and 3)

## Do real receipts satisfy the rules? (CORD v2, 1,000 genuine receipts, true values)

| Rule | Applies | Holds |
|---|---|---|
| count x unit price = price, for each item that prints a unit price | 727 | 99.7% |
| item prices (with add-ons, less item discounts) add up to the subtotal, or, when no subtotal is printed, with tax and service to the total | 968 | 97.4% |
| subtotal + tax + service - discount = total (or the prices already include the tax) | 642 | 97.4% |
| cash (+ card) - total = change | 530 | 96.6% |
| cash (+ card) covers the total | 616 | 98.4% |
| item counts add up to the printed number of items | 250 | 96.8% |

Genuine receipts that break some rule: **6.7%** (66/992).

With a free local OCR (docTR) reading each value at its layout label, genuine receipts breaking a rule rise to **10.9%** (from 7.2% on true values).

Forgeries of real photos read back exactly 98.6% of the time; as read, single-field forgeries break a rule 96.9% (T1) and 98.0% (T2), coherent ones keep every rule 88.9%.

## CVP on real receipt photos (CORD v2 test: 98 genuine, 98 T1, 98 T2, 90 T3, 90 T4)

| Visual branch trained on | Method | AUC | Attr. all | Attr. T2 | Attr. T4 | Ambiguity split | Flagged at validation threshold: genuine / T1 / T2 / T3 / T4 |
|---|---|---|---|---|---|---|---|
| synthetic documents only (zero-shot) | Visual-only | 0.546 | 60.0 | 11.0 | 56.7 | 40.0 |  |
|  | Constraint-only | 0.729 | 58.2 | 55.0 | 57.1 | 37.2 |  |
|  | CVP | 0.746 | 73.1 | 54.1 | 58.3 | 56.2 | 11.2 / 96.9 / 98.0 / 13.3 / 8.9 |
| real photos, crude edits only | Visual-only | 0.751 | 65.4 | 14.3 | 53.3 | 54.4 |  |
|  | Constraint-only | 0.729 | 58.2 | 55.0 | 57.1 | 37.2 |  |
|  | CVP | 0.838 | 75.6 | 56.5 | 55.0 | 64.9 | 14.3 / 100.0 / 98.0 / 98.9 / 13.3 |
| real photos, every category (has seen the blended renderer) | Visual-only | 0.892 | 92.3 | 75.0 | 97.8 | 85.4 |  |
|  | Constraint-only | 0.729 | 58.2 | 55.0 | 57.1 | 37.2 |  |
|  | CVP | 0.913 | 93.0 | 84.2 | 97.8 | 88.5 | 17.3 / 100.0 / 100.0 / 98.9 / 68.9 |

Chance on the ambiguity split: 38.1%. The last regime has seen the blended renderer that made the test forgeries, so it is an upper bound, not a test of unseen blending.

## Real human forgeries (Find it again!, 987 scans)

| Reading | Receipts | Checkable | Flagged by a rule |
|---|---|---|---|
| ocr (genuine) | 825 | 646 | 17.2% |
| ocr (forged) | 162 | 121 | 32.2% |
| transcription (genuine) | 825 | 551 | 29.0% |
| transcription (forged) | 162 | 107 | 33.6% |

Visual branch on the real forgeries (test scans): heat AUC, how often the hottest pixel is in an edited region (chance in brackets), and CVP's flags at a validation threshold.

| Visual branch | Heat AUC | Pointing | CVP flags genuine / forged |
|---|---|---|---|
| visual_resunet.pt | 0.527 | 8.6% (0.6%) | 12.6% / 17.1% |
| visual_cord_crude.pt | 0.508 | 2.9% (0.6%) | 12.6% / 17.1% |
| visual_cord_all.pt | 0.504 | 8.6% (0.6%) | 12.6% / 17.1% |

## Receipts edited by commercial AI editors (Phase 3: AIForge-Doc, CORD part)

AIForge-Doc rewrites one field with an AI image editor and pastes only that field's box back
(v1: Gemini 2.5 Flash Image 885, Ideogram v2 Edit 98; v2: GPT-Image-2 983, the same edits).
Each forgery is paired with its CORD photo, pixel-identical outside the edit. Nothing is
trained or tuned on AIForge-Doc (`scripts/real_aiforge.py`; 95% bootstrap intervals over
receipts in `results/real/aiforge_<version>.json`).

- 84.6% of the edits change a value some rule reads; with true values 95.4% of those break a rule.
- docTR reads the requested value back: GPT-Image-2 91.0%, Gemini 74.5%, Ideogram 50.0%.
- As read, a rule breaks on 82.9% (v2) / 82.2% (v1) of the AI edits and 12.2% of authentic receipts.

| Visual branch | Method | v2 AUC | v2 attr. | v2 ambiguity | v1 AUC | v1 attr. | v1 ambiguity |
|---|---|---|---|---|---|---|---|
| synthetic only (983 receipts) | Visual-only | 0.499 | 13.9 | 16.2 | 0.523 | 35.5 | 31.9 |
|  | Constraint-only | 0.846 | 64.1 | 42.2 | 0.843 | 64.5 | 43.9 |
|  | CVP | 0.845 | 66.8 | 46.9 | 0.848 | 74.1 | 60.9 |
| CORD, crude edits (99 test receipts) | Visual-only | 0.502 | 23.8 | 24.0 | 0.548 | 41.1 | 35.0 |
|  | Constraint-only | 0.843 | 63.1 | 42.0 | 0.828 | 61.7 | 41.4 |
|  | CVP | 0.843 | 72.0 | 57.0 | 0.845 | 73.8 | 60.0 |
| CORD, every category (99 test receipts) | Visual-only | 0.518 | 22.6 | 26.0 | 0.560 | 48.8 | 49.0 |
|  | Constraint-only | 0.843 | 63.1 | 42.0 | 0.828 | 61.7 | 41.4 |
|  | CVP | 0.834 | 71.4 | 56.0 | 0.844 | 76.8 | 67.0 |

Chance on the ambiguity split: 41.9% (983 receipts), 40.8% (99). Gain of CVP over the rules
in attribution, 95% interval: v2 0.4 to 4.9 points, v1 7.1 to 12.0.

The fine-tuned branches were built on photos saved once more as JPEG (quality 90), and
AIForge-Doc's PNGs were not: at the validation thresholds CVP then raises alarms on 32.3% /
20.2% of authentic receipts (14.3% / 17.3% on our own test set). With the same extra save of
both images (`--jpeg`, test receipts), the alarms are 14.1% / 17.2% of authentic receipts and
79.8-81.8% (v2) / 76.8-78.8% (v1) of AI edits, and the visual branches stay at chance
(AUC 0.497-0.579).
