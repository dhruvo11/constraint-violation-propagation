# Real-data results (Phase 2)

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
