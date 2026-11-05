| Type | Fields / rules | rho | OCR | Ambig. edits (chance) | Visual-only AUC / attr. / ambig. | Constraint-only AUC / attr. / ambig. | Concat fusion AUC / attr. / ambig. | CVP AUC / attr. / ambig. |
|---|---|---|---|---|---|---|---|---|
| Academic transcript | 10 / 10 | 0.70 | 99.9 | 244 (20.0) | 0.941 / 87.3 / 84.4 | 0.787 / 48.7 / 20.0 | 0.932 / 55.5 / 23.8 | 0.953 / 89.1 / 87.7 |
| GST tax invoice | 24 / 19 | 0.94 | 99.7 | 264 (50.0) | 0.957 / 88.0 / 87.9 | 0.810 / 57.0 / 50.6 | 0.950 / 62.4 / 70.5 | 0.959 / 90.6 / 91.3 |
| Shop receipt | 27 / 18 | 0.96 | 99.5 | 77 (50.0) | 0.965 / 84.4 / 86.4 | 0.791 / 62.8 / 43.5 | 0.953 / 64.4 / 72.7 | 0.966 / 91.9 / 76.6 |
| Certificate of completion | 10 / 10 | 0.90 | 99.1 | 359 (44.6) | 0.961 / 89.7 / 89.4 | 0.736 / 43.6 / 41.3 | 0.944 / 79.3 / 73.0 | 0.956 / 92.4 / 91.9 |
| Passport (Utopia specimen) | 10 / 18 | 0.90 | 99.8 | 396 (50.0) | 0.890 / 83.7 / 83.3 | 0.782 / 48.1 / 50.5 | 0.933 / 70.9 / 90.4 | 0.949 / 88.0 / 87.6 |
| Student ID card | 7 / 8 | 0.86 | 99.9 | 456 (50.0) | 0.923 / 81.0 / 81.8 | 0.780 / 46.4 / 50.1 | 0.937 / 80.7 / 91.9 | 0.942 / 87.6 / 93.2 |

CVP attribution head: class prior, prob heat (chosen on g1 validation data, results/head_selection.json).
