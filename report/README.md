# The CVP paper (IEEE Computer Society journal format)

`main.tex` is the paper, in the IEEEtran `journal,compsoc` format (the format of IEEE
Computer Society journals such as TPAMI). `refs.bib` holds the references; the recent ones
were checked against their arXiv or publisher pages.

## Build it

- **Overleaf:** run `./build.sh` (or take `CVP_IEEE_Overleaf.zip` if it is already there),
  then on Overleaf choose *New Project > Upload Project* and pick the zip. It compiles with
  Overleaf's default settings (pdfLaTeX + BibTeX).
- **On this laptop:** `./build.sh` runs `tectonic main.tex` and writes `main.pdf` and the zip.

## Author details

The paper names RV University (School of Computer Science and Engineering) and both USNs.
To thank your guide by name, edit the Acknowledgments paragraph near the end of `main.tex`.

## Where the content comes from

Nothing in `tables/`, `numbers/` or `figures/` is written by hand:

| Folder | Written by |
|---|---|
| `tables/<type>/`, `numbers/` | `scripts/evaluate.py` |
| `tables/cross_type.tex` | `scripts/summarize.py` |
| `tables/head_selection.tex`, `tables/head_ablation.tex` | `scripts/select_head.py`, `scripts/ablate_head.py` |
| `tables/real/` | `scripts/real_rule_validity.py`, `scripts/real_evaluate_cord.py` |
| `figures/<type>/`, `figures/cross_type.pdf`, `figures/doctypes.pdf` | `scripts/make_figures.py` |
| `figures/cross_*.pdf` (flags, regimes, robustness), `figures/real/` | `scripts/make_paper_figures.py` |

The numbers quoted in the text of `main.tex` were taken from the same results
(`results/<type>/results.json`, `results/real/`); if an experiment is re-run, re-check them.
