#!/bin/sh
# Build the paper and a zip for Overleaf.
#
#   report/build.sh          main.pdf (tectonic) and CVP_IEEE_Overleaf.zip
#
# The zip holds main.tex, refs.bib, every generated table and number, and the figures the
# paper includes. On Overleaf: New Project > Upload Project > choose the zip; the default
# compiler (pdfLaTeX) and BibTeX build it as is.
set -e
cd "$(dirname "$0")"
TECTONIC=$(command -v tectonic || echo /opt/homebrew/bin/tectonic)
"$TECTONIC" main.tex
FIGS=$(grep -oE 'includegraphics(\[[^]]*\])?\{[^}]+\}' main.tex | sed -E 's/.*\{(.*)\}/\1/' | sort -u)
rm -f CVP_IEEE_Overleaf.zip
zip -q CVP_IEEE_Overleaf.zip main.tex refs.bib README.md numbers/*.tex $(find tables -name '*.tex') $FIGS
echo "built main.pdf and CVP_IEEE_Overleaf.zip ($(echo "$FIGS" | wc -l | tr -d ' ') figures)"
