#!/bin/sh
# Bundle report/ into an Overleaf-uploadable zip (Overleaf: New Project -> Upload Project).
cd "$(dirname "$0")/../report" || exit 1
rm -f ../cvp_report_overleaf.zip
zip -qr ../cvp_report_overleaf.zip main.tex refs.bib numbers.tex results_text.tex tables figures/*.pdf figures/transcript.png -x '.*'
echo "wrote $(cd .. && pwd)/cvp_report_overleaf.zip"
