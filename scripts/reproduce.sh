#!/bin/bash
# Rebuild everything from scratch: the shared OCR, a corpus per document type, the three
# visual-branch regimes (trained on every type at once), features, evaluation and figures.
# Each heavy step runs alone (the target laptop has 8 GB of RAM); logs go to logs/.
# Expect several hours on an Apple-silicon laptop.
#   ./scripts/reproduce.sh              all steps
#   ./scripts/reproduce.sh evaluate     only the steps from evaluation on
#   TYPES="invoice receipt" ./scripts/reproduce.sh data     some types only
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs
FROM=${1:-ocr}
ALL=$($PY -c "from cvp import doctypes; print(' '.join(doctypes.names()))")
TYPES=${TYPES:-$ALL}
COMMA=$(echo $TYPES | tr ' ' ',')
STEPS=${VISUAL_STEPS:-3000}
started=0

step() {
  local name=$1; shift
  [ "$name" = "$FROM" ] && started=1
  [ $started = 1 ] || return 0
  echo "[$(date +%H:%M)] $name ..."
  "$@" > "logs/$name.log" 2>&1 || { echo "step '$name' failed; see logs/$name.log"; exit 1; }
  tail -n 2 "logs/$name.log"
}

per_type() {  # run a command once per document type, with {} replaced by the type
  for t in $TYPES; do "${@//\{\}/$t}" || return 1; done
}

step ocr       $PY scripts/train_ocr.py
step data      $PY scripts/make_dataset.py --doctypes $COMMA
step visual    $PY scripts/train_visual.py --steps $STEPS --exclude T2,T4 --extra train_g1
step extract   per_type $PY scripts/extract.py --doctype {}
step select    $PY scripts/select_head.py   # attribution-head design, chosen on val_g1 (never on test)
step evaluate  per_type $PY scripts/evaluate.py --doctype {}
step visual_regimes bash -c "
  $PY scripts/train_visual.py --steps $STEPS --exclude T2,T4 --out models/visual_resunet_blind.pt &&
  $PY scripts/train_visual.py --steps $STEPS --out models/visual_resunet_oracle.pt"
step extract_regimes per_type bash -c "
  $PY scripts/extract.py --doctype {} --visual models/visual_resunet_blind.pt --tag _blind --jobs val:clean,test:clean &&
  $PY scripts/extract.py --doctype {} --visual models/visual_resunet_oracle.pt --tag _oracle --jobs val:clean,test:clean"
step evaluate_regimes per_type $PY scripts/evaluate.py --doctype {}
step ablation  $PY scripts/ablate_head.py   # every head design on the test set, after the choice
step figures   bash -c "$PY scripts/make_figures.py && $PY scripts/summarize.py"
echo "[$(date +%H:%M)] done"
