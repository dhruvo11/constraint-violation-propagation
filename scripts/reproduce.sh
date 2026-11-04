#!/bin/bash
# Rebuild everything from scratch: OCR, corpus, the three visual-branch regimes, features,
# evaluation and figures. Each heavy step runs alone (the target laptop has 8 GB of RAM);
# logs go to logs/. Expect a few hours in total on an Apple-silicon laptop.
#   ./scripts/reproduce.sh            all steps
#   ./scripts/reproduce.sh evaluate   only the steps from evaluation on
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs
FROM=${1:-ocr}
STEPS=(ocr data visual extract evaluate figures)
started=0

step() {
  local name=$1; shift
  [ "$name" = "$FROM" ] && started=1
  [ $started = 1 ] || return 0
  echo "[$(date +%H:%M)] $name ..."
  "$@" > "logs/$name.log" 2>&1 || { echo "step '$name' failed; see logs/$name.log"; exit 1; }
  tail -n 2 "logs/$name.log"
}

step ocr      $PY scripts/train_ocr.py --docs 4000 --steps 1200
step data     $PY scripts/make_dataset.py
step visual   bash -c "
  $PY scripts/train_visual.py --steps 1100 --exclude T2,T4 --extra train_g1 &&
  $PY scripts/train_visual.py --steps 1100 --exclude T2,T4 --out models/visual_resunet_blind.pt &&
  $PY scripts/train_visual.py --steps 1100 --out models/visual_resunet_oracle.pt"
step extract  bash -c "
  $PY scripts/extract.py &&
  $PY scripts/extract.py --visual models/visual_resunet_blind.pt --tag _blind --jobs val:clean,test:clean &&
  $PY scripts/extract.py --visual models/visual_resunet_oracle.pt --tag _oracle --jobs val:clean,test:clean"
step evaluate $PY scripts/evaluate.py
step figures  $PY scripts/make_figures.py
echo "[$(date +%H:%M)] done"
