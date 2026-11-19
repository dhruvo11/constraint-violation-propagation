#!/bin/bash
# Rebuild every real-data result (Phases 2 and 3) from the downloaded datasets:
#   data/real/cord-v2   CORD v2 (Hugging Face naver-clova-ix/cord-v2, CC BY 4.0)
#   data/real/aiforge   AIForge-Doc v1 and v2, CORD part (Hugging Face Scam-AI, gated; last step)
# The synthetic models (scripts/reproduce.sh) must exist: the real visual branches are
# fine-tuned from models/visual_resunet.pt. Logs go to logs/.
#   ./scripts/reproduce_real.sh              all steps
#   ./scripts/reproduce_real.sh extract      from that step on
set -e
cd "$(dirname "$0")/.."
PY=".venv/bin/python -I"
mkdir -p logs
FROM=${1:-rules}
STEPS=${VISUAL_STEPS:-3000}
started=0

step() {
  local name=$1; shift
  [ "$name" = "$FROM" ] && started=1
  [ $started = 1 ] || return 0
  echo "[$(date +%H:%M)] $name ..."
  "$@" > "logs/real_$name.log" 2>&1 || { echo "step '$name' failed; see logs/real_$name.log"; exit 1; }
  grep -v -iE "warn|it/s" "logs/real_$name.log" | tail -n 3
}

each_split() {  # run a command once per CORD split, with {} replaced by the split
  for s in test validation train; do "${@//\{\}/$s}" || return 1; done
}

# Do real shops' receipts satisfy the rules, and how well does a free OCR read them?
step rules     $PY scripts/real_rule_validity.py
step ocr       each_split $PY scripts/real_ocr_cord.py --split {}
# Are the forgeries of real photos faithful as read?
step forgeries $PY scripts/real_forgery_check.py --split test
# Training data for the visual branch (CORD train/validation only), then two regimes:
# crude edits only (blended ones stay unseen) and every category.
step data      $PY scripts/make_real_dataset.py
step visual    bash -c "
  .venv/bin/python scripts/train_visual.py --doctypes cord --exclude T2,T4 --init models/visual_resunet.pt --lr 5e-4 --steps $STEPS --out models/visual_cord_crude.pt &&
  .venv/bin/python scripts/train_visual.py --doctypes cord --init models/visual_resunet.pt --lr 5e-4 --steps $STEPS --out models/visual_cord_all.pt"
# Both branches over the test receipts, per visual regime; then the evaluation.
step extract   bash -c "
  $PY scripts/real_extract_cord.py --split test &&
  $PY scripts/real_extract_cord.py --split test --visual models/visual_cord_crude.pt --tag _crude &&
  $PY scripts/real_extract_cord.py --split test --visual models/visual_cord_all.pt --tag _all"
step evaluate  bash -c "
  $PY scripts/real_evaluate_cord.py --split test &&
  $PY scripts/real_evaluate_cord.py --split test --tag _crude &&
  $PY scripts/real_evaluate_cord.py --split test --tag _all"
# Receipts edited by commercial AI editors (AIForge-Doc v1 and v2, CC BY-NC-SA 4.0). The datasets
# are gated on Hugging Face: accept their terms and log in (.venv/bin/hf auth login) first.
# Nothing is trained or tuned on them. The --jpeg runs repeat the CORD test receipts with both
# images saved once more as JPEG, the compression history of our own CORD forgeries.
step aiforge   bash -c "
  for v in v2 v1; do
    $PY scripts/real_aiforge.py --version \$v --download &&
    $PY scripts/real_aiforge.py --version \$v &&
    $PY scripts/real_aiforge.py --version \$v --evaluate &&
    $PY scripts/real_aiforge.py --version \$v --jpeg --split test &&
    $PY scripts/real_aiforge.py --version \$v --jpeg --evaluate || exit 1
  done &&
  $PY scripts/real_aiforge.py --table"
echo "[$(date +%H:%M)] done"
