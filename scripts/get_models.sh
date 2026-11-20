#!/bin/bash
# Download the trained models the app needs from the project's GitHub release (models-v1), check
# them against their SHA-256 and unpack them into models/. setup.sh runs this; it does nothing when
# the models are already here. To train them yourself instead: ./scripts/reproduce.sh (a few hours).
#   ./scripts/get_models.sh
set -e
cd "$(dirname "$0")/.."
URL="https://github.com/dhruvo11/constraint-violation-propagation/releases/download/models-v1/cvp-models-v1.zip"
SHA256="06eedd03d44d157dd094d9258705782fa401882abcf82514131e6ddd0f046b43"
ZIP="models/.cvp-models-v1.zip"
NEED="models/ocr_crnn.pt models/visual_resunet.pt models/visual_cord_crude.pt
      models/heads_transcript.pkl models/heads_invoice.pkl models/heads_receipt.pkl
      models/heads_certificate.pkl models/heads_passport.pkl models/heads_idcard.pkl"

missing=0
for f in $NEED; do [ -f "$f" ] || missing=1; done
if [ $missing = 0 ]; then
  echo "The trained models are already in models/."
  exit 0
fi

mkdir -p models
echo "Downloading the trained models (110 MB) from the project's GitHub release ..."
curl -L --fail --retry 3 --retry-delay 2 -C - -o "$ZIP" "$URL"
echo "Checking the download ..."
if ! echo "$SHA256  $ZIP" | shasum -a 256 -c - >/dev/null; then
  rm -f "$ZIP"
  echo "The download is damaged (wrong SHA-256). Run ./scripts/get_models.sh again."
  exit 1
fi
unzip -o -q "$ZIP"
rm -f "$ZIP"
echo "Done: the trained models are in models/."
