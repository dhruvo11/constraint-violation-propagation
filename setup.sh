#!/bin/bash
# One-command setup: creates .venv with every dependency and runs the test suite.
#   ./setup.sh
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "This project uses uv to manage Python. Install it first:"
  echo "  brew install uv        (or: curl -LsSf https://astral.sh/uv/install.sh | sh)"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating the Python 3.12 environment in .venv ..."
  uv venv --python 3.12 .venv
fi
echo "Installing dependencies (PyTorch is large; the first install can take a while) ..."
uv pip install --python .venv/bin/python -r requirements.txt -c requirements-lock.txt  # exact tested versions
echo "Installing what reads real documents (docTR OCR and dataset readers; see requirements-real.txt) ..."
uv pip install --python .venv/bin/python -c requirements-lock.txt -r requirements-real.txt
uv pip install --python .venv/bin/python --no-deps "python-doctr==1.1.0"

echo "Getting the trained models ..."
./scripts/get_models.sh || echo "Could not download the trained models (no internet?); the app needs them."

echo "Running the tests ..."
.venv/bin/python -m pytest

if [ ! -f models/visual_resunet.pt ] || [ ! -f models/ocr_crnn.pt ]; then
  echo
  echo "Setup done, but the trained models are missing. Download them with ./scripts/get_models.sh,"
  echo "or train everything yourself with ./scripts/reproduce.sh (a few hours)."
else
  echo
  echo "Setup done. Start the app by double-clicking 'Start CVP.command' or running:"
  echo "  .venv/bin/python -m webapp"
fi
