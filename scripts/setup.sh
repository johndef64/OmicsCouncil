#!/usr/bin/env bash
# Create an isolated environment and install the (tiny) dependencies.
# Usage: bash scripts/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
echo ">> Creating virtual environment at .venv"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> Installing dependencies (numpy, scikit-learn, networkx, PyYAML)"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ">> Done. Activate with:  source .venv/bin/activate"
