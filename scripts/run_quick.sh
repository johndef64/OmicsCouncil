#!/usr/bin/env bash
# Smoke test: a single classification run to confirm the install works.
# Usage: bash scripts/run_quick.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
[ -d .venv ] && source .venv/bin/activate || true

"$PY" experiments/run_classification.py --config configs/breast_cancer.yaml --out smoke.json
echo ">> Smoke test OK."
