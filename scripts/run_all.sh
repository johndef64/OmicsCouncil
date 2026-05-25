#!/usr/bin/env bash
# Run the full experimental pipeline: classification, ablation, robustness.
# The whole suite finishes in well under a minute on a CPU.
# Usage: bash scripts/run_all.sh [path/to/config.yaml]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/breast_cancer.yaml}"
PY="${PYTHON:-python3}"
[ -d .venv ] && source .venv/bin/activate || true

echo "############################################################"
echo "# OmicsCouncil — full pipeline on: $CONFIG"
echo "############################################################"

echo; echo ">>> [1/3] Classification + trustworthy metrics"
"$PY" experiments/run_classification.py --config "$CONFIG"

echo; echo ">>> [2/3] Ablation study"
"$PY" experiments/run_ablation.py --config "$CONFIG"

echo; echo ">>> [3/3] Missing-modality robustness"
"$PY" experiments/run_robustness.py --config "$CONFIG"

echo; echo ">>> All results written to results/"
