#!/bin/bash
# run_e5.sh — E5: Attack validity (PPL + BERTScore) for all 5 perturbation types
# Uses perturbation data at ${PERT_RAW_DIR}/
source "$(dirname "$0")/../scripts/env.sh"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${1:-greedy}"

echo "Running E5 validity check for variant: $VARIANT"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/e5_validity.py" \
  --pert-root  ${PERT_RAW_DIR} \
  --gpt2-path  ${MODELS_DIR}/gpt2-large \
  --out        ${RESULTS_DIR}/e5_validity_${VARIANT}.json \
  --variant    "$VARIANT" \
  --device     cuda \
  --gpu        0
