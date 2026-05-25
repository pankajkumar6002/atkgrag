#!/bin/bash
# run_e1_e2_gmt_kbqa.sh — E1/E2: stage-wise attack using GMT-KBQA (T5-base)
# Must be run from ${GMT_KBQA_ROOT}/ (repo root)
# Usage: bash run_e1_e2_gmt_kbqa.sh [CWQ|WebQSP] [variant]
source "$(dirname "$0")/../scripts/env.sh"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-CWQ}"
VARIANT="${2:-greedy}"

echo "E1/E2 GMT-KBQA | dataset=$DATASET | variant=$VARIANT"
echo "Changing to GMT-KBQA directory..."
cd ${GMT_KBQA_ROOT}

if [ "$DATASET" = "CWQ" ]; then
  OUT_DIR="${RESULTS_DIR}/e1_cwq_gmt_kbqa"
else
  OUT_DIR="${RESULTS_DIR}/e2_webqsp_gmt_kbqa"
fi

CUDA_VISIBLE_DEVICES=0,1 python "$SCRIPT_DIR/e1_e2_gmt_kbqa.py" \
  --dataset   "$DATASET" \
  --variant   "$VARIANT" \
  --out-dir   "$OUT_DIR" \
  --gmt-root  ${GMT_KBQA_ROOT} \
  --beam-size 10 \
  --batch-size 8 \
  --n-boot    1000 \
  --device    cuda \
  --gpu       0,1
