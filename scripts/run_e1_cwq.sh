#!/bin/bash
# run_e1_cwq.sh — E1: stage-wise attack CWQ, GNN-RAG model
# Usage: bash run_e1_cwq.sh [variant]   (default: greedy)
source "$(dirname "$0")/../scripts/env.sh"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${1:-greedy}"

echo "E1 CWQ GNN-RAG | variant=$VARIANT"

CUDA_VISIBLE_DEVICES=0,1,2,3 python "$SCRIPT_DIR/e1_cwq_gnn_rag.py" \
  --variant "$VARIANT" \
  --out-dir ${RESULTS_DIR}/e1_cwq_gnn_rag \
  --n-boot  1000
