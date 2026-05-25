#!/bin/bash
# run_e35.sh — E35: GNN-RAG weight norms (runs immediately, no perturbation needed)
source "$(dirname "$0")/../scripts/env.sh"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/e35_gnn_norms.py" \
  --ckpt      ${GNN_RAG_DATA}/pretrain/ReaRev_CWQ.ckpt \
  --data      ${GNN_RAG_DATA}/CWQ/test.json \
  --out       ${RESULTS_DIR}/e35_gnn_norms.json \
  --max-graphs 200
