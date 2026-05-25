#!/bin/bash
# run_all_variants.sh — Run E1/E2 across all 6 perturbation variants
# Usage: bash run_all_variants.sh [gnn_rag|gmt_kbqa|both] [CWQ|WebQSP|both]
source "$(dirname "$0")/../scripts/env.sh"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-gnn_rag}"    # gnn_rag | gmt_kbqa | both
DATASET="${2:-both}"     # CWQ | WebQSP | both

VARIANTS=(greedy temp_0.10 temp_0.25 temp_0.50 temp_0.75 temp_0.90)

run_gnn_rag_cwq() {
  for v in "${VARIANTS[@]}"; do
    echo "=== GNN-RAG CWQ variant=$v ==="
    bash "$SCRIPT_DIR/run_e1_cwq.sh" "$v"
  done
}

run_gnn_rag_webqsp() {
  for v in "${VARIANTS[@]}"; do
    echo "=== GNN-RAG WebQSP variant=$v ==="
    bash "$SCRIPT_DIR/run_e2_webqsp.sh" "$v"
  done
}

run_gmt_cwq() {
  for v in "${VARIANTS[@]}"; do
    echo "=== GMT-KBQA CWQ variant=$v ==="
    bash "$SCRIPT_DIR/run_e1_e2_gmt_kbqa.sh" CWQ "$v"
  done
}

run_gmt_webqsp() {
  for v in "${VARIANTS[@]}"; do
    echo "=== GMT-KBQA WebQSP variant=$v ==="
    bash "$SCRIPT_DIR/run_e1_e2_gmt_kbqa.sh" WebQSP "$v"
  done
}

case "$MODEL-$DATASET" in
  gnn_rag-CWQ)     run_gnn_rag_cwq ;;
  gnn_rag-WebQSP)  run_gnn_rag_webqsp ;;
  gnn_rag-both)    run_gnn_rag_cwq; run_gnn_rag_webqsp ;;
  gmt_kbqa-CWQ)   run_gmt_cwq ;;
  gmt_kbqa-WebQSP) run_gmt_webqsp ;;
  gmt_kbqa-both)   run_gmt_cwq; run_gmt_webqsp ;;
  both-CWQ)        run_gnn_rag_cwq; run_gmt_cwq ;;
  both-WebQSP)     run_gnn_rag_webqsp; run_gmt_webqsp ;;
  both-both)       run_gnn_rag_cwq; run_gnn_rag_webqsp; run_gmt_cwq; run_gmt_webqsp ;;
  *)               echo "Usage: $0 [gnn_rag|gmt_kbqa|both] [CWQ|WebQSP|both]"; exit 1 ;;
esac

echo "All runs complete."
