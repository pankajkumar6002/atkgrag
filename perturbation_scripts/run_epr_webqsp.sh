#!/bin/bash
# Run EPR-KGQA pipeline for WebQSP: clean baseline + p4 + p7 perturbations
# Usage: bash run_epr_webqsp.sh
# Run in tmux: tmux new -s epr_webqsp "bash run_epr_webqsp.sh"
source "$(dirname "$0")/../scripts/env.sh"

set -e
cd ${EPR_KGQA_ROOT}

export CUDA_VISIBLE_DEVICES=0  # GPU 0 (~66GB free)
export ODBC_DSN=freebase
export SPARQL_ENDPOINT=http://127.0.0.1:3001/sparql

EPR_PY="${EPR_PYTHON}"
SCRIPT=${REPO_ROOT}/perturbation_scripts/epr_pert_experiment.py
LOG_DIR=/tmp/epr_logs
mkdir -p $LOG_DIR

echo "========================================"
echo "EPR-KGQA WebQSP Pipeline"
echo "Started: $(date)"
echo "========================================"

# Stage 1: Clean baseline (ep + subg + nsm) — ap already done by biencoder
echo ""
echo "--- STEP 1: WebQSP Clean Baseline (ep + subg + nsm) ---"
$EPR_PY $SCRIPT --dataset webqsp --ptype clean --stage ep subg nsm --gpu 0 \
    2>&1 | tee $LOG_DIR/webqsp_clean.log

echo ""
echo "--- STEP 2: WebQSP p4 Perturbation (all stages) ---"
$EPR_PY $SCRIPT --dataset webqsp --ptype p4 --stage all --gpu 0 \
    2>&1 | tee -a $LOG_DIR/webqsp_p4.log

echo ""
echo "--- STEP 3: WebQSP p7 Perturbation (all stages) ---"
$EPR_PY $SCRIPT --dataset webqsp --ptype p7 --stage all --gpu 0 \
    2>&1 | tee -a $LOG_DIR/webqsp_p7.log

echo ""
echo "========================================"
echo "All WebQSP runs complete: $(date)"
echo "========================================"
