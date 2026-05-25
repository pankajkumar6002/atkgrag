#!/bin/bash
# Run EPR-KGQA pipeline for CWQ: clean baseline + p4 + p7 perturbations
# Usage: bash run_epr_cwq.sh
# Run in tmux: tmux new -s epr_cwq "bash run_epr_cwq.sh"
source "$(dirname "$0")/../scripts/env.sh"

set -e
cd ${EPR_KGQA_ROOT}

export CUDA_VISIBLE_DEVICES=2  # GPU 2 (free)
export ODBC_DSN=freebase
export SPARQL_ENDPOINT=http://127.0.0.1:3001/sparql

EPR_PY="${EPR_PYTHON}"
SCRIPT=${REPO_ROOT}/perturbation_scripts/epr_pert_experiment.py
LOG_DIR=/tmp/epr_logs
mkdir -p $LOG_DIR

echo "========================================"
echo "EPR-KGQA CWQ Pipeline"
echo "Started: $(date)"
echo "========================================"

# Stage 1: Clean baseline (ep + subg + nsm) — ap already done by biencoder
echo ""
echo "--- STEP 1: CWQ Clean Baseline (ep + subg + nsm) ---"
$EPR_PY $SCRIPT --dataset cwq --ptype clean --stage ep subg nsm --gpu 2 \
    2>&1 | tee $LOG_DIR/cwq_clean.log

echo ""
echo "--- STEP 2: CWQ p4 Perturbation (all stages) ---"
$EPR_PY $SCRIPT --dataset cwq --ptype p4 --stage all --gpu 2 \
    2>&1 | tee -a $LOG_DIR/cwq_p4.log

echo ""
echo "--- STEP 3: CWQ p7 Perturbation (all stages) ---"
$EPR_PY $SCRIPT --dataset cwq --ptype p7 --stage all --gpu 2 \
    2>&1 | tee -a $LOG_DIR/cwq_p7.log

echo ""
echo "========================================"
echo "All CWQ runs complete: $(date)"
echo "========================================"
