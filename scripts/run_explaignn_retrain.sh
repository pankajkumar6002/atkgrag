#!/bin/bash
# run_explaignn_retrain.sh
# Full EXPLAIGNN retrain on ConvMix with local CLOCQ server.
# Steps: silver annotation → QU (BART SR) train → ERS inference → GNN train → eval
# Usage: bash run_explaignn_retrain.sh
source "$(dirname "$0")/../scripts/env.sh"

set -euo pipefail

EXPLAIGNN_ROOT="${EXPLAIGNN_ROOT}"
PYTHON=""${PYTHON_EXP}""
CONFIG_PATH="$EXPLAIGNN_ROOT/config/convmix/explaignn_retrain.yml"
CONFIG_REL="config/convmix/explaignn_retrain.yml"
SOURCES="kb_text_table_info"
LOG="${RESULTS_DIR}/retrain/retrain_full.log"
SILVER_LOG="${RESULTS_DIR}/retrain/silver_annotation.log"
ANNOTATED_TRAIN="$EXPLAIGNN_ROOT/_intermediate_representations/convmix_retrain/annotated_train.json"
ANNOTATED_DEV="$EXPLAIGNN_ROOT/_intermediate_representations/convmix_retrain/annotated_dev.json"

mkdir -p ${RESULTS_DIR}/retrain
mkdir -p "$EXPLAIGNN_ROOT/_intermediate_representations/convmix_retrain"
cd "$EXPLAIGNN_ROOT"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "[RETRAIN] Starting at $(date)" | tee "$LOG"

# Step 1: Verify local CLOCQ is up
echo "[RETRAIN] Checking local CLOCQ server..." | tee -a "$LOG"
response=$(curl -s --max-time 10 -X GET "http://localhost:7778/test" 2>/dev/null || echo "FAILED")
if ! echo "$response" | grep -q "successful"; then
    echo "[ERROR] Local CLOCQ server not responding. Start it first." | tee -a "$LOG"
    exit 1
fi
echo "[RETRAIN] CLOCQ OK" | tee -a "$LOG"

# Step 2: Silver annotation (generates annotated_train/dev/test.json)
# Skip if all three files already exist (resume after crash)
if [ -f "$ANNOTATED_TRAIN" ] && [ -f "$ANNOTATED_DEV" ]; then
    echo "[RETRAIN] Step 2: Silver annotation already done, skipping." | tee -a "$LOG"
else
    echo "[RETRAIN] Step 2: Silver annotation (train + dev + test)..." | tee -a "$LOG"
    PYTHONPATH="$EXPLAIGNN_ROOT/explaignn/distant_supervision:${PYTHONPATH:-}" \
        $PYTHON explaignn/distant_supervision/silver_annotation.py --annotate "$CONFIG_PATH" 2>&1 | tee "$SILVER_LOG"

    if [ ! -f "$ANNOTATED_TRAIN" ]; then
        echo "[ERROR] annotated_train.json not found after silver annotation. Check $SILVER_LOG" | tee -a "$LOG"
        exit 1
    fi
    echo "[RETRAIN] Silver annotation complete. annotated_train.json exists." | tee -a "$LOG"
fi

# Step 3: Train QU (BART SR) + ERS inference + GNN train
echo "[RETRAIN] Step 3: Training pipeline (QU → ERS → GNN)..." | tee -a "$LOG"
$PYTHON explaignn/pipeline.py --train "$CONFIG_REL" "$SOURCES" 2>&1 | tee -a "$LOG"
echo "[RETRAIN] Training complete at $(date)" | tee -a "$LOG"

# Step 4: Eval on test set (pred-answers: i=3, 500->100->20->a_pred)
echo "[RETRAIN] Step 4: Test inference (--pred-answers)..." | tee -a "$LOG"
$PYTHON explaignn/pipeline.py --pred-answers "$CONFIG_REL" "$SOURCES" 2>&1 | tee -a "$LOG"

echo "[RETRAIN] All done at $(date)" | tee -a "$LOG"
echo "[RETRAIN] Results in _intermediate_representations/convmix_retrain/" | tee -a "$LOG"