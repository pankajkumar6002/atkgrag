#!/usr/bin/env bash
# Queue RoG (GCR-Llama-2-7b-chat-hf) inference for CWQ and WebQSP p1/p4/p7.
# Waits for the e3_cwq_llama tmux session to finish before starting.
# GNN candidates are already pre-generated (cands.jsonl reused, no GNN re-run needed).
# GPU: use 2 (53+ GB free after WebQSP rerun completes; GPU 3 now occupied by smlab jobs).
source "$(dirname "$0")/../scripts/env.sh"
set -e
source "${CONDA_SH}"
conda activate fb-perturb

SCRIPT="${REPO_ROOT}/experiments/e3_llm_reasoning.py"
LOG_CWQ="${RESULTS_DIR}/e3_llm_reasoning/cwq_rog_pert.log"
LOG_WEBQSP="${RESULTS_DIR}/e3_llm_reasoning/webqsp_rog_pert.log"
GPU=2

echo "[$(date)] Waiting for e3_cwq_llama session to finish..."
until ! tmux has-session -t e3_cwq_llama 2>/dev/null; do
    sleep 60
done
echo "[$(date)] e3_cwq_llama done. Starting RoG inference on GPU ${GPU}."

# Check GPU has enough free memory for RoG (~15 GB needed)
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null || echo 0)
echo "[$(date)] GPU ${GPU} free memory: ${FREE_MB} MiB"
if [ "$FREE_MB" -lt 15000 ]; then
    echo "WARNING: GPU ${GPU} has only ${FREE_MB} MiB free (need ~15000). Proceeding anyway — may OOM."
fi

# Verify GNN candidates exist for all 6 runs before starting
for DS in cwq webqsp; do
    for PT in p1 p4 p7; do
        CANDS="${RESULTS_DIR}/e3_llm_reasoning/${DS}/gnn_candidates/pert_${PT}/cands.jsonl"
        if [ ! -s "$CANDS" ]; then
            echo "ERROR: $CANDS missing or empty. Aborting."
            exit 1
        fi
        echo "  ${DS} ${PT}: $(wc -l < "$CANDS") cands"
    done
done

echo "[$(date)] Running RoG on CWQ p1/p4/p7 (GPU ${GPU})..."
CUDA_VISIBLE_DEVICES=$GPU python3 "$SCRIPT" \
    --dataset cwq \
    --model rog \
    --pert-types p1 p4 p7 \
    --mode pert \
    2>&1 | tee "$LOG_CWQ"

echo "[$(date)] CWQ RoG done. Running RoG on WebQSP p1/p4/p7 (GPU ${GPU})..."
CUDA_VISIBLE_DEVICES=$GPU python3 "$SCRIPT" \
    --dataset webqsp \
    --model rog \
    --pert-types p1 p4 p7 \
    --mode pert \
    2>&1 | tee "$LOG_WEBQSP"

echo "[$(date)] All RoG inference complete."
echo "  CWQ log:    $LOG_CWQ"
echo "  WebQSP log: $LOG_WEBQSP"