#!/usr/bin/env bash
# run_webqsp_pathonly.sh — WebQSP RoG+PathOnly CR and RS (Table 5, two missing --- cells)
#
# Condition B: FineTune-RoG checkpoint + predicted rule paths injected at inference.
# GNN candidates already exist at:
#   tmlr/results/e3_llm_reasoning/webqsp/gnn_candidates/pert_p4/cands.jsonl
#   tmlr/results/e3_llm_reasoning/webqsp/gnn_candidates/pert_p7/cands.jsonl
# Rule paths from:
#   tmlr/Repo/GNN-RAG/llm/results/gen_rule_path/RoG-webqsp/RoG/test/predictions_3_False.jsonl
#
# Usage:
#   tmux new-session -d -s webqsp_pathonly
#   tmux send-keys -t webqsp_pathonly "bash run_webqsp_pathonly.sh 0 2>&1 | tee ${RESULTS_DIR}/e3_llm_reasoning/webqsp_FineTune-RoG-PathOnly_pert.log" Enter
#
# Expected output: webqsp_FineTune-RoG-PathOnly_summary.json with pert_p4 and pert_p7 entries
# Expected values: ~similar to CWQ PathOnly (51.43% CR, 51.29% RS) — WebQSP typically higher
source "$(dirname "$0")/../scripts/env.sh"

set -e

GPU="${1:-0}"
CONDA_ROOT="$(conda info --base)"
ENV="GNN-RAG"
SCRIPT="${REPO_ROOT}/experiments/e3_rog_paths_isolation.py"
LOG="${RESULTS_DIR}/e3_llm_reasoning/webqsp_FineTune-RoG-PathOnly_pert.log"

echo "[$(date)] WebQSP RoG+PathOnly CR+RS — GPU ${GPU}"
echo "[$(date)] Conda env: ${ENV}"

# Verify GNN candidates exist
for pt in p4 p7; do
    CANDS="${RESULTS_DIR}/e3_llm_reasoning/webqsp/gnn_candidates/pert_${pt}/cands.jsonl"
    if [ ! -s "$CANDS" ]; then
        echo "ERROR: GNN candidates missing: $CANDS"
        exit 1
    fi
    echo "  webqsp $pt: $(wc -l < $CANDS) cands ✓"
done

# Verify rule paths exist
RULE_PATHS="${GNN_RAG_ROOT}/llm/results/gen_rule_path/RoG-webqsp/RoG/test/predictions_3_False.jsonl"
if [ ! -s "$RULE_PATHS" ]; then
    echo "ERROR: Rule paths missing: $RULE_PATHS"
    exit 1
fi
echo "  rule paths: $(wc -l < $RULE_PATHS) lines ✓"

PYTHON="${CONDA_ROOT}/envs/${ENV}/bin/python"
echo "[$(date)] Using python: ${PYTHON}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__)" || { echo "ERROR: torch not found in ${ENV}"; exit 1; }

echo "[$(date)] Running WebQSP PathOnly p4 (CR) + p7 (RS)..."
"${PYTHON}" "${SCRIPT}" \
    --dataset webqsp \
    --pert-types p4 p7 \
    --mode pert \
    --gpu "${GPU}" \
    2>&1 | tee "${LOG}"

echo "[$(date)] Done. Check: tmlr/results/e3_llm_reasoning/webqsp_FineTune-RoG-PathOnly_summary.json"