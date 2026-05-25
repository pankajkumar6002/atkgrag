#!/bin/bash
# run_subgraph_var_queue.sh — Queue all GNN-RAG + LLM variant runs for fixed subgraph_var p4/p7.
#
# GPU layout (GPU 2 skipped — busy with EPR):
#   GPU 1 — CWQ p4 GNN-RAG → E3 CWQ rog/rog+ra/llama
#   GPU 3 — CWQ p7 GNN-RAG → WebQSP p4+p7 GNN-RAG → E3 WebQSP rog/rog+ra/llama
#   GPU 0 — skipped (EPR WebQSP competing for memory, would OOM with batch=32)
#   GPU 2 — skipped (EPR CWQ)
#
# CWQ p4 and p7 GNN-RAG run in parallel on GPU 1 and GPU 3.
# After both finish, E3 CWQ runs sequentially on GPU 1 (rog → rog+ra → llama).
# WebQSP runs fully sequential on GPU 0.
# E3 LLM batch sizes increased: 32 for rog, 32 for llama (A100 80GB).
#
# Usage:
#   bash run_subgraph_var_queue.sh
#
# Results:
#   tmlr/results/e1_cwq_gnn_rag/full_graftnetv3_fixed/{p4,p7}_per_question.json
#   tmlr/results/e2_webqsp_gnn_rag/full_graftnetv3_fixed/{p4,p7}_per_question.json
#   tmlr/results/e3_llm_reasoning/{cwq,webqsp}/{FineTune-RoG,FineTune-RoG-RA,Llama-3.1-8B}_fixed/
source "$(dirname "$0")/../scripts/env.sh"

set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH=""${CONDA_SH}""
ENV="GNN-RAG"
SUBGRAPH_VAR="${PERT_DATASET}/subgraph_var/gnn_input"
E1_OUT="${RESULTS_DIR}/e1_cwq_gnn_rag"
E2_OUT="${RESULTS_DIR}/e2_webqsp_gnn_rag"
SESSION="subgraph_var_fixed"
PREAMBLE="source ${CONDA_SH} && conda activate ${ENV} && cd ${REPO_ROOT}"

# ── Window 0: CWQ p4 on GPU 1 ───────────────────────────────────────────────
# Runs GNN-RAG for CWQ p4, then waits for the CWQ p7 window (GPU 3) to finish
# by polling for its sentinel file, then runs all E3 CWQ variants.
CWQ_P4_CMDS=(
  "echo '=== [GPU 1] E1 CWQ p4 graftnetv3_fixed ==='"
  "CUDA_VISIBLE_DEVICES=1 python ${SCRIPTS}/e1_cwq_gnn_rag.py --pert-type p4 --el-source graftnetv3_fixed --out-dir ${E1_OUT} --n-boot 1000"
  "touch /tmp/gv3_fixed_cwq_p4_done"
  # Wait for CWQ p7 (GPU 3 window) to finish before E3
  "echo 'Waiting for CWQ p7 GNN-RAG to finish...'"
  "until [ -f /tmp/gv3_fixed_cwq_p7_done ]; do sleep 15; done"
  "echo '=== [GPU 1] E3 CWQ rog fixed (batch=32) ==='"
  "CUDA_VISIBLE_DEVICES=1 python ${SCRIPTS}/e3_llm_reasoning.py --dataset cwq --model rog --pert-types p4 p7 --mode pert --subgraph-dir ${SUBGRAPH_VAR} --cand-tag gnn_candidates_fixed --out-tag _fixed --batch-size 32 --gpu 1"
  "echo '=== [GPU 1] E3 CWQ rog+ra fixed (batch=32) ==='"
  "CUDA_VISIBLE_DEVICES=1 python ${SCRIPTS}/e3_llm_reasoning.py --dataset cwq --model rog --add-rule --pert-types p4 p7 --mode pert --subgraph-dir ${SUBGRAPH_VAR} --cand-tag gnn_candidates_fixed --out-tag _fixed --batch-size 32 --gpu 1"
  "echo '=== [GPU 1] E3 CWQ llama fixed (batch=32) ==='"
  "CUDA_VISIBLE_DEVICES=1 python ${SCRIPTS}/e3_llm_reasoning.py --dataset cwq --model llama --pert-types p4 p7 --mode pert --subgraph-dir ${SUBGRAPH_VAR} --cand-tag gnn_candidates_fixed --out-tag _fixed --batch-size 32 --gpu 1"
  "rm -f /tmp/gv3_fixed_cwq_p4_done /tmp/gv3_fixed_cwq_p7_done"
  "echo 'CWQ ALL DONE'"
)

# ── Window 1: CWQ p7 on GPU 3 ───────────────────────────────────────────────
CWQ_P7_CMDS=(
  "echo '=== [GPU 3] E1 CWQ p7 graftnetv3_fixed ==='"
  "CUDA_VISIBLE_DEVICES=3 python ${SCRIPTS}/e1_cwq_gnn_rag.py --pert-type p7 --el-source graftnetv3_fixed --out-dir ${E1_OUT} --n-boot 1000"
  "touch /tmp/gv3_fixed_cwq_p7_done"
  "echo 'CWQ p7 GNN-RAG done — E3 will run in cwq_p4 window'"
)

# ── Window 2: WebQSP on GPU 0 ────────────────────────────────────────────────
WSP_CMDS=(
  "echo '=== [GPU 0] E2 WebQSP p4 graftnetv3_fixed ==='"
  "CUDA_VISIBLE_DEVICES=3 python ${SCRIPTS}/e2_webqsp_gnn_rag.py --pert-type p4 --el-source graftnetv3_fixed --out-dir ${E2_OUT} --n-boot 1000"
  "echo '=== [GPU 3] E2 WebQSP p7 graftnetv3_fixed ==='"
  "CUDA_VISIBLE_DEVICES=3 python ${SCRIPTS}/e2_webqsp_gnn_rag.py --pert-type p7 --el-source graftnetv3_fixed --out-dir ${E2_OUT} --n-boot 1000"
  "echo '=== [GPU 3] E3 WebQSP rog fixed (batch=32) ==='"
  "CUDA_VISIBLE_DEVICES=3 python ${SCRIPTS}/e3_llm_reasoning.py --dataset webqsp --model rog --pert-types p4 p7 --mode pert --subgraph-dir ${SUBGRAPH_VAR} --cand-tag gnn_candidates_fixed --out-tag _fixed --batch-size 32 --gpu 3"
  "echo '=== [GPU 3] E3 WebQSP rog+ra fixed (batch=32) ==='"
  "CUDA_VISIBLE_DEVICES=3 python ${SCRIPTS}/e3_llm_reasoning.py --dataset webqsp --model rog --add-rule --pert-types p4 p7 --mode pert --subgraph-dir ${SUBGRAPH_VAR} --cand-tag gnn_candidates_fixed --out-tag _fixed --batch-size 32 --gpu 3"
  "echo '=== [GPU 3] E3 WebQSP llama fixed (batch=32) ==='"
  "CUDA_VISIBLE_DEVICES=3 python ${SCRIPTS}/e3_llm_reasoning.py --dataset webqsp --model llama --pert-types p4 p7 --mode pert --subgraph-dir ${SUBGRAPH_VAR} --cand-tag gnn_candidates_fixed --out-tag _fixed --batch-size 32 --gpu 3"
  "echo 'WebQSP ALL DONE'"
)

# ── Build chained command strings ─────────────────────────────────────────────
join_cmds() {
  local IFS=$'\n'
  local result="$1"; shift
  for cmd in "$@"; do result="${result} && ${cmd}"; done
  echo "$result"
}

CWQ_P4_CHAIN=$(join_cmds "${PREAMBLE}" "${CWQ_P4_CMDS[@]}")
CWQ_P7_CHAIN=$(join_cmds "${PREAMBLE}" "${CWQ_P7_CMDS[@]}")
WSP_CHAIN=$(join_cmds "${PREAMBLE}" "${WSP_CMDS[@]}")

# ── Clean sentinel files from any previous run ────────────────────────────────
rm -f /tmp/gv3_fixed_cwq_p4_done /tmp/gv3_fixed_cwq_p7_done

# ── Launch tmux session ───────────────────────────────────────────────────────
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -x 220 -y 50

# Window 0: CWQ p4 + E3 CWQ (GPU 1)
tmux rename-window -t "${SESSION}:0" "cwq_p4_e3"
tmux send-keys -t "${SESSION}:cwq_p4_e3" "${CWQ_P4_CHAIN}; echo 'CWQ_P4_E3 WINDOW DONE'" Enter

# Window 1: CWQ p7 (GPU 3)
tmux new-window -t "${SESSION}" -n "cwq_p7"
tmux send-keys -t "${SESSION}:cwq_p7" "${CWQ_P7_CHAIN}; echo 'CWQ_P7 WINDOW DONE'" Enter

# Window 2: WebQSP (GPU 0)
tmux new-window -t "${SESSION}" -n "wqsp"
tmux send-keys -t "${SESSION}:wqsp" "${WSP_CHAIN}; echo 'WQSP WINDOW DONE'" Enter

echo "Launched tmux session '${SESSION}' with 3 parallel windows:"
echo "  cwq_p4_e3  (GPU 1): E1 CWQ p4 → [waits for p7] → E3 CWQ rog/rog+ra/llama (batch=32)"
echo "  cwq_p7     (GPU 3): E1 CWQ p7  → signals p7 done"
echo "  wqsp       (GPU 0): E2 WebQSP p4+p7 → E3 WebQSP rog/rog+ra/llama (batch=32)"
echo ""
echo "Monitor:"
echo "  tmux attach -t ${SESSION}"
echo "  tmux select-window -t ${SESSION}:cwq_p4_e3  && tmux attach -t ${SESSION}"
echo ""
echo "Results:"
echo "  ${E1_OUT}/full_graftnetv3_fixed/"
echo "  ${E2_OUT}/full_graftnetv3_fixed/"
echo "  tmlr/results/e3_llm_reasoning/{cwq,webqsp}/{FineTune-RoG,FineTune-RoG-RA,Llama-3.1-8B}_fixed/"