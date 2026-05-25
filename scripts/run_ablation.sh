#!/usr/bin/env bash
# run_ablation.sh — unified ablation launcher (NSM + GraftNetV3, any ptypes)
#
# Key improvements over original:
#   - Parallel E1/E2 inference on separate GPU pairs (cuts inference time ~50%)
#   - Parallel NSM + GraftNetV3 subgraph builds when RAM allows (--parallel-build)
#   - Skip logic: skips build if gnn_input already exists, skips inference if
#     per_question.json already present (honours e1/e2 --skip-existing logic)
#   - Single entry point replacing run_remaining_p1.sh
#   - Auto-runs update_ablation_results.py on completion
#
# Usage:
#   PTYPES="p1"            bash run_ablation.sh           # p1 only, sequential build
#   PTYPES="p4 p7 p1"      bash run_ablation.sh           # full run (default)
#   PTYPES="p1" PARALLEL_BUILD=1 bash run_ablation.sh     # parallel NSM+GV3 builds
#   CELLS="graftnetv3_elq" PTYPES="p1" bash run_ablation.sh  # single cell resume
#
# Environment:
#   PTYPES           space-separated list (default: "p4 p7 p1")
#   CELLS            space-separated subset of cells (default: all 4)
#   WORKERS          build_ablation.py --workers (default: 8)
#   PARALLEL_BUILD   set to 1 to run NSM and GraftNetV3 builds concurrently
#                    (needs ~400 GB free RAM; safe when other jobs are light)
#   GPU_E1           CUDA devices for E1/CWQ inference  (default: "0,1")
#   GPU_E2           CUDA devices for E2/WebQSP inference (default: "2,3")
source "$(dirname "$0")/../scripts/env.sh"

set -euo pipefail
cd ${REPO_ROOT}

PYTHON=python3
BUILD=tmlr/scripts/build_ablation.py
E1=tmlr/scripts/experiments/e1_cwq_gnn_rag.py
E2=tmlr/scripts/experiments/e2_webqsp_gnn_rag.py
UPDATE=tmlr/scripts/update_ablation_results.py
LOG_DIR=tmlr/logs
mkdir -p "$LOG_DIR"

WORKERS=${WORKERS:-8}
PTYPES=${PTYPES:-"p4 p7 p1"}
CELLS=${CELLS:-"nsm_gem nsm_elq graftnetv3_gem graftnetv3_elq"}
PARALLEL_BUILD=${PARALLEL_BUILD:-0}
GPU_E1=${GPU_E1:-"0,1"}
GPU_E2=${GPU_E2:-"2,3"}

GNN_INPUT=new_exp/gnn_input
RES_E1=tmlr/results/e1_cwq_gnn_rag
RES_E2=tmlr/results/e2_webqsp_gnn_rag

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Helpers ───────────────────────────────────────────────────────────────────

subgraph_done() {
    # Returns 0 (true) if both CWQ and WebQSP gnn_input exist for this cell+ptype
    local cell=$1 ptype=$2
    local method=${cell%_*} seed=${cell##*_}
    [ -f "${GNN_INPUT}/cwq_${ptype}_${cell}/test_subgraph.json" ] && \
    [ -f "${GNN_INPUT}/webqsp_${ptype}_${cell}/test_subgraph.json" ]
}

infer_done() {
    local exp=$1 cell=$2 ptype=$3
    [ -f "${exp}/full_${cell}/p${ptype##p}_per_question.json" ] || \
    [ -f "${exp}/full_${cell}/${ptype}_per_question.json" ]
}

# ── Build one cell (sequential datasets) ──────────────────────────────────────

build_cell() {
    local cell=$1 ptype=$2
    local method=${cell%_*} seed=${cell##*_}
    if subgraph_done "$cell" "$ptype"; then
        log "SKIP build ${cell} ${ptype} — subgraphs already exist"
        return 0
    fi
    log "BUILD  ${cell}  ptype=${ptype}  workers=${WORKERS}"
    $PYTHON $BUILD \
        --method "$method" \
        --seed-mode "$seed" \
        --ptype "$ptype" \
        --dataset both \
        --workers "$WORKERS"
    log "BUILD done: ${cell} ${ptype}"
}

# ── Infer E1+E2 in parallel on different GPU pairs ───────────────────────────

infer_cell() {
    local cell=$1 ptype=$2
    local e1_done=0 e2_done=0
    infer_done "$RES_E1" "$cell" "$ptype" && e1_done=1
    infer_done "$RES_E2" "$cell" "$ptype" && e2_done=1

    if [ $e1_done -eq 1 ] && [ $e2_done -eq 1 ]; then
        log "SKIP infer ${cell} ${ptype} — both E1+E2 results exist"
        return 0
    fi

    log "INFER ${cell} ${ptype} — E1(GPUs ${GPU_E1}) + E2(GPUs ${GPU_E2}) in parallel"

    local pids=()
    if [ $e1_done -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=$GPU_E1 $PYTHON $E1 \
            --pert-type "$ptype" --el-source "$cell" --mode full \
            --out-dir "$RES_E1" \
            > "${LOG_DIR}/infer_e1_${cell}_${ptype}.log" 2>&1 &
        pids+=($!)
    fi
    if [ $e2_done -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=$GPU_E2 $PYTHON $E2 \
            --pert-type "$ptype" --el-source "$cell" --mode full \
            --out-dir "$RES_E2" \
            > "${LOG_DIR}/infer_e2_${cell}_${ptype}.log" 2>&1 &
        pids+=($!)
    fi

    # Wait for both, capture exit codes
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || { log "ERROR: inference PID $pid failed"; failed=1; }
    done
    [ $failed -eq 0 ] && log "INFER done: ${cell} ${ptype}" || return 1
}

# ── Main ──────────────────────────────────────────────────────────────────────

log "=== Ablation launcher ==="
log "    PTYPES=${PTYPES}  CELLS=${CELLS}  WORKERS=${WORKERS}"
log "    PARALLEL_BUILD=${PARALLEL_BUILD}  GPU_E1=${GPU_E1}  GPU_E2=${GPU_E2}"

for ptype in $PTYPES; do
    log "===== ptype=${ptype} ====="

    if [ "$PARALLEL_BUILD" = "1" ]; then
        # Build NSM and GraftNetV3 concurrently (needs ~400 GB free)
        log "Parallel build: NSM cells + GraftNetV3 cells simultaneously"
        local_pids=()
        for cell in $CELLS; do
            build_cell "$cell" "$ptype" &
            local_pids+=($!)
        done
        for pid in "${local_pids[@]}"; do wait "$pid" || true; done
    else
        # Sequential build (safer on memory)
        for cell in $CELLS; do
            build_cell "$cell" "$ptype"
        done
    fi

    # Inference: each cell E1+E2 runs in parallel, cells run sequentially
    for cell in $CELLS; do
        infer_cell "$cell" "$ptype"
    done

    log "===== ptype=${ptype} complete ====="
done

log "All cells done. Updating results..."
$PYTHON $UPDATE
log "sum.md and consolidated_results updated."
log "Results in:"
log "  ${RES_E1}/full_{nsm,graftnetv3}_{gem,elq}/"
log "  ${RES_E2}/full_{nsm,graftnetv3}_{gem,elq}/"
