#!/usr/bin/env bash
# Generate GNN candidates for webqsp p1/p4/p7 using fixed e3_gnn_worker (num_ins=2)
source "$(dirname "$0")/../scripts/env.sh"
set -e
source "${CONDA_SH}"
conda activate fb-perturb

SCRIPT_DIR="${REPO_ROOT}/experiments"
GNN_INPUT="${PERT_DATASET}/gnn_input"
CANDS_ROOT="${RESULTS_DIR}/e3_llm_reasoning/webqsp/gnn_candidates"

for PTYPE in p1 p4 p7; do
    echo "=========================================="
    echo "  WebQSP ${PTYPE} GNN candidates"
    echo "=========================================="

    INPUT="${GNN_INPUT}/webqsp_${PTYPE}/test_subgraph.json"
    OUTDIR="${CANDS_ROOT}/pert_${PTYPE}"
    mkdir -p "$OUTDIR"
    OUTPUT="${OUTDIR}/cands.jsonl"
    SAVE_INFO="${OUTDIR}/test.info"

    # Write temp input JSONL
    TMP_INPUT=$(mktemp /tmp/webqsp_${PTYPE}_input_XXXXXX.jsonl)
    cp "$INPUT" "$TMP_INPUT"

    python3 "$SCRIPT_DIR/e3_gnn_worker.py" \
        --dataset webqsp \
        --input "$TMP_INPUT" \
        --output "$OUTPUT" \
        --save-info "$SAVE_INFO"

    rm -f "$TMP_INPUT"
    echo "  Done: ${OUTPUT}"
    wc -l "$OUTPUT" 2>/dev/null || true
done

echo "All webqsp GNN candidates generated."