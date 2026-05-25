#!/usr/bin/env bash
# run_lightprof.sh — Train and evaluate LightPROF on CWQ/WebQSP perturbed data.
#
# Pipeline:
#   1. Train HopPredictor on WebQSP + CWQ (~few min each)
#   2. Train Knowledge Adapter on orig train JSONL (~GPU hours)
#   3. Evaluate adapter on p1/p4/p7 perturbed test sets
#
# GPU: 3 (81 GB free). LLaMA-3.1-8B frozen (fp16), adapter ~1M params.
# Prereqs: freebase_triples_full.tsv + *_test.jsonl must exist.
#   Run first: python tmlr/scripts/build_lightprof_data.py --step all
source "$(dirname "$0")/../scripts/env.sh"

set -e
source "${CONDA_SH}"
conda activate fb-perturb

LIGHTPROF_DIR="${EXTERNAL_DIR}/LightPROF"
RESULTS_DIR="${RESULTS_DIR}/lightprof"
GPU=3
CKPT_WEBQSP="$RESULTS_DIR/checkpoints/webqsp"
CKPT_CWQ="$RESULTS_DIR/checkpoints/cwq"

mkdir -p "$RESULTS_DIR/checkpoints"
cd "$LIGHTPROF_DIR"

# ---------------------------------------------------------------------------
# Step 1: Train HopPredictor (fast — BERT linear head, few min each)
# ---------------------------------------------------------------------------
echo "[$(date)] Training HopPredictor (WebQSP) ..."
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/train_hop_predictor.py \
    --data data/hop_labels_webqsp.json \
    --output "$RESULTS_DIR/checkpoints/hop_predictor_webqsp.pt" \
    --epochs 5 --batch-size 32 --lr 2e-5 \
    2>&1 | tee "$RESULTS_DIR/hop_predictor_webqsp.log"

echo "[$(date)] Training HopPredictor (CWQ) ..."
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/train_hop_predictor.py \
    --data data/hop_labels_cwq.json \
    --output "$RESULTS_DIR/checkpoints/hop_predictor_cwq.pt" \
    --epochs 5 --batch-size 32 --lr 2e-5 \
    2>&1 | tee "$RESULTS_DIR/hop_predictor_cwq.log"

# ---------------------------------------------------------------------------
# Step 2: Train Knowledge Adapter on WebQSP original questions
# ---------------------------------------------------------------------------
echo "[$(date)] Training Knowledge Adapter on WebQSP orig ..."
CUDA_VISIBLE_DEVICES=$GPU python3 lightprof/train.py \
    --config configs/perturbed/webqsp_perturbed.toml \
    --checkpoint-dir "$CKPT_WEBQSP" \
    2>&1 | tee "$RESULTS_DIR/webqsp_train.log"

# ---------------------------------------------------------------------------
# Step 3: Evaluate WebQSP p1/p4/p7 (each needs test_path in TOML config)
# ---------------------------------------------------------------------------
echo "[$(date)] Evaluating WebQSP p1/p4/p7 ..."
for PT in p1 p4 p7; do
    # Generate a temp config with this perturbation's test path
    EVAL_CFG="/tmp/lightprof_webqsp_${PT}.toml"
    sed "s|test_path = .*|test_path = \"data/webqsp_${PT}_test.jsonl\"|" \
        configs/perturbed/webqsp_perturbed.toml > "$EVAL_CFG"

    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/evaluate.py \
        --config "$EVAL_CFG" \
        --checkpoint "$CKPT_WEBQSP/final_adapter.pt" \
        --dataset webqsp \
        --output "$RESULTS_DIR/webqsp_${PT}_results.json" \
        2>&1 | tee "$RESULTS_DIR/webqsp_${PT}_eval.log"

    echo "[$(date)] WebQSP ${PT}: $(grep 'Final Hits@1' "$RESULTS_DIR/webqsp_${PT}_eval.log" | tail -1)"
done

# ---------------------------------------------------------------------------
# Step 4: Train Knowledge Adapter on CWQ original questions
# ---------------------------------------------------------------------------
echo "[$(date)] Training Knowledge Adapter on CWQ orig ..."
CUDA_VISIBLE_DEVICES=$GPU python3 lightprof/train.py \
    --config configs/perturbed/cwq_perturbed.toml \
    --checkpoint-dir "$CKPT_CWQ" \
    2>&1 | tee "$RESULTS_DIR/cwq_train.log"

# ---------------------------------------------------------------------------
# Step 5: Evaluate CWQ p1/p4/p7
# ---------------------------------------------------------------------------
echo "[$(date)] Evaluating CWQ p1/p4/p7 ..."
for PT in p1 p4 p7; do
    EVAL_CFG="/tmp/lightprof_cwq_${PT}.toml"
    sed "s|test_path = .*|test_path = \"data/cwq_${PT}_test.jsonl\"|" \
        configs/perturbed/cwq_perturbed.toml > "$EVAL_CFG"

    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/evaluate.py \
        --config "$EVAL_CFG" \
        --checkpoint "$CKPT_CWQ/final_adapter.pt" \
        --dataset cwq \
        --output "$RESULTS_DIR/cwq_${PT}_results.json" \
        2>&1 | tee "$RESULTS_DIR/cwq_${PT}_eval.log"

    echo "[$(date)] CWQ ${PT}: $(grep 'Final Hits@1' "$RESULTS_DIR/cwq_${PT}_eval.log" | tail -1)"
done

echo "[$(date)] LightPROF pipeline complete. Results in $RESULTS_DIR"