#!/usr/bin/env bash
# run_s1s4_all.sh
# ================
# Full pipeline for S1-S4 secondary perturbation ablation.
# Runs in order: subgraph build → GNN inference → PPR analysis → fast-path EM
#
# Prerequisites:
#   - SPARQL endpoint running (Virtuoso on :8890)
#   - Freebase fb_en.txt at data/resources/fb_en.txt
#   - GNN-RAG ReaRev checkpoint at external/GNN-RAG/data/pretrain/ReaRev_CWQ.ckpt
#   - NSM checkpoint at external/EPR-KGQA/NSM_H/checkpoint/pretrain/CWQ_nsm-h1.ckpt
#
# Total runtime estimate: ~48-72h (subgraph BFS dominates)
# Can run steps independently; each script has --resume / skip logic.
source "$(dirname "$0")/../scripts/env.sh"

set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="${LOGS_DIR}"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "======================================================================"
echo "S1-S4 Secondary Perturbation Full Pipeline  (${TS})"
echo "======================================================================"

# ── STEP 1: Build NSM and GraftNet subgraphs for S1-S4 ─────────────────────
# Run p2 and p3 first (2-hop NSM BFS is faster than 4-hop CWQ BFS),
# then p5 and p6. Within each ptype, nsm is faster than graftnetv3.
# Estimated: ~6h per (ptype × dataset × method) cell → ~48h for all 8 gem cells.
#
# Option A: Run all cells (recommended, ~48-72h):
# python exp_s1s4_subgraph_build.py --ptype all --dataset both --method both --seed-mode gem \
#     2>&1 | tee "$LOG_DIR/s1s4_build_all_${TS}.log"
#
# Option B: Run one ptype at a time (for better checkpointing):

echo ""
echo "STEP 1: Subgraph construction (NSM + GraftNet, GEM seeds, P2/P3/P5/P6)"
echo "Estimated: ~2-6h per cell | 8 gem cells total = ~24-48h"
echo "----------------------------------------------------------------------"

for PTYPE in p2 p3 p5 p6; do
    for DATASET in cwq webqsp; do
        for METHOD in nsm graftnetv3; do
            echo ""
            echo "[BUILD] ${DATASET}/${PTYPE}/${METHOD}_gem"
            python exp_s1s4_subgraph_build.py \
                --ptype "$PTYPE" \
                --dataset "$DATASET" \
                --method "$METHOD" \
                --seed-mode gem \
                --workers 8 \
                --bfs-workers 6 \
                2>&1 | tee -a "$LOG_DIR/s1s4_build_${TS}.log"
        done
    done
done

# ── STEP 2: GNN-RAG inference for all S1-S4 NSM/GraftNet cells ─────────────
echo ""
echo "STEP 2: GNN-RAG inference for S1-S4 subgraph variants"
echo "Estimated: ~30-90min per cell (GPU-bound) | 8 cells = ~8-12h"
echo "----------------------------------------------------------------------"

python exp_s1s4_inference.py \
    --ptype all \
    --dataset both \
    --el-source both_gem \
    2>&1 | tee "$LOG_DIR/s1s4_inference_${TS}.log"

# ── STEP 3: Fast-path EM for S1 and S4 (fills Table 5 gaps) ────────────────
echo ""
echo "STEP 3: Fast-path gold-MID EM for S1 (p2) and S4 (p6) — Table 5"
echo "Estimated: ~30min per cell (4 GPU runs) = ~2h"
echo "----------------------------------------------------------------------"

python exp_fast_path_all.py \
    --ptype p2 p6 \
    --dataset both \
    --exp4 \
    2>&1 | tee "$LOG_DIR/s1s4_fast_path_${TS}.log"

# ── STEP 4: PPR mass analysis (mechanistic CR analysis) ─────────────────────
echo ""
echo "STEP 4: PPR mass / answer reachability breakdown (Part A, ~5min)"
echo "----------------------------------------------------------------------"

python exp_ppr_mass.py \
    --part A \
    --ptype p4 \
    --variants-a elq nsm_gem graftnetv3_gem \
    2>&1 | tee "$LOG_DIR/ppr_mass_${TS}.log"

# ── STEP 5: NSM decoder isolation (EXP-3, GPU required) ────────────────────
echo ""
echo "STEP 5: NSM decoder isolation for CR (200-question sample)"
echo "Estimated: ~2h (GPU inference)"
echo "----------------------------------------------------------------------"

python exp_nsm_isolation.py \
    --dataset cwq \
    --ptype p4 \
    --sample 200 \
    2>&1 | tee "$LOG_DIR/nsm_isolation_${TS}.log"

# ── STEP 6: Cross-attack subgraph transfer (Experiment #3) ─────────────────
echo ""
echo "STEP 6: Cross-attack transfer — CR-Q+RS-SG, RS-Q+CR-SG, S3-Q+RS-SG"
echo "Estimated: ~3-6h (3 cells x 2 GPU passes each)"
echo "----------------------------------------------------------------------"

python exp_cross_attack_transfer.py \
    --cell all \
    --dataset both \
    --sg-el-source elq \
    2>&1 | tee "$LOG_DIR/cross_attack_${TS}.log"

echo ""
echo "======================================================================"
echo "ALL STEPS COMPLETE"
echo "Logs: $LOG_DIR/s1s4_*_${TS}.log  $LOG_DIR/cross_attack_${TS}.log"
echo ""
echo "NEXT: Update LaTeX"
echo "  1. ablate.tex Table (tab:secondary): add NSM and GraftNet columns"
echo "  2. exp.tex Table 5 (tab:rearev_hits): add S1, S4 rows (fast-path EM)"
echo "  3. exp.tex Table 5: add NSM row from nsm_isolation result"
echo "  4. analysis.tex: add PPR mass breakdown from ppr_mass_cwq_p4.json"
echo "  5. analysis.tex: add cross-attack transfer oracle (3A EM vs pipeline EM)"
echo "======================================================================"
