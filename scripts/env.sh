#!/usr/bin/env bash
# Source this file from any shell script in scripts/ or perturbation_scripts/
# to get REPO_ROOT and all derived paths.
#
# Usage:  source "$(dirname "$0")/../scripts/env.sh"
#    or:  source "$(dirname "$0")/env.sh"   (if already in scripts/)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Data ---
PERT_DATASET="${REPO_ROOT}/data/perturbations"
PERT_RAW_DIR="${REPO_ROOT}/data/perturbations_raw"
RESOURCES_DIR="${REPO_ROOT}/data/resources"
MID_CACHE_DIR="${REPO_ROOT}/data/mid_cache"
CWQ_NXHD_DIR="${REPO_ROOT}/data/cwq_4hop_nxhd"
NEW_SEED_NXHD_DIR="${REPO_ROOT}/data/new_seed_nxhd"
PLAN_B_DIR="${REPO_ROOT}/data/plan_b"
FB_2HOPS_NEIGHBORHOODS="${REPO_ROOT}/data/freebase_2hops/stagg.neighborhoods"

# --- External repos ---
EXTERNAL_DIR="${REPO_ROOT}/external"
GNN_RAG_ROOT="${EXTERNAL_DIR}/GNN-RAG"
GNN_RAG_DATA="${GNN_RAG_ROOT}/data"
BLINK_ROOT="${EXTERNAL_DIR}/BLINK"
GMT_KBQA_ROOT="${EXTERNAL_DIR}/GMT-KBQA"
GMT_KBQA_DATA="${EXTERNAL_DIR}/GMT-KBQA-data"
EXPLAIGNN_ROOT="${EXTERNAL_DIR}/EXPLAIGNN"
EPR_KGQA_ROOT="${EXTERNAL_DIR}/EPR-KGQA"

# --- Models ---
MODELS_DIR="${REPO_ROOT}/models"

# --- Outputs ---
RESULTS_DIR="${REPO_ROOT}/results"
LOGS_DIR="${REPO_ROOT}/logs"

# --- Conda ---
# Auto-detect conda.sh; override by setting CONDA_SH before sourcing this file
if [ -z "${CONDA_SH:-}" ]; then
    CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi

# --- Python interpreters for conda envs (override via env vars) ---
PYTHON_EXP="${PYTHON_EXP:-python}"   # explaignn env
PYTHON_GNN="${PYTHON_GNN:-python}"   # GNN-RAG env
EPR_PYTHON="${EPR_PYTHON:-python}"   # EPR-KGQA env

export REPO_ROOT PERT_DATASET PERT_RAW_DIR RESOURCES_DIR MID_CACHE_DIR \
       CWQ_NXHD_DIR NEW_SEED_NXHD_DIR PLAN_B_DIR FB_2HOPS_NEIGHBORHOODS \
       EXTERNAL_DIR GNN_RAG_ROOT GNN_RAG_DATA BLINK_ROOT GMT_KBQA_ROOT \
       GMT_KBQA_DATA EXPLAIGNN_ROOT EPR_KGQA_ROOT MODELS_DIR RESULTS_DIR \
       LOGS_DIR CONDA_SH PYTHON_EXP PYTHON_GNN EPR_PYTHON
