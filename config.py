"""
Central path configuration for CascadeAttack.
All paths are derived from REPO_ROOT (the forgejo/ directory).
External repos are expected under external/; data under data/.
Override any path via environment variables of the same name.
"""
import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("CASCADEATTACK_ROOT", str(Path(__file__).resolve().parent)))

# --- Data ---
PERT_DATASET      = REPO_ROOT / "data" / "perturbations"       # canonical p1-p7 sets
PERT_RAW_DIR      = REPO_ROOT / "data" / "perturbations_raw"   # temperature-variant raw sets
RESOURCES_DIR     = REPO_ROOT / "data" / "resources"
MID_CACHE_DIR     = REPO_ROOT / "data" / "mid_cache"
CWQ_NXHD_DIR      = REPO_ROOT / "data" / "cwq_4hop_nxhd"
NEW_SEED_NXHD_DIR = REPO_ROOT / "data" / "new_seed_nxhd"
PLAN_B_DIR        = REPO_ROOT / "data" / "plan_b"
FB_2HOPS_NEIGHBORHOODS = REPO_ROOT / "data" / "freebase_2hops" / "stagg.neighborhoods"

# --- External repos (clone into external/) ---
EXTERNAL_DIR    = REPO_ROOT / "external"
GNN_RAG_ROOT    = EXTERNAL_DIR / "GNN-RAG"
GNN_RAG_DATA    = GNN_RAG_ROOT / "data"
BLINK_ROOT      = EXTERNAL_DIR / "BLINK"
GMT_KBQA_ROOT   = EXTERNAL_DIR / "GMT-KBQA"
GMT_KBQA_DATA   = EXTERNAL_DIR / "GMT-KBQA-data"
EXPLAIGNN_ROOT  = EXTERNAL_DIR / "EXPLAIGNN"
EPR_KGQA_ROOT   = EXTERNAL_DIR / "EPR-KGQA"
UNIQKGQA_ROOT   = EXTERNAL_DIR / "UniKGQA"
ROG_ROOT        = EXTERNAL_DIR / "reasoning-on-graphs"

# --- Models (download separately, see LARGE_FILES.md) ---
MODELS_DIR      = REPO_ROOT / "models"

# --- Outputs ---
RESULTS_DIR     = REPO_ROOT / "results"
LOGS_DIR        = REPO_ROOT / "logs"

# --- Python interpreters for conda envs (override via env vars) ---
PYTHON_EXP = os.environ.get("PYTHON_EXP", "python")   # explaignn env
PYTHON_GNN = os.environ.get("PYTHON_GNN", "python")   # GNN-RAG env
EPR_PYTHON = os.environ.get("EPR_PYTHON", "python")   # EPR-KGQA env
