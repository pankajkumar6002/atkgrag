#!/bin/bash
# Usage: ./run_build_subgraphs.sh [dataset] [stage]
#   dataset: cwq | webqsp | both  (default: both)
#   stage:   1 | 2 | 3 | all     (default: all)
#
# Examples:
#   ./run_build_subgraphs.sh both all        # full pipeline
#   ./run_build_subgraphs.sh cwq 1           # Stage 1 only for CWQ
#   ./run_build_subgraphs.sh webqsp 2        # Stage 2 only for WebQSP
#   ./run_build_subgraphs.sh both all --dry-run   # dry run (extra args passed through)
source "$(dirname "$0")/../scripts/env.sh"

set -euo pipefail

DATASET=${1:-both}
STAGE=${2:-all}
shift 2 2>/dev/null || true   # remaining args passed through (e.g. --dry-run)

BASE=${REPO_ROOT}
LOG_DIR="${LOGS_DIR}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/build_subgraphs_$(date +%Y%m%d_%H%M%S).log"

echo "=== build_subgraphs pipeline ===" | tee "${LOG_FILE}"
echo "Dataset: ${DATASET} | Stage: ${STAGE} | Extra: $*" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

python "${REPO_ROOT}/perturbation_scripts/build_subgraphs.py" \
    --dataset "${DATASET}" \
    --stage "${STAGE}" \
    --workers-s1 20 \
    --workers-s2 40 \
    --fb-file "${BASE}/freebase/data/fb_en.txt" \
    --tsv-path "${RESOURCES_DIR}/entities_sup_names.tsv" \
    "$@" \
    2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo "" | tee -a "${LOG_FILE}"
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "=== DONE (exit 0) ===" | tee -a "${LOG_FILE}"
else
    echo "=== FAILED (exit ${EXIT_CODE}) ===" | tee -a "${LOG_FILE}"
fi
exit "${EXIT_CODE}"
