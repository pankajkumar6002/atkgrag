#!/usr/bin/env bash
# End-to-end MetaQA replication: perturb -> build eval sets -> evaluate.
#
# Prerequisites (see data/METAQA.md):
#   1. MetaQA 2-hop/3-hop data extracted under data/metaqa/ (with the
#      train/dev/test.json symlinks created).
#   2. ReaRev backbone trained per hop (checkpoints in
#      external/GNN-RAG/gnn/checkpoint/pretrain/).
#   3. A served Llama-3.3-70B-Instruct endpoint for the LLM-generated types.
#
# Usage:
#   bash scripts/run_metaqa.sh                 # all seven types, both hops
#   API_BASE=http://host:8010/v1 bash scripts/run_metaqa.sh
set -euo pipefail

source "$(dirname "$0")/env.sh"

API_BASE="${API_BASE:-http://localhost:8010/v1}"
MODEL="${MODEL:-llama-3.3-70b}"
HOPS="${HOPS:-3 2}"
GNN_DIR="${REPO_ROOT}/external/GNN-RAG/gnn"

echo "== 1/4: generating LLM perturbation types (CR, RS, S1-S4) =="
python "${REPO_ROOT}/perturbation_scripts/metaqa_perturb_crrs.py" \
  --hops ${HOPS} --sets p4 p7 p2 p3 p5 p6 --full-merge \
  --api-base "${API_BASE}" --model "${MODEL}"

echo "== 2/4: generating Entity Swap (ES, deterministic) =="
python "${REPO_ROOT}/perturbation_scripts/metaqa_es_p1.py" \
  --hops ${HOPS} --full-merge

echo "== 3/4: building NSM-format evaluation folders =="
python "${REPO_ROOT}/src/build_metaqa_eval_sets.py" \
  --hops ${HOPS} --sets p1 p2 p3 p4 p5 p6 p7 --full

echo "== 4/4: evaluating each perturbed set against the trained checkpoint =="
cd "${GNN_DIR}"
for hop in ${HOPS}; do
  for p in p1 p2 p3 p4 p5 p6 p7; do
    echo "--- metaqa-${hop}hop-${p}-full"
    python main.py ReaRev \
      --entity_dim 50 --batch_size 32 --eval_every 2 \
      --data_folder "${REPO_ROOT}/data/metaqa/metaqa-${hop}hop-${p}-full/" \
      --lm sbert --num_iter 2 --num_ins 3 --num_gnn 3 \
      --relation_word_emb True \
      --word2id vocab_new.txt --word_emb_file word_emb_300d.npy \
      --load_experiment "metaqa${hop}-h1.ckpt" --is_eval --name metaqa
  done
done

echo "Done. Reference values: METAQA_RESULTS.md"
