# MetaQA: Non-Freebase Robustness Replication

MetaQA (WikiMovies KB) is the non-Freebase benchmark used to test whether the
failure modes identified on CWQ/WebQSP are schema-specific. This document
reproduces the full pipeline: data setup, backbone training, perturbation
generation for all seven types, and evaluation.

Reported in the paper: Appendix L (Table 17), with the cross-KB conclusion in
Section 6.

---

## 1. Data setup

MetaQA is used in the NSM/ReaRev preprocessed format (subgraph-per-question),
the same format GNN-RAG consumes for CWQ/WebQSP.

```bash
# Download the ReaRev preprocessed archives (2-hop and 3-hop)
# https://drive.google.com/drive/folders/1qRXeuoL-ArQY7pJFnMpNnBu0G-cOz6xv
mkdir -p data/metaqa
tar xzf metaqa-2hop.tgz -C data/metaqa/
tar xzf metaqa-3hop.tgz -C data/metaqa/
```

**Gotcha 1 : filenames.** The archives ship `train_simple.json` /
`dev_simple.json` / `test_simple.json`, but the loader expects
`train.json` / `dev.json` / `test.json`. Symlink them:

```bash
for hop in 2 3; do
  cd data/metaqa/metaqa-${hop}hop
  ln -sf train_simple.json train.json
  ln -sf dev_simple.json   dev.json
  ln -sf test_simple.json  test.json
  cd -
done
```

Expected per hop: `entities.txt`, `relations.txt`, `vocab_new.txt`,
`word_emb_300d.npy`, and the three `*_simple.json` splits.
Test-set sizes: 14,872 (2-hop), 14,274 (3-hop).

---

## 2. Train the ReaRev backbone

GNN-RAG ships no public MetaQA checkpoint, so the backbone is trained from
scratch, once per hop.

**Gotcha 2 : encoder flag.** `--lm lstm` is broken in the GNN-RAG fork of
ReaRev (`BaseInstruction` gained a `constraint` argument that only the BERT
path passes). Use `--lm sbert --relation_word_emb True`.

```bash
cd external/GNN-RAG/gnn

# 3-hop
CUDA_VISIBLE_DEVICES=0 python main.py ReaRev \
  --entity_dim 50 --num_epoch 10 --batch_size 32 --eval_every 2 \
  --data_folder <REPO_ROOT>/data/metaqa/metaqa-3hop/ \
  --lm sbert --num_iter 2 --num_ins 3 --num_gnn 3 \
  --relation_word_emb True \
  --word2id vocab_new.txt --word_emb_file word_emb_300d.npy \
  --experiment_name metaqa3 --name metaqa

# 2-hop: same command with --data_folder .../metaqa-2hop/ and --experiment_name metaqa2
```

Runtime is roughly 2 hours per hop on a single modern GPU. Checkpoints land in
`checkpoint/pretrain/`; the best-Hit@1 checkpoint (`*-h1.ckpt`) is used for all
evaluations below.

**Clean baselines obtained (used as the reference points in Table 17):**

| Split | Hit@1 | F1 | EM |
|---|---|---|---|
| MetaQA-2hop | 99.74 | 98.66 | 99.97 |
| MetaQA-3hop | 95.88 | 75.28 | 98.10 |

For reference, the GNN-RAG authors report 98.6 Hit@1 on MetaQA-3.

---

## 3. Generate the perturbation sets

All seven types use the same prompts and validity gates as the Freebase
experiments; only the knowledge base changes. Answer preservation is verified
directly against the MetaQA triple store, which replaces the SPARQL denotation
check used for Freebase.

### 3.1 LLM-generated types (CR, RS, S1-S4)

Requires a served Llama-3.3-70B-Instruct endpoint (OpenAI-compatible; we used
vLLM). Set `--api-base` accordingly.

```bash
python perturbation_scripts/metaqa_perturb_crrs.py \
  --hops 3 2 --sets p4 p7 p2 p3 p5 p6 \
  --full-merge \
  --api-base http://localhost:8010/v1 --model llama-3.3-70b
```

Gates (identical to the Freebase sets): BERTScore >= 0.85 (roberta-large),
perplexity ratio <= 2.5, entity-surface preservation, output must differ from
the original, single line. Four attempts, then fall back to the original
question (`is_perturbed: false`) : the same survivorship design as the main
paper, so reported severities are conservative lower bounds.

### 3.2 Entity Swap (ES / p1)

ES is deterministic and needs no LLM for the substitution itself: it infers the
question's relation-path signature from the KB, swaps in a same-type entity,
recomputes the gold answer, and rebuilds an answer-path-faithful subgraph.

```bash
python perturbation_scripts/metaqa_es_p1.py --hops 3 2 --full-merge
```

Note: the released ES JSON omits the rebuilt `subgraph` field to keep the
repository small. Re-running the command above regenerates it deterministically
(seeded), producing the exact sets used in the paper.

Outputs land in `data/perturbations/metaqa{2,3}hop_p{1..7}_full.json`.

---

## 4. Evaluate

### 4.1 Fixed-subgraph condition (Table 17)

Perturbs the question text while holding the retrieved subgraph fixed, which
isolates the decoder stage.

```bash
# Build NSM-format eval folders from the perturbed sets
python src/build_metaqa_eval_sets.py --hops 3 2 --sets p1 p2 p3 p4 p5 p6 p7 --full

# Evaluate a folder against the trained checkpoint
cd external/GNN-RAG/gnn
python main.py ReaRev --entity_dim 50 --batch_size 32 --eval_every 2 \
  --data_folder <REPO_ROOT>/data/metaqa/metaqa-3hop-p4-full/ \
  --lm sbert --num_iter 2 --num_ins 3 --num_gnn 3 --relation_word_emb True \
  --word2id vocab_new.txt --word_emb_file word_emb_300d.npy \
  --load_experiment metaqa3-h1.ckpt --is_eval --name metaqa
```

### 4.2 End-to-end (cascade) condition

Re-runs the entire pipeline on the perturbed text: entity linking, subgraph
construction, and GNN inference are all recomputed. Two seeding conditions are
supported: an automatic longest-match linker, and the gold topic-entity
annotations that MetaQA ships (which all published systems in this line
consume).

```bash
python src/build_metaqa_cascade_sets.py --hops 3 2 --sets p1 p2 p3 p4 p5 p6 p7
python src/metaqa_stage_isolation.py --hops 3 2
```

Only within-condition deltas are comparable across rows: absolute accuracies in
the cascade condition are bound to the reference linker and retriever, and are
therefore not comparable to the numbers obtained with the subgraphs shipped
with the dataset. ES is excluded from the cascade comparison because it changes
the gold answer by design.

---

## 5. Results

The 7 x 2 robustness table (all seven perturbation types, both hops, full test
sets) is reproduced in `results/METAQA_RESULTS.md` and reported as Table 17
(Appendix L) in the paper.
