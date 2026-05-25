# Query-Side Attacks on GNN-Based KGQA: Tracing Failures Through the Pipeline

> **Anonymous submission** — under review. Code provided for reproducibility.  
> Anonymous repository: `https://anonymous.4open.science/r/cascadeattack`

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Knowledge graph question answering (KGQA) systems based on graph neural networks (GNNs) are vulnerable to **query-side adversarial attacks** — perturbations applied only to the input question, with no access to the KG, model weights, or training data.

We introduce **CascadeAttack**, the first systematic, stage-resolved robustness study of a GNN-based KGQA pipeline. Our **stage-isolation protocol** fixes upstream outputs and perturbs only the query to attribute failures precisely to each pipeline stage (entity linking → subgraph retrieval → GNN reasoning → answer generation).

**Two primary attack types:**

- **Compositional Restructuring (CR)** — rewrites the question via hop-order reversal or constraint injection, leaving all entity mentions intact and preserving the SPARQL answer. Targets multi-hop GNN reasoning.
- **Relation Synonym Swap (RS)** — replaces predicate surface forms with synonyms, keeping entity mentions unchanged. Targets GNN relation attention.

**Key findings:**

- Under CR, 74.0% of ELQ subgraphs contain the gold answer (identical to clean coverage) yet GNN-RAG achieves only 0.68% EM on CWQ — the GNN decoder fails, not entity linking.
- Stage isolation shows the GNN decoder retains 52.76% CWQ EM when the subgraph retrieval stage is held fixed under RS — subgraph retrieval is the primary bottleneck.
- Our question-embedding-weighted PPR variant (GraftNet) reorients PPR edge weights toward restructured queries, recovering substantial accuracy.

---

## Repository Structure

```
cascadeattack/           ← repo root (this directory)
│
├── config.py            ← central path config (all paths derive from here)
├── requirements.txt
├── environment.yml
├── LARGE_FILES.md       ← instructions for downloading large files
│
├── experiments/         ← experiment runners
│   ├── metrics.py             # Δ_EL, Δ_SR, Δ_EA, Δ_AG implementations
│   ├── e1_cwq_gnn_rag.py      # E1: stage-wise attack, CWQ, GNN-RAG
│   ├── e2_webqsp_gnn_rag.py   # E2: stage-wise attack, WebQSP, GNN-RAG
│   ├── e1_e2_gmt_kbqa.py      # E1/E2: GMT-KBQA pipeline
│   ├── e3_gnn_worker.py       # E3: GNN stage isolation
│   ├── e3_llm_reasoning.py    # E3: LLM reasoning stage
│   ├── e35_gnn_norms.py       # E35: GNN weight norms
│   ├── e5_validity.py         # E5: perturbation validity (PPL + BERTScore)
│   ├── exp_s1s4_*.py          # Ablation: subgraph and inference variants
│   ├── exp_cross_attack_transfer.py
│   └── collect_results.py
│
├── src/                 ← pipeline utilities (entity linking, subgraph builders, etc.)
│   ├── el_agent.py            # LLM-based entity linking agent
│   ├── el_elq.py              # ELQ entity linker wrapper
│   ├── seed_entity_agent_llama.py
│   ├── build_subgraph_nxhd.py
│   ├── build_gnn_input_from_pkl.py
│   ├── build_p7_graftnet_subgraph.py
│   ├── dense_retrieval.py
│   ├── compute_all_results.py
│   └── ...
│
├── perturbation_scripts/ ← generate perturbation datasets
│   ├── prepare_data_llama.py  # generate via Llama-3.1-8B
│   ├── llama3.28b.py
│   ├── epr_pert_experiment.py
│   └── graphrag_pert_experiment.py
│
├── scripts/             ← shell runners for experiments
│   ├── env.sh                 # auto-detected paths (sourced by all scripts)
│   ├── run_e1_cwq.sh
│   ├── run_e2_webqsp.sh
│   ├── run_e35.sh
│   ├── run_e5.sh
│   └── ...
│
├── external/            ← clone target model repos here (see Setup §3)
├── data/                ← datasets and preprocessed files (see data/README.md)
│   ├── perturbations/         # canonical p1–p7 perturbation sets
│   ├── perturbations_raw/     # temperature-variant raw sets
│   ├── resources/             # Freebase entity files
│   ├── mid_cache/
│   ├── cwq_4hop_nxhd/
│   └── freebase_2hops/
├── models/              ← LLM weights (downloaded separately)
└── results/             ← experiment outputs (created on run)
```

---

## Setup

### 1. Clone this repository

```bash
git clone https://anonymous.4open.science/r/cascadeattack
cd cascadeattack
```

### 2. Create the Python environment

```bash
conda env create -f environment.yml
conda activate cascadeattack
```

Or with pip:

```bash
pip install -r requirements.txt
```

> **Path configuration is automatic.** `config.py` and `scripts/env.sh` compute all paths relative to the repo root using `Path(__file__).resolve()`. No manual path editing is needed as long as you follow the directory layout below.

---

### 3. Clone external model repositories

The experiments test three KGQA pipelines as attack targets. Clone each into `external/` inside the repo:

```bash
mkdir -p external

# GNN-RAG (ReaRev) — primary target
git clone https://github.com/cmavro/GNN-RAG.git            external/GNN-RAG

# GMT-KBQA (T5-based) — second target
git clone https://github.com/fc-liu/GMT-KBQA.git            external/GMT-KBQA

# UniKGQA (NSM baseline)
git clone https://github.com/RUCKBReasoning/UniKGQA.git      external/UniKGQA

# Reasoning on Graphs (RoG)
git clone https://github.com/rmanluo/reasoning-on-graphs.git external/reasoning-on-graphs

# SubgraphRAG
git clone https://github.com/SubgraphRAG/SubgraphRAG.git     external/SubgraphRAG

# BLINK / ELQ entity linker
git clone https://github.com/facebookresearch/BLINK.git      external/BLINK

# EXPLAIGNN (for E3 experiments)
git clone https://github.com/explaignn/EXPLAIGNN.git         external/EXPLAIGNN
```

Each external repo has its own dependencies. Set up separate conda environments for models that require them:

```bash
# GNN-RAG environment
cd external/GNN-RAG && pip install -r requirements.txt && cd ../..

# EXPLAIGNN environment (requires its own env)
conda create -n explaignn python=3.9
conda activate explaignn
cd external/EXPLAIGNN && pip install -r requirements.txt && cd ../..
conda activate cascadeattack   # switch back
```

Tell the experiment scripts which Python interpreter to use for each env:

```bash
# Add to your shell profile or set before running scripts:
export PYTHON_GNN="$(conda run -n gnn_rag which python)"
export PYTHON_EXP="$(conda run -n explaignn which python)"
export EPR_PYTHON="$(conda run -n EPR_KGQA which python)"
```

---

### 4. Download datasets

**CWQ (ComplexWebQuestions):**
```bash
# Via UniKGQA (after cloning in step 3):
cp -r external/UniKGQA/data/CWQ/CWQ_ori/ data/CWQ/
# Or download directly from the official source and place at data/CWQ/
```

**WebQSP:**
```bash
cp -r external/UniKGQA/data/webqsp/webqsp_ori/data/ data/WebQSP/
# Expected: data/WebQSP/WebQSP.test.json  (1,639 questions)
```

See [data/README.md](data/README.md) for full details including Freebase/Virtuoso setup.

---

### 5. Download perturbation data and model checkpoints

All supplementary data and checkpoints are in one Google Drive folder:

> **[Google Drive — data and checkpoints](https://drive.google.com/drive/folders/1oe_iDAhKNzr6suaK7Pa7Da9Z_X8NXgHZ?usp=drive_link)**

| File / folder | Size | Place at |
|---|---|---|
| `perturbation_questions/` — perturbation JSONs (p1–p7) | ~80 MB | `data/perturbations/` |
| `perturbation_el/` — entity linking results + embeddings | ~213 MB | `data/perturbations/` |
| `subgraphs_p4/` — GraftNet P4 subgraphs | ~415 MB | `data/perturbations/subgraphs_p4/` |
| `subgraphs_p7/` — GraftNet P7 subgraphs | ~2.3 GB | `data/perturbations/subgraphs_p7/` |
| `gnn_input.zip` — pre-built GNN inputs | ~20 GB | `data/perturbations/gnn_input/` |
| `subgraph_var.zip` — subgraph variant ablation | ~8.4 GB | `data/perturbations/subgraph_var/` |
| `gnn_rag_checkpoints/` — GNN-RAG model checkpoints | ~271 MB | `external/GNN-RAG/data/pretrain/` |

GMT-KBQA data and models are hosted by the original authors — download from their [official Google Drive](https://drive.google.com/drive/folders/1QT4EG5wxtcLc8_XT5dTZ2WCi1jtByAyf) and extract into `external/GMT-KBQA-data/`.

After downloading, the layout should be:

```
data/perturbations/          ← canonical p1–p7 JSON files
data/perturbations_raw/      ← raw temperature-variant files
external/GNN-RAG/data/pretrain/   ← GNN-RAG checkpoints (.ckpt)
external/GMT-KBQA/exps/      ← GMT-KBQA trained models
models/Llama-3.1-8B-Instruct/    ← for perturbation generation
models/gpt2-large/               ← for validity check (E5)
```

LLM weights (Llama, Mistral) can be downloaded directly from HuggingFace:

```bash
pip install huggingface_hub
# Llama requires HuggingFace access approval at meta-llama/Llama-3.1-8B-Instruct
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/Llama-3.1-8B-Instruct
huggingface-cli download gpt2-large --local-dir models/gpt2-large
```

---

### 6. Set up Freebase / Virtuoso (full pipeline only)

The full E1/E2 pipeline needs a Freebase SPARQL endpoint. See [data/README.md](data/README.md#freebase-virtuoso-setup). Fast mode skips this.

Quick sanity check:
```bash
curl "http://localhost:3001/sparql?query=SELECT+%3Fn+WHERE+{<http://rdf.freebase.com/ns/m.0d05l6>+<http://rdf.freebase.com/ns/type.object.name>+%3Fn}&format=json"
# Expected: JSON response containing "London"
```

---

## Running Experiments

All shell scripts auto-detect paths via `scripts/env.sh`. Run from the repo root.

### Step 0: Verify GNN norms (no data needed)

Confirms GNN-RAG checkpoint loads correctly:
```bash
bash scripts/run_e35.sh
# → results/e35_gnn_norms_cwq.json
```

### Step 1: Validate perturbation quality (E5)

Check perturbations are semantically valid (PPL < 50, BERTScore-F1 > 0.85):
```bash
bash scripts/run_e5.sh
# → results/e5_validity_all_variants.json
```

### Step 2: Stage-wise attack — CWQ (E1)

```bash
# Fast mode: reuses original subgraphs, no Virtuoso needed (~6–8 h on 4× A100)
bash scripts/run_e1_cwq.sh fast

# Full mode: re-runs ELQ on perturbed questions, requires Virtuoso
bash scripts/run_e1_cwq.sh full
```

### Step 3: Stage-wise attack — WebQSP (E2)

```bash
bash scripts/run_e2_webqsp.sh fast
```

### Step 4: GMT-KBQA pipeline (E1/E2)

```bash
bash scripts/run_e1_e2_gmt_kbqa.sh
```

### Step 5: Stage isolation (E3)

```bash
python experiments/e3_gnn_worker.py   --pert-type p1 --dataset cwq
python experiments/e3_llm_reasoning.py --pert-type p2 --dataset cwq
```

### Step 6: Ablation — subgraph variants (P4/P7 GraftNet)

```bash
bash scripts/run_ablation.sh
```

### Step 7: Collect and aggregate all results

```bash
python experiments/collect_results.py --results-dir results/ --output results/summary_table.json
```

---

## Expected Results

Key numbers on CWQ test set, GNN-RAG:

| Attack | Δ_EL (SeedHit) | EM clean | EM perturbed | Δ_AG |
|---|---|---|---|---|
| CR (Compositional Restructuring) | ≤ 0.3 pp | 36.2% | 0.68% | −35.5 pp |
| RS (Relation Synonym Swap) | ≤ 0.3 pp | 36.2% | 18.3% | −17.9 pp |

Stage isolation: under CR, the GNN decoder is the primary failure point (not entity linking). Under RS, subgraph retrieval absorbs most degradation.

---

## Generating Your Own Perturbations

```bash
# CR perturbations using Llama-3.1-8B
python perturbation_scripts/prepare_data_llama.py \
    --dataset cwq \
    --strategy compositional_restructuring \
    --model models/Llama-3.1-8B-Instruct \
    --output data/perturbations/CWQ/

# RS perturbations
python perturbation_scripts/prepare_data_llama.py \
    --dataset cwq \
    --strategy relation_synonym \
    --model models/Llama-3.1-8B-Instruct \
    --output data/perturbations/CWQ/
```

---

## Execution Order Summary

```
1. [No data needed]
   bash scripts/run_e35.sh              # GNN weight norms

2. [After downloading perturbation data]
   bash scripts/run_e5.sh               # Validity check (~1 h, GPU 0)
   bash scripts/run_e1_cwq.sh fast      # E1 fast (~6–8 h, 4× A100)
   bash scripts/run_e2_webqsp.sh fast

3. [After Virtuoso + Freebase]
   bash scripts/run_e1_cwq.sh full      # Full cascade
   bash scripts/run_e1_e2_gmt_kbqa.sh

4. [Final aggregation]
   python experiments/collect_results.py
```

---

## Citation

> Citation to be added upon acceptance.

---

## License

MIT — see [LICENSE](LICENSE).

The external KGQA systems (GNN-RAG, GMT-KBQA, UniKGQA, RoG) are subject to their own licenses. Please consult each repository.

---

## Acknowledgements

This work builds on:
- [GNN-RAG](https://github.com/cmavro/GNN-RAG) (Mavromatis & Karypis, 2024)
- [GMT-KBQA](https://github.com/fc-liu/GMT-KBQA) (Liu et al., 2023)
- [UniKGQA](https://github.com/RUCKBReasoning/UniKGQA) (Jiang et al., 2023)
- [BLINK/ELQ](https://github.com/facebookresearch/BLINK) (Wu et al., 2020)
- [Reasoning on Graphs](https://github.com/rmanluo/reasoning-on-graphs) (Luo et al., 2024)
