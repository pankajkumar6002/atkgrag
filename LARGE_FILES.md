# Large Files — Upload Guide

Files are split into two categories: **you upload** vs **already hosted by original authors**.
GitHub hard limit: 100 MB per file, 2 GB per repository.

---

## What You Need to Upload Yourself

| Your local source | Drive folder | Size | Status | Repo destination |
|---|---|---|---|---|
| `tmlr/pert/dataset/cwq_*.json`, `webqsp_*.json` | `perturbation_questions/` | ~80 MB | ✅ Done | `data/perturbations/` |
| `tmlr/pert/dataset/el_elq/`, `el_agent/`, `embeddings/` | `perturbation_el/` | ~213 MB | ✅ Done | `data/perturbations/` |
| `tmlr/pert/dataset/subgraphs_p4/` | `subgraphs_p4/` | ~415 MB | ✅ Done | `data/perturbations/subgraphs_p4/` |
| `tmlr/pert/dataset/subgraphs_p7/` | `subgraphs_p7/` | ~2.3 GB | ✅ Done | `data/perturbations/subgraphs_p7/` |
| `tmlr/Repo/GNN-RAG/data/pretrain/*.ckpt` | `gnn_rag_checkpoints/` | ~271 MB | ✅ Done | `external/GNN-RAG/data/pretrain/` |
| `tmlr/pert/dataset/gnn_input/` | `gnn_input.zip` | ~20 GB | ✅ Done | `data/perturbations/gnn_input/` |
| `tmlr/pert/dataset/subgraph_var/` | `subgraph_var.zip` | ~8.4 GB | ✅ Done | `data/perturbations/subgraph_var/` |
| `tmlr/pert/dataset/gnn-rag/` | — | 12 GB | ❌ Skip | Reviewers regenerate |
| `perturbation/CWQ/`, `perturbation/WebQSP/` | — | 260 MB | ❌ Skip | Adds no value over p1–p7 |

---

## Already Hosted — Just Point Users There

Do not re-upload. Reference their links directly.

| Resource | Link | What's included |
|---|---|---|
| GMT-KBQA data + checkpoints | [Google Drive (GMT-KBQA official)](https://drive.google.com/drive/folders/1QT4EG5wxtcLc8_XT5dTZ2WCi1jtByAyf) | Trained models, entity retrieval, FACC1, preprocessed CWQ/WebQSP |
| CWQ (ComplexWebQuestions) | [Dropbox](https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AACuu4v3YNkhirzBOeeaHYala) | `ComplexWebQuestions_test/train/dev.json` |
| WebQSP | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering-2/) | `WebQSP.test/train.json` |
| Freebase RDF | [Google Developers (archived)](https://developers.google.com/freebase) | `freebase-rdf-latest.nt.gz` (~24 GB) |
| Wikidata dump | [Wikimedia](https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz) | `latest-all.json.gz` (~80 GB) — only to rebuild `alias_to_mid.pkl` |
| Llama-3.1-8B-Instruct | [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Requires HF access approval |
| Mistral-7B-Instruct-v0.3 | [HuggingFace](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Public |
| GPT-2-large | [HuggingFace](https://huggingface.co/gpt2-large) | Public — validity check (E5) |

---

## Google Drive Structure

Root folder: `https://drive.google.com/drive/folders/1oe_iDAhKNzr6suaK7Pa7Da9Z_X8NXgHZ?usp=drive_link`

```
(Drive root)/
├── perturbation_questions/          ← individual JSON files
│   ├── cwq_orig.json
│   ├── cwq_p1.json … cwq_p7_enriched.json
│   ├── webqsp_orig.json
│   └── webqsp_p1.json … webqsp_p7_enriched.json
├── perturbation_el/                 ← entity linking outputs
│   ├── el_elq/       (~48 MB)
│   ├── el_agent/     (~90 MB)
│   └── embeddings/   (~75 MB)
├── subgraphs_p4/                    ← GraftNet P4 subgraphs (~415 MB)
├── subgraphs_p7/                    ← GraftNet P7 subgraphs (~2.3 GB)
├── gnn_rag_checkpoints/             ← GNN-RAG model checkpoints
│   ├── ReaRev_CWQ.ckpt      (~91 MB)
│   ├── ReaRev_webqsp.ckpt   (~90 MB)
│   └── ReaRe_webqsp_v2.ckpt (~90 MB)
├── gnn_input.zip                    ← pre-built GNN inputs (~20 GB)
└── subgraph_var.zip                 ← subgraph variant ablation (~8.4 GB)
```

**Skipped files** (not uploaded, not needed by reviewers):
- `cwq_p1_fixed3.json`, `cwq_p1_git_backup.json`
- `webqsp_p1_fixed3.json`, `webqsp_p1_verified_fixed.json`
- `denotation_report.json`

---

## How Reviewers Extract Files

After downloading from Drive, place files as follows:

```bash
# Perturbation questions
# Copy individual JSONs into data/perturbations/
mkdir -p data/perturbations
cp perturbation_questions/*.json data/perturbations/

# Entity linking results
unzip perturbation_el.zip -d perturbation_el/
mv perturbation_el/el_elq     data/perturbations/el_elq
mv perturbation_el/el_agent   data/perturbations/el_agent
mv perturbation_el/embeddings data/perturbations/embeddings

# Subgraphs
cp -r subgraphs_p4/ data/perturbations/subgraphs_p4/
cp -r subgraphs_p7/ data/perturbations/subgraphs_p7/

# GNN-RAG checkpoints
mkdir -p external/GNN-RAG/data/pretrain
cp gnn_rag_checkpoints/*.ckpt external/GNN-RAG/data/pretrain/

# GNN inputs (~20 GB)
unzip -d data/perturbations/gnn_input/ gnn_input.zip

# Subgraph variants (~8.4 GB)
unzip -d data/perturbations/subgraph_var/ subgraph_var.zip
```

---

## Checklist

**Google Drive — done:**
- [x] `perturbation_questions/` — 18 JSON files (cwq_orig, cwq_p1–p7_enriched, webqsp_orig, webqsp_p1–p7_enriched)
- [x] `perturbation_el/` — el_elq/ + el_agent/ + embeddings/ (~213 MB)
- [x] `subgraphs_p4/` (~415 MB)
- [x] `subgraphs_p7/` (~2.3 GB)
- [x] `gnn_rag_checkpoints/` — ReaRev_CWQ.ckpt + ReaRev_webqsp.ckpt + ReaRe_webqsp_v2.ckpt (~271 MB)
- [x] `gnn_input.zip` (~20 GB)
- [x] `subgraph_var.zip` (~8.4 GB)

**README.md — still to do:**
- [ ] Verify Drive folder is **"Anyone with the link — Viewer"** (not restricted to your account)

**External links — verify these resolve:**
- [ ] GMT-KBQA Drive: https://drive.google.com/drive/folders/1QT4EG5wxtcLc8_XT5dTZ2WCi1jtByAyf
- [ ] CWQ Dropbox link still active
- [ ] HuggingFace Llama access approved at `meta-llama/Llama-3.1-8B-Instruct`
