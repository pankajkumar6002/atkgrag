# Data Setup

## Datasets

### CWQ (ComplexWebQuestions)

1. Download the official CWQ dataset from the [WebQSP/CWQ release](https://www.microsoft.com/en-us/research/publication/the-web-as-a-knowledge-base-for-answering-complex-questions/)
2. Or use the version shipped with UniKGQA:
   ```bash
   # After cloning UniKGQA (see README.md)
   cp -r external/UniKGQA/data/CWQ/CWQ_ori/ data/CWQ/
   ```
3. Expected files:
   - `data/CWQ/ComplexWebQuestions_train.json`
   - `data/CWQ/ComplexWebQuestions_dev.json`
   - `data/CWQ/ComplexWebQuestions_test.json` (3,531 questions)

### WebQSP

1. Download from [Facebook Research](https://github.com/nickmahilani/WebQSP)
2. Or use the version shipped with UniKGQA:
   ```bash
   cp -r external/UniKGQA/data/webqsp/webqsp_ori/data/ data/WebQSP/
   ```
3. Expected files:
   - `data/WebQSP/WebQSP.train.json`
   - `data/WebQSP/WebQSP.test.json` (1,639 questions)

---

## Perturbation Data

Download the pre-generated perturbation sets from the link in [LARGE_FILES.md](../LARGE_FILES.md) and extract to:

```
data/perturbations/
  CWQ/
    p1_compositional_restructuring.json   # 3,531 CR-perturbed questions
    p2_relation_synonym.json              # 3,531 RS-perturbed questions
    p3_entity_alias.json
    p4_active_passive.json
    p5_question_reordering.json
    p6_adversarial_entity.json
  WebQSP/
    p1_compositional_restructuring.json   # 1,639 CR-perturbed questions
    p2_relation_synonym.json
    ...
```

**Format (CWQ):** JSON list, each record has keys:
```json
{
  "ID": "...",
  "question": "<perturbed question>",
  "original_question": "<original question>",
  "perturbation_strategy": "compositional_restructuring",
  "perturbation_changed": true,
  "sparql": "..."
}
```

**Format (WebQSP):** `{"Questions": [...]}`, each record has:
```json
{
  "QuestionId": "...",
  "ProcessedQuestion": "<perturbed>",
  "original_question": "<original>",
  "perturbation_strategy": "...",
  "Parses": [{"Sparql": "..."}]
}
```

---

## Freebase / Virtuoso Setup

The full pipeline (E1 full mode, E2 full mode, GMT-KBQA) requires a Freebase SPARQL endpoint.

### Option A: Use an existing Virtuoso instance

If you already have Virtuoso running with Freebase loaded:
```bash
# Verify it is working (expected response contains "London")
curl "http://localhost:3001/sparql?query=SELECT+%3Fn+WHERE+{<http://rdf.freebase.com/ns/m.0d05l6>+<http://rdf.freebase.com/ns/type.object.name>+%3Fn}&format=json"
```

Update the endpoint in the experiment scripts if your port differs from `3001`.

### Option B: Set up Virtuoso from Freebase RDF dump

1. Download the Freebase RDF dump (see [LARGE_FILES.md](../LARGE_FILES.md)):
   ```
   freebase-rdf-latest.nt.gz  (~24 GB compressed)
   ```

2. Install Virtuoso Open Source:
   ```bash
   sudo apt-get install virtuoso-opensource
   # or build from source: https://github.com/openlink/virtuoso-opensource
   ```

3. Load Freebase into Virtuoso:
   ```bash
   # Start Virtuoso
   virtuoso-t +configfile virtuoso.ini +wait
   
   # Load the RDF (this takes several hours)
   isql 1111 dba dba exec="ld_dir('/path/to/', 'freebase-rdf-latest.nt', 'http://freebase.com'); rdf_loader_run();"
   ```

4. Confirm it is running and test the endpoint.

### Option C: Fast mode (no Virtuoso needed)

For the **fast mode** experiments, Virtuoso is not required. The experiments reuse pre-built subgraphs:
```bash
bash scripts/run_e1_cwq.sh data/perturbations/ fast
```

---

## GNN-RAG Subgraph Cache

Pre-built ELQ-seeded subgraphs for CWQ and WebQSP are available in [LARGE_FILES.md](../LARGE_FILES.md). Download and place at:

```
data/gnn_input/
  cwq_p1/
    test_subgraph.json
  cwq_p2/
    test_subgraph.json
  ...
  webqsp_p1/
    test_subgraph.json
  ...
```

---

## GMT-KBQA Data and Models

GMT-KBQA's authors host all preprocessed data and trained checkpoints on their official Google Drive. **You do not need to prepare this data yourself.** Download directly from:

> https://drive.google.com/drive/folders/1QT4EG5wxtcLc8_XT5dTZ2WCi1jtByAyf

Extract into `external/GMT-KBQA-data/`. The package includes:
- Trained model checkpoints (`exps/CWQ_GMT_KBQA/`, `exps/WebQSP_GMT_KBQA/`)
- Entity retrieval models and candidates
- Preprocessed CWQ and WebQSP splits (generation, relation retrieval, s-expressions)
- FACC1 common data and Freebase relation maps (`data/common_data/`)

Their README states: *"all data preparation steps can be skipped, as we've provided those results."*

---

## Freebase Entity Utilities

For entity linking (ELQ mode), the FACC1 surface forms and `fb_en.txt` are included in the GMT-KBQA Google Drive package above (`data/common_data/`). Place them at:

- `external/GMT-KBQA/data/common_data/facc1/`
- `external/GMT-KBQA/data/common_data/fb_relation_rich_map.json`
