#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
Plan A: Build dense subgraphs for perturbed test sets by reusing original GT-seed subgraphs.

For each of 14 perturbed test sets (7 strategies × CWQ + WebQSP):
- Copies the subgraph from the original GNN test.json using ID lookup
- Replaces only the question text with the perturbed version
- Builds auxiliary files: vocab.txt, word_emb.npy, test.dep (if spaCy available)

Output: Pre-process/plan_a/{dataset}/{strategy}/
  - test_subgraph.json  (GNN format with perturbed question)
  - vocab.txt
  - word_emb.npy
  - test.dep            (requires spaCy)

Usage:
    python build_subgraphs_plan_a.py [--dataset cwq|webqsp|both] [--strategy all|s1_active_passive|...]

    # Run all
    python build_subgraphs_plan_a.py

    # Run only CWQ
    python build_subgraphs_plan_a.py --dataset cwq

    # Run single strategy
    python build_subgraphs_plan_a.py --dataset webqsp --strategy s1_active_passive
"""

import argparse
import copy
import glob
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = str(Path(__file__).resolve().parents[1])
GNN_DIR = os.path.join(REPO_ROOT, "GNN")
RESULTS_DIR = os.path.join(GNN_DIR, "results")
OUTPUT_BASE = os.path.join(REPO_ROOT, "Pre-process", "plan_a")
GLOVE_PATH = os.path.join(REPO_ROOT, "freebase/GraftNet/preprocessing/downloads/glove.840B.300d.txt")

DATASETS = {
    "cwq":    {"orig_json": os.path.join(GNN_DIR, "CWQ/test.json"),
               "entities_txt": os.path.join(GNN_DIR, "CWQ/entities.txt"),
               "relations_txt": os.path.join(GNN_DIR, "CWQ/relations.txt")},
    "webqsp": {"orig_json": os.path.join(GNN_DIR, "webqsp/test.json"),
               "entities_txt": os.path.join(GNN_DIR, "webqsp/entities.txt"),
               "relations_txt": os.path.join(GNN_DIR, "webqsp/relations.txt")},
}

STRATEGIES = [
    "s1_active_passive",
    "s2_reorder",
    "s3_relation_synonym",
    "s4_entity_alias",
    "s5_adversarial_para",
    "s6_grammatical_perturb",
    "s7_filler_injection",
]

GLOVE_DIM = 300


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_orig_by_id(orig_json_path):
    """Load original test.json into a dict keyed by id."""
    print(f"  Loading original data from {orig_json_path} ...", flush=True)
    orig = {}
    with open(orig_json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            orig[entry["id"]] = entry
    print(f"  Loaded {len(orig)} original entries.", flush=True)
    return orig


def load_relations(relations_txt):
    """Load relations.txt → list of relation strings."""
    with open(relations_txt) as f:
        return [l.strip() for l in f if l.strip()]


# ─── Vocab ────────────────────────────────────────────────────────────────────

def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def build_vocab(questions, relation_words):
    """Build ordered vocabulary from questions and relation tokens."""
    vocab = OrderedDict()
    for q in questions:
        for tok in tokenize(q):
            if tok and tok not in vocab:
                vocab[tok] = len(vocab)
    for word in relation_words:
        if word and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def relation_words(relations_list):
    """Split relation names into tokens (e.g. 'location.country.capital' → words)."""
    words = []
    for rel in relations_list:
        for part in re.split(r"[._]", rel):
            part = part.strip().lower()
            if part:
                words.append(part)
    return words


def write_vocab(vocab, path):
    with open(path, "w") as f:
        for word in vocab:
            f.write(word + "\n")
    print(f"    vocab.txt: {len(vocab)} words → {path}", flush=True)


# ─── Embeddings ───────────────────────────────────────────────────────────────

def load_glove(glove_path, dim=300):
    """Load full GloVe into memory once. Returns dict word→np.array."""
    if not os.path.exists(glove_path):
        print(f"  WARNING: GloVe not found at {glove_path}. Embeddings will be zero vectors.", flush=True)
        return {}
    print(f"  Loading GloVe from {glove_path} (one-time) ...", flush=True)
    word2emb = {}
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < dim + 1:
                continue
            try:
                emb = np.array([float(x) for x in parts[-dim:]], dtype=np.float32)
                if len(emb) == dim:
                    word2emb[parts[0]] = emb
            except ValueError:
                continue
    print(f"  GloVe loaded: {len(word2emb)} vectors.", flush=True)
    return word2emb


def build_embeddings(vocab, word2emb, dim=300):
    """Build word embedding matrix from pre-loaded GloVe dict."""
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    found = 0
    for word, idx in vocab.items():
        if word in word2emb:
            matrix[idx] = word2emb[word]
            found += 1
    print(f"    Embeddings: {found}/{len(vocab)} words found in GloVe.", flush=True)
    return matrix


# ─── Dep parses ───────────────────────────────────────────────────────────────

def build_dep_parses(entries, out_path):
    """Generate spaCy dependency parses. Skipped if spaCy not available."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        print("    WARNING: spaCy/en_core_web_sm not available. Skipping test.dep generation.", flush=True)
        print("    To generate: pip install spacy && python -m spacy download en_core_web_sm", flush=True)
        return

    print(f"    Generating dependency parses for {len(entries)} entries ...", flush=True)
    with open(out_path, "w") as f:
        for entry in entries:
            doc = nlp(entry["question"])
            dep = []
            for token in doc:
                # 1-indexed head, 0 for REPO_ROOT (matches step5_build_dependencies.py convention)
                head_idx = 0 if token.head == token else token.head.i + 1
                dep.append([token.text.lower(), token.tag_, str(head_idx), token.dep_])
            out = {"id": entry["id"], "question": entry["question"], "dep": dep}
            f.write(json.dumps(out) + "\n")
    print(f"    test.dep written → {out_path}", flush=True)


# ─── Core: build one strategy ─────────────────────────────────────────────────

def process_strategy(dataset, strategy, orig_by_id, relations_list, word2emb, glove_dim=300):
    """Build test_subgraph.json + auxiliary files for one (dataset, strategy) pair."""
    # Input paths
    elq_path  = os.path.join(RESULTS_DIR, "output_req", dataset, strategy, "test_elq.jsonl")
    test_path = os.path.join(RESULTS_DIR, "output_req", dataset, strategy, "test.jsonl")

    if not os.path.exists(elq_path):
        print(f"  SKIP: {elq_path} not found.", flush=True)
        return

    # Output dir
    out_dir = os.path.join(OUTPUT_BASE, dataset, strategy)
    os.makedirs(out_dir, exist_ok=True)

    # Load perturbed entries (prefer test_elq.jsonl — it has all fields)
    perturbed = load_jsonl(elq_path)
    print(f"  {dataset}/{strategy}: {len(perturbed)} entries", flush=True)

    # Build merged entries
    merged = []
    missing = 0
    for entry in perturbed:
        qid = entry["id"]
        if qid not in orig_by_id:
            missing += 1
            continue
        orig = orig_by_id[qid]
        out = {
            "id": qid,
            "question": entry["question"],
            "question_orig": entry.get("question_orig", orig["question"]),
            "answers": orig["answers"],          # use original answers (test.jsonl has empty [])
            "entities": orig["entities"],
            "subgraph": copy.deepcopy(orig["subgraph"]),
        }
        # Carry over elq fields for reference
        out["elq_seed_mid"] = entry.get("elq_seed_mid", "")
        out["gt_seed_mids"] = entry.get("gt_seed_mids", [])
        out["elq_gt_mid_match"] = entry.get("elq_gt_mid_match", None)
        out["strategy"] = entry.get("strategy", strategy)
        merged.append(out)

    if missing:
        print(f"    WARNING: {missing} entries not found in original test.json (ID mismatch).", flush=True)

    # Write test_subgraph.json
    subgraph_path = os.path.join(out_dir, "test_subgraph.json")
    with open(subgraph_path, "w") as f:
        for entry in merged:
            f.write(json.dumps(entry) + "\n")
    print(f"    test_subgraph.json: {len(merged)} entries → {subgraph_path}", flush=True)

    # Collect questions for vocab/dep
    questions = [e["question"] for e in merged]

    # Build vocab
    rel_words = relation_words(relations_list)
    vocab = build_vocab(questions, rel_words)
    vocab_path = os.path.join(out_dir, "vocab.txt")
    write_vocab(vocab, vocab_path)

    # Build embeddings
    emb_matrix = build_embeddings(vocab, word2emb, glove_dim)
    emb_path = os.path.join(out_dir, "word_emb.npy")
    np.save(emb_path, emb_matrix)
    print(f"    word_emb.npy: shape {emb_matrix.shape} → {emb_path}", flush=True)

    # Build dep parses
    dep_path = os.path.join(out_dir, "test.dep")
    build_dep_parses(merged, dep_path)

    return len(merged)


# ─── Symlink entities/relations (shared per dataset) ──────────────────────────

def setup_dataset_symlinks(dataset):
    """Symlink entities.txt and relations.txt from original GNN data dir."""
    dataset_out = os.path.join(OUTPUT_BASE, dataset)
    os.makedirs(dataset_out, exist_ok=True)
    info = DATASETS[dataset]
    for fname in ["entities.txt", "relations.txt"]:
        src = info[fname.replace(".txt", "_txt")]
        dst = os.path.join(dataset_out, fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            print(f"  Symlinked {dst} → {src}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plan A: Build perturbed subgraphs via GT-seed reuse")
    parser.add_argument("--dataset", choices=["cwq", "webqsp", "both"], default="both")
    parser.add_argument("--strategy", default="all",
                        help="Strategy name or 'all'. E.g. s1_active_passive")
    args = parser.parse_args()

    datasets = ["cwq", "webqsp"] if args.dataset == "both" else [args.dataset]
    strategies = STRATEGIES if args.strategy == "all" else [args.strategy]

    # Load GloVe once — shared across all datasets/strategies
    word2emb = load_glove(GLOVE_PATH, GLOVE_DIM)

    for dataset in datasets:
        print(f"\n{'='*60}", flush=True)
        print(f"Dataset: {dataset.upper()}", flush=True)
        print(f"{'='*60}", flush=True)

        info = DATASETS[dataset]
        orig_by_id = load_orig_by_id(info["orig_json"])
        relations_list = load_relations(info["relations_txt"])
        setup_dataset_symlinks(dataset)

        total = 0
        for strategy in strategies:
            print(f"\n  Strategy: {strategy}", flush=True)
            n = process_strategy(dataset, strategy, orig_by_id, relations_list, word2emb, GLOVE_DIM)
            if n:
                total += n

        print(f"\n  {dataset.upper()} done. Total entries written: {total}", flush=True)

    print(f"\nAll done. Outputs in: {OUTPUT_BASE}", flush=True)


if __name__ == "__main__":
    main()
