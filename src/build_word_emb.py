#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_word_emb.py — Extend GNN-RAG vocab + rebuild word_emb.npy for perturbations.

For each dataset:
  1. Collect all unique tokens from perturbed questions in data/perturbations/{dataset}_p{1-6}.json
  2. Diff against original vocab.txt → find new tokens
  3. Fetch GloVe vectors for new tokens (from word_emb.npy index or GloVe file)
  4. Concatenate: original word_emb.npy rows + new token rows → extended word_emb_pert.npy
  5. Write extended vocab.txt alongside

GloVe fallback order:
  a. Look up in original word_emb.npy (already indexed by vocab.txt)
  b. Load GloVe from Pre-process/plan_b embeddings if available
  c. Random init with UNK magnitude for truly unseen tokens (logged)

Output:
  data/perturbations/embeddings/{dataset}_pert_vocab.txt       (extended vocab)
  data/perturbations/embeddings/{dataset}_pert_word_emb.npy    (extended embeddings)

Usage:
  python build_word_emb.py --dataset cwq
  python build_word_emb.py --dataset webqsp
  python build_word_emb.py --all
  python build_word_emb.py --dataset cwq --verify  # print stats only
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import json
import numpy as np

DATASET_DIR = REPO_ROOT / "data" / "perturbations"
GNN_DATA = REPO_ROOT / "external" / "GNN-RAG" / "data"
PLAN_B = REPO_ROOT / "data" / "plan_b"
OUT_DIR = DATASET_DIR / "embeddings"

PERT_TYPES = ["p1", "p2", "p3", "p4", "p5", "p6"]

# Map dataset name to GNN-RAG data subdir name
DATASET_SUBDIR = {"cwq": "CWQ", "webqsp": "webqsp"}


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer matching GNN-RAG preprocessing."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def load_vocab(vocab_path: Path) -> Tuple[List[str], Dict[str, int]]:
    with open(vocab_path) as f:
        vocab = [line.strip() for line in f if line.strip()]
    token2idx = {t: i for i, t in enumerate(vocab)}
    return vocab, token2idx


def collect_new_tokens(
    dataset: str,
    token2idx: Dict[str, int],
) -> Set[str]:
    new_tokens: Set[str] = set()
    for p_type in PERT_TYPES:
        path = DATASET_DIR / f"{dataset}_{p_type}.json"
        if not path.exists():
            continue
        with open(path) as f:
            records = json.load(f)
        for rec in records:
            for tok in tokenize(rec.get("perturbed_question", "")):
                if tok not in token2idx:
                    new_tokens.add(tok)
    return new_tokens


def get_embedding_dim(word_emb: np.ndarray) -> int:
    return word_emb.shape[1]


def build_new_vectors(
    new_tokens: List[str],
    orig_emb: np.ndarray,
    plan_b_emb: np.ndarray,
    plan_b_vocab: Dict[str, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Build embedding rows for new_tokens."""
    dim = get_embedding_dim(orig_emb)
    # Estimate UNK magnitude from mean of original embeddings
    unk_magnitude = float(np.linalg.norm(orig_emb, axis=1).mean())

    rows = []
    unk_tokens: List[str] = []

    for tok in new_tokens:
        if tok in plan_b_vocab:
            rows.append(plan_b_emb[plan_b_vocab[tok]])
        else:
            # Random unit vector scaled to UNK magnitude
            vec = rng.standard_normal(dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8) * unk_magnitude
            rows.append(vec)
            unk_tokens.append(tok)

    if unk_tokens:
        print(f"    UNK (random init) tokens: {len(unk_tokens)}")
        if len(unk_tokens) <= 20:
            print(f"    {unk_tokens}")

    return np.stack(rows, axis=0).astype(np.float32) if rows else np.zeros((0, dim), np.float32)


def load_plan_b_emb(dataset: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """Try to load a plan_b vocab+embedding as a GloVe fallback."""
    subdir = DATASET_SUBDIR.get(dataset, dataset)
    # Try any strategy's embedding — they all use same GloVe base
    for strategy in ["s1_active_passive", "s4_entity_alias", "s2_reorder"]:
        emb_path = PLAN_B / dataset / strategy / "word_emb.npy"
        vocab_path = PLAN_B / dataset / strategy / "vocab.txt"
        if emb_path.exists() and vocab_path.exists():
            emb = np.load(str(emb_path))
            _, vocab = load_vocab(vocab_path)
            print(f"    Loaded plan_b fallback from {emb_path.parent}")
            return emb, vocab
    return np.zeros((0, 100), np.float32), {}


def build_dataset(dataset: str, verify: bool) -> None:
    subdir = DATASET_SUBDIR[dataset]
    vocab_path = GNN_DATA / subdir / "vocab.txt"
    emb_path = GNN_DATA / subdir / "word_emb.npy"

    if not vocab_path.exists() or not emb_path.exists():
        print(f"  [SKIP] Original vocab/emb not found for {dataset}")
        return

    vocab, token2idx = load_vocab(vocab_path)
    orig_emb = np.load(str(emb_path))
    dim = get_embedding_dim(orig_emb)
    print(f"  Original vocab: {len(vocab)} tokens, emb shape: {orig_emb.shape}")

    new_tokens_set = collect_new_tokens(dataset, token2idx)
    new_tokens = sorted(new_tokens_set)
    print(f"  New tokens from perturbations: {len(new_tokens)}")

    if verify:
        print(f"  Sample new tokens: {new_tokens[:20]}")
        return

    plan_b_emb, plan_b_vocab = load_plan_b_emb(dataset)
    rng = np.random.default_rng(42)

    new_rows = build_new_vectors(new_tokens, orig_emb, plan_b_emb, plan_b_vocab, rng)

    extended_vocab = vocab + new_tokens
    extended_emb = np.concatenate([orig_emb, new_rows], axis=0) if new_rows.shape[0] > 0 else orig_emb

    print(f"  Extended vocab: {len(extended_vocab)} tokens, emb shape: {extended_emb.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_vocab = OUT_DIR / f"{dataset}_pert_vocab.txt"
    out_emb = OUT_DIR / f"{dataset}_pert_word_emb.npy"

    with open(out_vocab, "w") as f:
        f.write("\n".join(extended_vocab) + "\n")
    np.save(str(out_emb), extended_emb)

    print(f"  Saved vocab → {out_vocab}")
    print(f"  Saved emb   → {out_emb}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend GNN-RAG vocab + word_emb for perturbations")
    parser.add_argument("--dataset", choices=["cwq", "webqsp"], default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Print stats only, no writes")
    args = parser.parse_args()

    if args.all:
        datasets = ["cwq", "webqsp"]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("Specify --dataset or --all")

    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        build_dataset(dataset, verify=args.verify)


if __name__ == "__main__":
    main()