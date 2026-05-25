#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
e5_validity.py — Experiment E5: Attack Validity Check

Computes PPL (GPT-2-large) and BERTScore-F1 (DeBERTa-xlarge-mnli) for every
perturbed question across all 5 perturbation types, for both greedy and
temperature-sampled variants.

Perturbation data layout (already present):
  data/perturbations_raw/CWQ/{greedy,temp_*}/
      ComplexWebQuestions_test_pert_{type}.json
  data/perturbations_raw/WebQSP/{greedy,temp_*}/
      WebQSP.test_pert_{type}.json

Each CWQ record: {ID, question (perturbed), original_question,
                  perturbation_strategy, perturbation_changed, ...}
Each WebQSP record (inside "Questions" list): {QuestionId, ProcessedQuestion (perturbed),
                                                original_question, perturbation_strategy,
                                                perturbation_changed, ...}

Usage:
  CUDA_VISIBLE_DEVICES=0 python e5_validity.py \
    --pert-root data/perturbations_raw \
    --gpt2-path models/gpt2-large \
    --out results/e5_validity.json \
    --variant greedy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from bert_score import score as bertscore_fn
from transformers import AutoModelForCausalLM, AutoTokenizer


PERT_TYPES = [
    "active_passive_flip",
    "adversarial_entity",
    "entity_alias_substitution",
    "question_reordering",
    "relation_synonym",
]

# Map to paper condition labels
PERT_TO_CONDITION = {
    "entity_alias_substitution": "Lex",
    "relation_synonym": "Lex",
    "active_passive_flip": "Str",
    "question_reordering": "Str",
    "adversarial_entity": "Cmp",
}

PPL_THRESHOLD = 50.0
BERTSCORE_THRESHOLD = 0.85
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cwq_pairs(path: Path) -> List[Tuple[str, str, bool]]:
    """Return [(original_q, perturbed_q, changed), ...]"""
    with path.open() as f:
        records = json.load(f)
    pairs = []
    for r in records:
        orig = str(r.get("original_question") or "").strip()
        pert = str(r.get("question") or "").strip()
        changed = bool(r.get("perturbation_changed", True))
        if orig and pert:
            pairs.append((orig, pert, changed))
    return pairs


def load_webqsp_pairs(path: Path) -> List[Tuple[str, str, bool]]:
    with path.open() as f:
        data = json.load(f)
    questions = data["Questions"] if isinstance(data, dict) else data
    pairs = []
    for r in questions:
        orig = str(r.get("original_question") or "").strip()
        pert = str(r.get("ProcessedQuestion") or r.get("RawQuestion") or "").strip()
        changed = bool(r.get("perturbation_changed", True))
        if orig and pert:
            pairs.append((orig, pert, changed))
    return pairs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def load_gpt2(model_path: str, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    return tok, model


def compute_ppl_single(tok, model, text: str, max_length: int = 128) -> float:
    enc = tok(text, return_tensors="pt", truncation=True,
               max_length=max_length).to(model.device)
    with torch.inference_mode():
        out = model(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss).item())


def compute_bertscore_batch(
    originals: List[str], perturbeds: List[str], device: str, batch_size: int = 64
) -> List[float]:
    all_f1: List[float] = []
    for i in range(0, len(originals), batch_size):
        _, _, F1 = bertscore_fn(
            perturbeds[i: i + batch_size],
            originals[i: i + batch_size],
            model_type=BERTSCORE_MODEL,
            lang="en",
            rescale_with_baseline=False,
            device=device,
            verbose=False,
        )
        all_f1.extend(float(f) for f in F1)
    return all_f1


def evaluate_pairs(
    pairs: List[Tuple[str, str, bool]],
    gpt2_tok, gpt2_model,
    device: str,
) -> Dict[str, Any]:
    if not pairs:
        return {"skipped": True, "reason": "no pairs"}

    # Only evaluate changed perturbations for pass-rate stats
    changed = [(o, p) for o, p, c in pairs if c]
    all_pairs = pairs

    originals = [o for o, p, _ in all_pairs]
    perturbeds = [p for o, p, _ in all_pairs]
    n = len(all_pairs)
    n_changed = len(changed)

    print(f"    PPL for {n} questions ({n_changed} changed)...")
    ppls = [compute_ppl_single(gpt2_tok, gpt2_model, p) for p in perturbeds]

    print(f"    BERTScore for {n} questions...")
    bscores = compute_bertscore_batch(originals, perturbeds, device)

    ppl_pass = [ppl < PPL_THRESHOLD for ppl in ppls]
    bs_pass = [bs > BERTSCORE_THRESHOLD for bs in bscores]
    both_pass = [a and b for a, b in zip(ppl_pass, bs_pass)]

    return {
        "n_total": n,
        "n_changed": n_changed,
        "ppl_pass_rate": sum(ppl_pass) / n,
        "bs_pass_rate": sum(bs_pass) / n,
        "both_pass_rate": sum(both_pass) / n,
        "ppl_mean": sum(ppls) / n,
        "ppl_median": float(sorted(ppls)[n // 2]),
        "bs_mean": sum(bscores) / n,
        "bs_median": float(sorted(bscores)[n // 2]),
        "ppl_threshold": PPL_THRESHOLD,
        "bs_threshold": BERTSCORE_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E5: attack validity check")
    p.add_argument("--pert-root", type=Path,
                   default=REPO_ROOT / "data" / "perturbations_raw")
    p.add_argument("--gpt2-path", type=str,
                   default=str(REPO_ROOT / "models" / "gpt2-large"))
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "results" / "e5_validity.json")
    p.add_argument("--variant", default="greedy",
                   help="Subdirectory: greedy | temp_0.10 | ... (default: greedy)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f"Loading GPT-2-large from {args.gpt2_path}...")
    gpt2_tok, gpt2_model = load_gpt2(args.gpt2_path, args.device)

    results: Dict[str, Any] = {}

    for pert_type in PERT_TYPES:
        condition = PERT_TO_CONDITION.get(pert_type, "?")
        print(f"\n[{pert_type}] (condition: {condition})")
        results[pert_type] = {"condition": condition}

        # CWQ
        cwq_file = args.pert_root / "CWQ" / args.variant / \
                   f"ComplexWebQuestions_test_pert_{pert_type}.json"
        print(f"  CWQ: {cwq_file}")
        if cwq_file.exists():
            pairs = load_cwq_pairs(cwq_file)
            results[pert_type]["cwq"] = evaluate_pairs(pairs, gpt2_tok, gpt2_model, args.device)
        else:
            results[pert_type]["cwq"] = {"skipped": True, "reason": "file not found"}

        # WebQSP
        webqsp_file = args.pert_root / "WebQSP" / args.variant / \
                      f"WebQSP.test_pert_{pert_type}.json"
        print(f"  WebQSP: {webqsp_file}")
        if webqsp_file.exists():
            pairs = load_webqsp_pairs(webqsp_file)
            results[pert_type]["webqsp"] = evaluate_pairs(pairs, gpt2_tok, gpt2_model, args.device)
        else:
            results[pert_type]["webqsp"] = {"skipped": True, "reason": "file not found"}

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Type':<30} {'Cond':<5} {'Dataset':<8} {'n':>5} "
          f"{'PPL%':>7} {'BS%':>7} {'Both%':>7}")
    print("-" * 80)
    for pert_type in PERT_TYPES:
        cond = results[pert_type]["condition"]
        for ds in ("cwq", "webqsp"):
            r = results[pert_type][ds]
            if r.get("skipped"):
                print(f"  {pert_type:<28} {cond:<5} {ds:<8} {'—':>5} "
                      f"{'—':>7} {'—':>7} {'—':>7}  skipped")
            else:
                print(f"  {pert_type:<28} {cond:<5} {ds:<8} {r['n_total']:>5} "
                      f"{r['ppl_pass_rate']:>7.3f} {r['bs_pass_rate']:>7.3f} "
                      f"{r['both_pass_rate']:>7.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
