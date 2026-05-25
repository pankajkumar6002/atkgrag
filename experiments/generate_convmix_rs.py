#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
generate_convmix_rs.py — RS (Relation Synonym / p7) perturbation for ConvMix.

Uses Llama-3.3-70B-Instruct (device_map="auto", all GPUs) to substitute
relations/predicates in KB-answerable ConvMix test turns with synonyms,
keeping entities and answers unchanged.

Quality is gated by PPL (Llama-3.1-8B), BERTScore (roberta-large), and Jaccard
checks — same pipeline as perturb_relation.py.

Outputs:
  _benchmarks/convmix/test_set/test_set_ALL_rs.json   — full benchmark with RS turns
  tmlr/results/convmix_rs/perturbation_meta.json       — per-turn stats
  tmlr/results/convmix_rs/rs_cache.json                — resume cache (keyed by question_id)

Usage (fb-perturb env):
  python  # set PYTHON_FB env var to override generate_convmix_rs.py
  python  # set PYTHON_FB env var to override generate_convmix_rs.py --dry-run
  python  # set PYTHON_FB env var to override generate_convmix_rs.py --resume
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from bert_score import BERTScorer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXPLAIGNN_ROOT = REPO_ROOT / "external" / "EXPLAIGNN"
TEST_SET_IN    = EXPLAIGNN_ROOT / "_benchmarks/convmix/test_set/test_set_ALL.json"
TEST_SET_OUT   = EXPLAIGNN_ROOT / "_benchmarks/convmix/test_set/test_set_ALL_rs.json"
META_OUT       = REPO_ROOT / "results" / "convmix_rs" / "perturbation_meta.json"
CACHE_FILE     = REPO_ROOT / "results" / "convmix_rs" / "rs_cache.json"
LOG_DIR        = REPO_ROOT / "results" / "convmix_rs"

GEN_MODEL_PATH = (
    REPO_ROOT / "models" / "Llama-3.3-70B-Instruct"
    / "models--meta-llama--Llama-3.3-70B-Instruct"
    / "snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
)
PPL_MODEL_PATH = REPO_ROOT / "models" / "Llama-3.1-8B-Instruct"

# ---------------------------------------------------------------------------
# Prompt  (RS — Relation Synonym substitution, same as perturb_relation.py)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a linguistic data augmenter specializing in semantic-preserving "
    "perturbations for Question Answering datasets.\n\n"
    "Task: Rewrite the question by substituting ONLY the relations, predicates, "
    "or relational nouns with their synonyms or semantically equivalent phrases.\n\n"
    "STRICT RULES:\n"
    "- Do NOT change proper nouns (person names, place names, entity names).\n"
    "- The answer to the question must remain exactly the same.\n"
    "- The rewritten question must be grammatically correct and natural.\n"
    "- Do NOT add or remove constraints or facts.\n"
    "- Do NOT introduce temporal modifiers not in the original.\n"
    "- Return ONE single-line question only. No explanation.\n\n"
    "Examples:\n"
    "Original: What country bordering France contains an airport that serves Nijmegen?\n"
    "Perturbed: What country neighboring France contains an airport that serves Nijmegen?\n\n"
    "Original: what does jamaican people speak?\n"
    "Perturbed: what language does jamaican people communicate in?\n\n"
    "Original: Who is the author of the book Hamlet?\n"
    "Perturbed: Who wrote the book Hamlet?"
)

USER_TEMPLATE = (
    "Original question: {question}\n"
    "Rewrite by substituting only the relation/predicate words with synonyms. "
    "Keep all entities unchanged. Return one question only."
)

logger = logging.getLogger("generate_convmix_rs")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rs_gen_{ts}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh])
    logger.info("Log → %s", log_file)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(path: Path, max_memory: dict | None = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    p = str(path)
    logger.info("Loading %s …", path.name)
    try:
        tok = AutoTokenizer.from_pretrained(p, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(p, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    if max_memory:
        kwargs["max_memory"] = max_memory
        logger.info("max_memory: %s", max_memory)
    model = AutoModelForCausalLM.from_pretrained(p, **kwargs)
    model.eval()
    logger.info("Loaded %s", path.name)
    return tok, model


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
def clean_output(text: str) -> str:
    text = text.strip().replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"^(perturbed|rewritten|output|question|answer|result)\s*[:\-]\s*",
        "", text, flags=re.IGNORECASE,
    )
    if ". " in text and text.rstrip().endswith("?"):
        parts = text.split(". ")
        for part in reversed(parts):
            part = part.strip()
            if part.endswith("?") and len(part) > 8:
                text = part
                break
    return text.strip(" \"'")


def normalize_trivial(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_trivial(original: str, candidate: str) -> bool:
    return normalize_trivial(original) == normalize_trivial(candidate)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------
def batched_generate(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    batch_messages: Sequence[List[Dict[str, str]]],
    max_new_tokens: int = 128,
) -> List[str]:
    prompts = [
        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    input_len = enc["input_ids"].shape[1]
    decoded = tok.batch_decode(out[:, input_len:], skip_special_tokens=True)
    return [clean_output(x) for x in decoded]


# ---------------------------------------------------------------------------
# PPL scoring
# ---------------------------------------------------------------------------
def compute_ppl_batch(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    texts: List[str],
) -> List[float]:
    if not texts:
        return []
    valid_mask = [bool(t and t.strip()) for t in texts]
    valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]
    if not valid_texts:
        return [float("inf")] * len(texts)

    orig_side = tok.padding_side
    tok.padding_side = "right"
    enc = tok(
        valid_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(model.device)
    tok.padding_side = orig_side

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels         = input_ids.clone()
    labels[attention_mask == 0] = -100

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())
        real_counts = (shift_labels != -100).sum(dim=1).float().clamp(min=1)
        ppls = torch.exp(token_loss.sum(dim=1) / real_counts).cpu().tolist()

    result, vi = [], 0
    for ok in valid_mask:
        if ok:
            result.append(ppls[vi]); vi += 1
        else:
            result.append(float("inf"))
    return result


# ---------------------------------------------------------------------------
# Core: process a batch of ConvMix KB turns
# ---------------------------------------------------------------------------
def process_batch(
    batch: List[Dict],
    gen_tok, gen_model,
    ppl_tok, ppl_model,
    bert_scorer: BERTScorer,
    *,
    ppl_threshold: float,
    ppl_ratio: float,
    bertscore_threshold: float,
    jaccard_min: float,
    attempts: int,
    max_new_tokens: int,
) -> List[Dict]:
    B = len(batch)
    questions = [r["question"] for r in batch]
    orig_ppls = compute_ppl_batch(ppl_tok, ppl_model, questions)

    chosen: List[Optional[Tuple[str, Dict]]] = [None] * B
    attempts_left = [attempts] * B
    last_attempt: List[Dict] = [{} for _ in range(B)]

    while any(a > 0 and c is None for a, c in zip(attempts_left, chosen)):
        pending = [i for i, (a, c) in enumerate(zip(attempts_left, chosen))
                   if a > 0 and c is None]

        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(question=questions[i])},
            ]
            for i in pending
        ]

        candidates = batched_generate(gen_tok, gen_model, messages, max_new_tokens)

        trivial_mask = [is_trivial(questions[i], c) for i, c in zip(pending, candidates)]

        # --- Step 1: BERTScore (all non-trivial candidates) ---
        non_trivial_idx = [j for j, t in enumerate(trivial_mask) if not t]
        if non_trivial_idx:
            bs_cands = [candidates[j] for j in non_trivial_idx]
            bs_refs  = [questions[pending[j]] for j in non_trivial_idx]
            _, _, F1 = bert_scorer.score(bs_cands, bs_refs)
            bert_map = {j: float(F1[k].cpu()) for k, j in enumerate(non_trivial_idx)}
        else:
            bert_map = {}
        bert_pass_idx = {j for j in non_trivial_idx if bert_map.get(j, 0.0) > bertscore_threshold}

        # --- Step 2: PPL (only BERTScore-passing candidates) ---
        ppls = compute_ppl_batch(
            ppl_tok, ppl_model,
            [candidates[j] if j in bert_pass_idx else "" for j in range(len(pending))],
        )
        ppl_pass_idx = {
            j for j in bert_pass_idx
            if (ppls[j] / max(orig_ppls[pending[j]], 1e-6)) < ppl_ratio
            and ppls[j] < ppl_threshold
        }

        # --- Step 3: Jaccard (only PPL-passing candidates) ---
        jac_vals = [
            jaccard(questions[pending[j]], candidates[j]) if j in ppl_pass_idx else 0.0
            for j in range(len(pending))
        ]
        jac_pass_idx = {
            j for j in ppl_pass_idx
            if jaccard_min <= jac_vals[j] < 0.99
        }

        for j, (i_batch, candidate) in enumerate(zip(pending, candidates)):
            attempts_left[i_batch] -= 1
            bv      = bert_map.get(j, 0.0)
            ppl_val = ppls[j]
            op      = orig_ppls[i_batch]
            jv      = jac_vals[j]

            if trivial_mask[j]:
                reason = "trivial"
            elif bv <= bertscore_threshold:
                reason = "bertscore"
            elif (ppl_val / max(op, 1e-6)) >= ppl_ratio:
                reason = "ppl_ratio"
            elif ppl_val >= ppl_threshold:
                reason = "ppl_abs"
            elif jv >= 0.99:
                reason = "jaccard_identical"
            elif jv < jaccard_min:
                reason = "jaccard_too_different"
            else:
                reason = "ok"

            if reason == "ok" and j in jac_pass_idx:
                chosen[i_batch] = (candidate, {
                    "ppl_orig":  round(op, 2),
                    "ppl_pert":  round(ppl_val, 2),
                    "ppl_ratio": round(ppl_val / max(op, 1e-6), 3),
                    "bertscore": round(bv, 4),
                    "jaccard":   round(jv, 4),
                    "reason":    "ok",
                })
            else:
                last_attempt[i_batch] = {
                    "last_reason":    reason,
                    "last_candidate": candidate[:120],
                    "ppl_orig":       round(op, 2),
                    "ppl_pert":       round(ppl_val, 2) if j in bert_pass_idx else None,
                    "ppl_ratio":      round(ppl_val / max(op, 1e-6), 3) if j in bert_pass_idx else None,
                    "bertscore":      round(bv, 4) if j in non_trivial_idx else None,
                    "jaccard":        round(jv, 4) if j in ppl_pass_idx else None,
                }

    results = []
    for i, src in enumerate(batch):
        if chosen[i] is not None:
            rs_q, vmeta = chosen[i]
            results.append({
                "question_id": src["question_id"],
                "rs_question": rs_q,
                "valid": True,
                "meta": vmeta,
            })
        else:
            results.append({
                "question_id": src["question_id"],
                "rs_question": src["question"],   # fallback: keep original
                "valid": False,
                "meta": {"reason": "constraint_unsatisfied",
                         "last_attempt": last_attempt[i] or None},
            })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",             action="store_true", help="Process first 20 KB turns only")
    ap.add_argument("--resume",              action="store_true", help="Resume from cache")
    ap.add_argument("--retry-fallbacks",     action="store_true",
                    help="Retry only fallback (constraint-unsatisfied) turns with relaxed thresholds")
    ap.add_argument("--batch-size",          type=int,   default=4)
    ap.add_argument("--attempts",            type=int,   default=5)
    ap.add_argument("--max-new-tokens",      type=int,   default=128)
    ap.add_argument("--ppl-threshold",       type=float, default=400.0)
    ap.add_argument("--ppl-ratio",           type=float, default=3.0)
    ap.add_argument("--bertscore-threshold", type=float, default=0.85)
    ap.add_argument("--jaccard-min",         type=float, default=0.3)
    ap.add_argument("--gpu-list",            default="0,1,2,3")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(LOG_DIR)

    logger.info(
        "=== ConvMix RS Generation (Relation Synonym/p7, Llama-3.3-70B) ===  "
        "GPUs=%s  batch=%d  attempts=%d  ppl_ratio=%.1f  "
        "ppl_cap=%.0f  jac_min=%.2f  bert=%.2f%s",
        args.gpu_list, args.batch_size, args.attempts, args.ppl_ratio,
        args.ppl_threshold, args.jaccard_min, args.bertscore_threshold,
        "  [RETRY-FALLBACKS]" if args.retry_fallbacks else "",
    )

    # Load benchmark
    with open(TEST_SET_IN) as f:
        benchmark = json.load(f)
    logger.info("Loaded %d conversations, %d total turns",
                len(benchmark), sum(len(c["questions"]) for c in benchmark))

    # Load cache (always needed for retry mode)
    cache: Dict[str, str] = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        logger.info("Loaded cache: %d entries", len(cache))
    elif args.resume or args.retry_fallbacks:
        logger.warning("Cache file not found: %s", CACHE_FILE)

    # Collect KB-answerable turns
    kb_turns: List[Tuple[int, int]] = []
    for ci, conv in enumerate(benchmark):
        for ti, turn in enumerate(conv["questions"]):
            if turn.get("answer_src") == "kb":
                kb_turns.append((ci, ti))
    logger.info("KB-answerable turns: %d", len(kb_turns))

    if args.dry_run:
        kb_turns = kb_turns[:20]
        logger.info("[DRY RUN] Processing %d turns only", len(kb_turns))

    # Identify turns to process
    if args.retry_fallbacks:
        # Find fallback question_ids from meta file and evict them from cache
        if not META_OUT.exists():
            logger.error("Meta file not found: %s — run full generation first.", META_OUT)
            sys.exit(1)
        with open(META_OUT) as f:
            prior_meta = json.load(f)
        fallback_qids = {m["question_id"] for m in prior_meta if not m.get("valid", True)}
        logger.info("Fallback turns to retry: %d", len(fallback_qids))
        # Remove fallbacks from cache so they get reprocessed
        for qid in fallback_qids:
            cache.pop(qid, None)
        todo_turns = [
            (ci, ti) for ci, ti in kb_turns
            if benchmark[ci]["questions"][ti]["question_id"] in fallback_qids
        ]
    else:
        todo_turns = [
            (ci, ti) for ci, ti in kb_turns
            if benchmark[ci]["questions"][ti]["question_id"] not in cache
        ]
    logger.info("To process: %d  Cached: %d", len(todo_turns), len(kb_turns) - len(todo_turns))

    # Load models only if needed
    if todo_turns:
        n_gpus = len(args.gpu_list.split(","))
        if n_gpus >= 3:
            max_mem = {0: "76GiB", 1: "76GiB", 2: "55GiB"}
        elif n_gpus == 2:
            max_mem = {0: "76GiB", 1: "55GiB"}
        else:
            max_mem = None
        gen_tok, gen_model = load_model(GEN_MODEL_PATH, max_memory=max_mem)
        ppl_tok, ppl_model = load_model(PPL_MODEL_PATH)
        bert_scorer = BERTScorer(
            lang="en", model_type="roberta-large",
            device="cuda:0", rescale_with_baseline=False,
        )
        logger.info("All models loaded.")
    else:
        logger.info("All turns cached — skipping model load.")
        gen_tok = gen_model = ppl_tok = ppl_model = bert_scorer = None

    # Process in batches
    n_valid = n_fallback = 0
    t_start = time.time()
    n_batches = math.ceil(len(todo_turns) / args.batch_size) if todo_turns else 0

    for batch_idx, start in enumerate(range(0, len(todo_turns), args.batch_size)):
        batch_idxs = todo_turns[start: start + args.batch_size]
        batch_src = []
        for ci, ti in batch_idxs:
            turn = benchmark[ci]["questions"][ti]
            batch_src.append({
                "question_id": turn["question_id"],
                "question":    turn["question"],
            })

        results = process_batch(
            batch_src, gen_tok, gen_model, ppl_tok, ppl_model, bert_scorer,
            ppl_threshold=args.ppl_threshold,
            ppl_ratio=args.ppl_ratio,
            bertscore_threshold=args.bertscore_threshold,
            jaccard_min=args.jaccard_min,
            attempts=args.attempts,
            max_new_tokens=args.max_new_tokens,
        )

        for r in results:
            cache[r["question_id"]] = r["rs_question"]
            if r["valid"]:
                n_valid += 1
            else:
                n_fallback += 1

        elapsed = time.time() - t_start
        done = min((batch_idx + 1) * args.batch_size, len(todo_turns))
        rate = done / elapsed if elapsed > 0 else 1
        eta = (len(todo_turns) - done) / rate if rate > 0 else 0
        logger.info(
            "batch %d/%d  valid=%d fallback=%d  elapsed=%.1fmin ETA=%.1fmin",
            batch_idx + 1, n_batches, n_valid, n_fallback, elapsed / 60, eta / 60,
        )

        # Periodic cache save
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)

    # Build output benchmark and meta
    out_benchmark = copy.deepcopy(benchmark)
    meta: List[Dict] = []

    for ci, ti in kb_turns:
        turn   = benchmark[ci]["questions"][ti]
        qid    = turn["question_id"]
        orig_q = turn["question"]
        rs_q   = cache.get(qid, orig_q)

        out_benchmark[ci]["questions"][ti]["question"]         = rs_q
        out_benchmark[ci]["questions"][ti]["rs_orig_question"] = orig_q
        out_benchmark[ci]["questions"][ti]["is_rs_perturbed"]  = True

        orig_words = set(re.findall(r'\w+', orig_q.lower()))
        rs_words   = set(re.findall(r'\w+', rs_q.lower()))
        union = orig_words | rs_words
        jac   = len(orig_words & rs_words) / len(union) if union else 1.0

        meta.append({
            "question_id": qid,
            "conv_id":     benchmark[ci]["conv_id"],
            "turn":        ti,
            "domain":      benchmark[ci].get("domain", ""),
            "original":    orig_q,
            "perturbed":   rs_q,
            "jaccard":     round(jac, 4),
            "valid":       rs_q != orig_q,
        })

    with open(TEST_SET_OUT, "w") as f:
        json.dump(out_benchmark, f, indent=2)
    logger.info("RS benchmark → %s", TEST_SET_OUT)

    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Meta         → %s", META_OUT)

    jaccards = [m["jaccard"] for m in meta]
    n_valid_meta = sum(1 for m in meta if m["valid"])
    logger.info(
        "=== Summary ===  Total KB turns: %d  Valid: %d (%.1f%%)  Fallback: %d",
        len(meta), n_valid_meta, 100 * n_valid_meta / max(len(meta), 1),
        len(meta) - n_valid_meta,
    )
    if jaccards:
        logger.info("Mean Jaccard: %.3f  Median: %.3f",
                    sum(jaccards) / len(jaccards),
                    sorted(jaccards)[len(jaccards) // 2])
    logger.info("Next: run EXPLAIGNN with explaignn_rs.yml")


if __name__ == "__main__":
    main()
