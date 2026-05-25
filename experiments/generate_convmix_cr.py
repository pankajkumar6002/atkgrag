#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
generate_convmix_cr.py — CR (Compositional Restructuring / Attack 4) perturbation for ConvMix.

Uses Llama-3.3-70B-Instruct for adversarial paraphrasing of all KB-answerable ConvMix test
turns. Quality is gated by the same 70B model via:
  1. LLM-judge (YES/NO self-critique): does the rewrite preserve meaning and answer?
  2. 70B log-prob PPL: is the rewrite fluent (PPL ratio vs original < threshold)?

No separate BERTScore or 8B PPL model needed.

Prompt: Attack 4 from prompt.md — "Adversarial Paraphrase Attack": maximally change surface
form while keeping meaning, answer, and information needs identical.

Outputs:
  _benchmarks/convmix/test_set/test_set_ALL_cr.json  — full benchmark with CR turns
  tmlr/results/convmix_cr/perturbation_meta.json      — per-turn stats
  tmlr/results/convmix_cr/cr_cache.json               — resume cache (keyed by question_id)

Usage:
  python  # set PYTHON_FB env var to override generate_convmix_cr.py
  python  # set PYTHON_FB env var to override generate_convmix_cr.py --dry-run
  python  # set PYTHON_FB env var to override generate_convmix_cr.py --resume
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXPLAIGNN_ROOT = REPO_ROOT / "external" / "EXPLAIGNN"
TEST_SET_IN    = EXPLAIGNN_ROOT / "_benchmarks/convmix/test_set/test_set_ALL.json"
TEST_SET_OUT   = EXPLAIGNN_ROOT / "_benchmarks/convmix/test_set/test_set_ALL_cr.json"
META_OUT       = Path("tmlr/results/convmix_cr/perturbation_meta.json")
CACHE_FILE     = Path("tmlr/results/convmix_cr/cr_cache.json")
LOG_DIR        = REPO_ROOT / "results" / "convmix_cr"

GEN_MODEL_PATH = (
    REPO_ROOT / "models" / "Llama-3.3-70B-Instruct"
    / "models--meta-llama--Llama-3.3-70B-Instruct"
    / "snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
)
# ---------------------------------------------------------------------------
# Prompt  (Attack 4 — Adversarial Paraphrase Attack, from prompt.md)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are given a question. "
    "Paraphrase it so that the wording and phrasing are as different as possible "
    "from the original while the meaning, answer, and information needs remain identical.\n\n"
    "Constraints:\n"
    "- You may replace phrases, change syntax, and reorder parts of the question.\n"
    "- Do not change the underlying entities, relations, or constraints.\n"
    "- Output only the paraphrased question."
)

USER_TEMPLATE = (
    "Question: {question}\n"
    "Perturbed question:"
)

JUDGE_SYSTEM = (
    "You are a quality checker for paraphrased questions.\n"
    "Given an original question and a rewritten version, reply with exactly one word: "
    "YES if the rewrite preserves the original meaning and answer, NO otherwise."
)

JUDGE_USER_TEMPLATE = (
    "Original: {original}\n"
    "Rewrite: {rewrite}\n"
    "Does the rewrite preserve the original meaning and answer? Reply YES or NO."
)

logger = logging.getLogger("generate_convmix_cr")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"cr_gen_{ts}.log"
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


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_trivial(original: str, candidate: str) -> bool:
    return normalize(original) == normalize(candidate)


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
# 70B-based scoring (PPL + LLM judge) — no separate model needed
# ---------------------------------------------------------------------------
def compute_ppl_batch(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    texts: List[str],
) -> List[float]:
    """Compute PPL of plain texts using the 70B model's own logits."""
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
    )
    # Move to the first device of a potentially sharded model
    first_device = next(model.parameters()).device
    enc = {k: v.to(first_device) for k, v in enc.items()}
    tok.padding_side = orig_side

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels         = input_ids.clone()
    labels[attention_mask == 0] = -100

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
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


def llm_judge_batch(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    originals: List[str],
    candidates: List[str],
) -> List[bool]:
    """Ask the 70B to self-judge whether each candidate preserves meaning. Returns YES/NO list."""
    if not originals:
        return []
    messages = [
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": JUDGE_USER_TEMPLATE.format(
                original=orig, rewrite=cand)},
        ]
        for orig, cand in zip(originals, candidates)
    ]
    # Generate with very low temperature — we only want YES/NO
    prompts = [
        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages
    ]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    first_device = next(model.parameters()).device
    enc = {k: v.to(first_device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=4,
            do_sample=False,
            temperature=1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    input_len = enc["input_ids"].shape[1]
    decoded = tok.batch_decode(out[:, input_len:], skip_special_tokens=True)
    return [d.strip().upper().startswith("YES") for d in decoded]


# ---------------------------------------------------------------------------
# ConvMix data helpers
# ---------------------------------------------------------------------------
def collect_kb_turns(benchmark: List[Dict]) -> List[Tuple[int, int]]:
    """Return (conv_idx, turn_idx) for all KB-answerable turns."""
    kb_turns = []
    for ci, conv in enumerate(benchmark):
        for ti, turn in enumerate(conv["questions"]):
            if turn.get("answer_src") == "kb":
                kb_turns.append((ci, ti))
    return kb_turns


# ---------------------------------------------------------------------------
# Core: process a batch of ConvMix turns
# ---------------------------------------------------------------------------
def process_batch(
    batch: List[Dict],
    gen_tok, gen_model,
    *,
    ppl_ratio: float,
    ppl_threshold: float,
    attempts: int,
    max_new_tokens: int,
) -> List[Dict]:
    """
    Filter chain (all using the same 70B):
      1. Trivial check (exact match after normalisation)
      2. LLM-judge: YES/NO meaning-preservation self-critique
      3. 70B log-prob PPL: ratio vs original < ppl_ratio AND abs < ppl_threshold
    Returns list of dicts with keys: question_id, cr_question, valid, meta.
    """
    B = len(batch)
    questions = [r["question"] for r in batch]
    orig_ppls = compute_ppl_batch(gen_tok, gen_model, questions)

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

        # Step 1: trivial filter
        trivial_mask = [is_trivial(questions[i], c) for i, c in zip(pending, candidates)]

        # Step 2: LLM-judge for non-trivial candidates
        judge_input_idx = [j for j, t in enumerate(trivial_mask) if not t]
        judge_results: Dict[int, bool] = {}
        if judge_input_idx:
            judge_pass = llm_judge_batch(
                gen_tok, gen_model,
                [questions[pending[j]] for j in judge_input_idx],
                [candidates[j] for j in judge_input_idx],
            )
            judge_results = {j: p for j, p in zip(judge_input_idx, judge_pass)}

        # Step 3: PPL for candidates that passed judge
        ppl_input_idx = [j for j in judge_input_idx if judge_results.get(j, False)]
        ppls: Dict[int, float] = {}
        if ppl_input_idx:
            ppl_vals = compute_ppl_batch(
                gen_tok, gen_model,
                [candidates[j] for j in ppl_input_idx],
            )
            ppls = {j: v for j, v in zip(ppl_input_idx, ppl_vals)}

        for j, (i_batch, candidate) in enumerate(zip(pending, candidates)):
            attempts_left[i_batch] -= 1
            op = orig_ppls[i_batch]
            ppl_val = ppls.get(j, float("inf"))

            if trivial_mask[j]:
                reason = "trivial"
            elif not judge_results.get(j, False):
                reason = "judge_no"
            elif (ppl_val / max(op, 1e-6)) >= ppl_ratio:
                reason = "ppl_ratio"
            elif ppl_val >= ppl_threshold:
                reason = "ppl_abs"
            else:
                reason = "ok"

            if reason == "ok":
                chosen[i_batch] = (candidate, {
                    "ppl_orig":  round(op, 2),
                    "ppl_pert":  round(ppl_val, 2),
                    "ppl_ratio": round(ppl_val / max(op, 1e-6), 3),
                    "judge":     "YES",
                    "reason":    "ok",
                })
            else:
                last_attempt[i_batch] = {
                    "last_reason":    reason,
                    "last_candidate": candidate[:120],
                    "ppl_orig":       round(op, 2),
                    "ppl_pert":       round(ppl_val, 2) if j in ppls else None,
                    "ppl_ratio":      round(ppl_val / max(op, 1e-6), 3) if j in ppls else None,
                    "judge":          "YES" if judge_results.get(j) else "NO",
                }

    results = []
    for i, src in enumerate(batch):
        if chosen[i] is not None:
            cr_q, vmeta = chosen[i]
            results.append({"question_id": src["question_id"], "cr_question": cr_q,
                            "valid": True, "meta": vmeta})
        else:
            results.append({"question_id": src["question_id"],
                            "cr_question": src["question"],
                            "valid": False,
                            "meta": {"reason": "constraint_unsatisfied",
                                     "last_attempt": last_attempt[i] or None}})
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Process first 20 KB turns only")
    ap.add_argument("--resume",  action="store_true", help="Resume from cache")
    ap.add_argument("--batch-size",          type=int,   default=4)
    ap.add_argument("--attempts",            type=int,   default=5)
    ap.add_argument("--max-new-tokens",      type=int,   default=128)
    ap.add_argument("--ppl-threshold", type=float, default=400.0)
    ap.add_argument("--ppl-ratio",     type=float, default=3.0)
    ap.add_argument("--gpu-list",      default="0,1,2,3")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(LOG_DIR)

    logger.info(
        "=== ConvMix CR Generation (Attack 4, Llama-3.3-70B) ===  "
        "GPUs=%s  batch=%d  attempts=%d  ppl_ratio=%.1f  ppl_cap=%.0f",
        args.gpu_list, args.batch_size, args.attempts, args.ppl_ratio, args.ppl_threshold,
    )

    # Load benchmark
    with open(TEST_SET_IN) as f:
        benchmark = json.load(f)
    logger.info("Loaded %d conversations, %d total turns",
                len(benchmark), sum(len(c["questions"]) for c in benchmark))

    # Resume cache
    cache: Dict[str, str] = {}
    if args.resume and CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        logger.info("Resumed cache: %d entries", len(cache))

    # Collect KB-answerable turns
    kb_turns = collect_kb_turns(benchmark)
    logger.info("KB-answerable turns: %d", len(kb_turns))
    if args.dry_run:
        kb_turns = kb_turns[:20]
        logger.info("[DRY RUN] Processing %d turns only", len(kb_turns))

    # Separate already-cached turns
    todo_turns = [(ci, ti) for ci, ti in kb_turns
                  if benchmark[ci]["questions"][ti]["question_id"] not in cache]
    logger.info("Turns to process: %d  (cached: %d)", len(todo_turns), len(kb_turns) - len(todo_turns))

    # Load models only if there's work to do
    if todo_turns:
        # With 2 GPUs (CUDA:0=76GiB, CUDA:1=55GiB): keep headroom on the
        # GPU that shares memory with another process (CUDA:1 / physical GPU 2).
        n_gpus = len(args.gpu_list.split(","))
        if n_gpus >= 3:
            max_mem = {0: "76GiB", 1: "76GiB", 2: "55GiB"}
        elif n_gpus == 2:
            max_mem = {0: "76GiB", 1: "55GiB"}
        else:
            max_mem = None
        gen_tok, gen_model = load_model(GEN_MODEL_PATH, max_memory=max_mem)
        logger.info("Model loaded.")
    else:
        logger.info("All turns in cache — skipping model loading.")
        gen_tok = gen_model = None

    # Process in batches
    n_valid, n_fallback = 0, 0
    t_start = time.time()

    n_batches = math.ceil(len(todo_turns) / args.batch_size)
    for batch_idx, start in enumerate(range(0, len(todo_turns), args.batch_size)):
        batch_idxs = todo_turns[start: start + args.batch_size]
        batch_src = []
        for ci, ti in batch_idxs:
            turn = benchmark[ci]["questions"][ti]
            batch_src.append({
                "question_id": turn["question_id"],
                "question":    turn["question"],
                "answer":      (turn.get("answer_text")
                                or (turn["answers"][0]["label"] if turn.get("answers") else "?")),
            })

        results = process_batch(
            batch_src, gen_tok, gen_model,
            ppl_threshold=args.ppl_threshold,
            ppl_ratio=args.ppl_ratio,
            attempts=args.attempts,
            max_new_tokens=args.max_new_tokens,
        )

        for r in results:
            cache[r["question_id"]] = r["cr_question"]
            if r["valid"]:
                n_valid += 1
            else:
                n_fallback += 1

        elapsed = time.time() - t_start
        done = (batch_idx + 1) * args.batch_size
        rate = done / elapsed if elapsed > 0 else 1
        eta = (len(todo_turns) - done) / rate if rate > 0 else 0
        logger.info("batch %d/%d  valid=%d fallback=%d  elapsed=%.1fmin ETA=%.1fmin",
                    batch_idx + 1, n_batches, n_valid, n_fallback,
                    elapsed / 60, eta / 60)

        # Periodic cache save
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)

    # Build output benchmark and meta
    out_benchmark = copy.deepcopy(benchmark)
    meta: List[Dict] = []

    for ci, ti in kb_turns:
        turn = benchmark[ci]["questions"][ti]
        qid  = turn["question_id"]
        orig_q = turn["question"]
        cr_q = cache.get(qid, orig_q)

        out_benchmark[ci]["questions"][ti]["question"]         = cr_q
        out_benchmark[ci]["questions"][ti]["cr_orig_question"] = orig_q
        out_benchmark[ci]["questions"][ti]["is_cr_perturbed"]  = True

        meta.append({
            "question_id": qid,
            "conv_id":     benchmark[ci]["conv_id"],
            "turn":        ti,
            "domain":      benchmark[ci].get("domain", ""),
            "original":    orig_q,
            "perturbed":   cr_q,
            "valid":       cr_q != orig_q,
        })

    with open(TEST_SET_OUT, "w") as f:
        json.dump(out_benchmark, f, indent=2)
    logger.info("Perturbed benchmark → %s", TEST_SET_OUT)

    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Perturbation meta   → %s", META_OUT)

    n_valid_meta = sum(1 for m in meta if m["valid"])
    logger.info("=== Summary ===  Total KB turns: %d  Valid: %d (%.1f%%)  Fallback: %d",
                len(meta), n_valid_meta, 100 * n_valid_meta / max(len(meta), 1),
                len(meta) - n_valid_meta)
    logger.info("Next step: run_explaignn_cr.sh")


if __name__ == "__main__":
    main()
