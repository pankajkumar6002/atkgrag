#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
e1_e2_gmt_kbqa.py — Experiments E1/E2: Stage-wise Attack using GMT-KBQA (T5-base)

Evaluates the pretrained GMT-KBQA T5 model on all 5 perturbation types
for CWQ (E1) and WebQSP (E2).

GMT-KBQA generates s-expressions (logical forms) from questions, then
executes them against Freebase via Virtuoso at http://localhost:3001/sparql.

Strategy: substitute only the question text in the GMT-KBQA generation data
(replacing 'question' field) — entity/relation candidates remain from the
original preprocessing.  This isolates the question encoder's sensitivity.

Paths:
  Model CWQ:    GMT-KBQA-data/GMT-KBQA/exps/CWQ_GMT_KBQA/model_saved/
  Model WebQSP: GMT-KBQA-data/GMT-KBQA/exps/WebQSP_GMT_KBQA/model_saved/
  Data CWQ:     GMT-KBQA-data/GMT-KBQA/data/CWQ_uploaded/generation/merged/CWQ_test.json
  Data WebQSP:  GMT-KBQA-data/GMT-KBQA/data/WebQSP_uploaded/generation/merged/WebQSP_test.json
  Perturbations: data/perturbations_raw/{CWQ,WebQSP}/{variant}/

Usage:
  cd external/GMT-KBQA
  CUDA_VISIBLE_DEVICES=0,1 python ../tmlr/scripts/experiments/e1_e2_gmt_kbqa.py \
    --dataset CWQ --variant greedy \
    --out-dir results/e1_cwq_gmt_kbqa

  CUDA_VISIBLE_DEVICES=0,1 python ../tmlr/scripts/experiments/e1_e2_gmt_kbqa.py \
    --dataset WebQSP --variant greedy \
    --out-dir results/e2_webqsp_gmt_kbqa

NOTE: Must be run from external/GMT-KBQA/ (the repo root),
      or set --gmt-root accordingly.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

# Paths
GMT_KBQA_ROOT = REPO_ROOT / "external" / "GMT-KBQA"
GMT_DATA_ROOT = REPO_ROOT / "external" / "GMT-KBQA-data"
PERT_ROOT = REPO_ROOT / "data" / "perturbations_raw"
SPARQL_ENDPOINT = "http://localhost:3001/sparql"

PERT_TYPES = [
    "active_passive_flip",
    "adversarial_entity",
    "entity_alias_substitution",
    "question_reordering",
    "relation_synonym",
]
PERT_TO_CONDITION = {
    "entity_alias_substitution": "Lex",
    "relation_synonym": "Lex",
    "active_passive_flip": "Str",
    "question_reordering": "Str",
    "adversarial_entity": "Cmp",
}

SCRIPTS_ROOT = Path(__file__).parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from metrics import exact_match_kgqa, f1_kgqa, mean_ci, save_results


# ---------------------------------------------------------------------------
# SPARQL execution
# ---------------------------------------------------------------------------

def sparql_execute(query: str, endpoint: str = SPARQL_ENDPOINT) -> Optional[List[str]]:
    """Execute SPARQL and return list of answer strings, or None on error."""
    try:
        params = urllib.parse.urlencode({
            "query": query,
            "format": "application/sparql-results+json"
        })
        req = urllib.request.Request(f"{endpoint}?{params}")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        bindings = data.get("results", {}).get("bindings", [])
        answers = []
        for b in bindings:
            for key in b:
                v = b[key].get("value", "")
                # Extract MID from URI
                if "rdf.freebase.com/ns/" in v:
                    v = v.split("rdf.freebase.com/ns/")[-1]
                answers.append(v)
        return answers
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GMT-KBQA generation
# ---------------------------------------------------------------------------

def load_gmt_model(model_dir: Path, device: str = "cuda"):
    """Load GMT-KBQA T5 model and tokenizer from saved checkpoint.

    GMT-KBQA saves the model as T5_generation (self.t5 = T5ForConditionalGeneration),
    so the state_dict keys are prefixed with 't5.'.  We strip that prefix and load
    directly into T5ForConditionalGeneration.
    """
    if str(GMT_KBQA_ROOT) not in sys.path:
        sys.path.insert(0, str(GMT_KBQA_ROOT))

    from transformers import T5ForConditionalGeneration, AutoTokenizer

    tokenizer_dir = model_dir / "custom_tokenizer"
    if tokenizer_dir.exists():
        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    else:
        tok = AutoTokenizer.from_pretrained("t5-base")

    # Load config explicitly from config_file.json if config.json absent
    from transformers import T5Config
    config_path = model_dir / "config.json"
    config = T5Config.from_pretrained(str(config_path))

    # Load raw state dict and strip 't5.' wrapper prefix
    ckpt_path = model_dir / "pytorch_model.bin"
    raw_sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # Keys are either 't5.xxx' (wrapped) or 'xxx' (direct T5) — normalise
    if any(k.startswith("t5.") for k in raw_sd):
        sd = {k[len("t5."):]: v for k, v in raw_sd.items() if k.startswith("t5.")}
    else:
        sd = raw_sd

    model = T5ForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [GMT] Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"  [GMT] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

    model = model.to(torch.float16).to(device)
    model.eval()
    print(f"  [GMT] Loaded from {ckpt_path} — {sum(p.numel() for p in model.parameters()):,} params")
    return tok, model


def generate_sexprs_batch(
    tok,
    model,
    input_texts: List[str],
    beam_size: int = 10,
    max_tgt_len: int = 190,
    max_src_len: int = 256,
    batch_size: int = 8,
    device: str = "cuda",
) -> List[List[str]]:
    """
    Generate s-expressions for a list of input texts.
    Returns list of lists (top-beam_size candidates per input).
    """
    all_outputs = []
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i: i + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_src_len,
        ).to(device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_tgt_len,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                early_stopping=True,
                decoder_start_token_id=model.config.pad_token_id or 0,
            )
        # out shape: (batch_size * beam_size, seq_len)
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        # Regroup: beam_size sequences per input
        for j in range(len(batch)):
            all_outputs.append(decoded[j * beam_size: (j + 1) * beam_size])
    return all_outputs


def denormalize_sexpr(normed_expr: str, record: Dict[str, Any]) -> str:
    """
    Convert a normed s-expression (text labels in [ ]) back to Freebase MIDs.
    Builds label→MID maps from the per-question candidate lists.

    Logic mirrors GMT-KBQA/eval_topk_prediction_final.py:denormalize_s_expr_new.
    For perturbation experiments we only substitute question text, so the same
    entity/relation candidates apply to both clean and perturbed questions.
    """
    # Build entity label → MID
    ent_map: Dict[str, str] = {}
    for ent in (record.get("disambiguated_cand_entity") or []):
        if isinstance(ent, dict) and ent.get("id") and ent.get("label"):
            ent_map[ent["label"].lower()] = ent["id"]
    for ent in (record.get("cand_entity_list") or []):
        if isinstance(ent, dict) and ent.get("id") and ent.get("label"):
            ent_map.setdefault(ent["label"].lower(), ent["id"])
    for k, v in (record.get("gold_entity_map") or {}).items():
        ent_map[k.lower()] = v

    # Build relation label → relation_id  (textualized → dotted.path)
    rel_map: Dict[str, str] = {}
    for rel_entry in (record.get("cand_relation_list") or []):
        if len(rel_entry) >= 1:
            rel_id = rel_entry[0]
            textualized = _textualize_relation(rel_id).lower()
            rel_map[textualized] = rel_id

    # Parse tokens and resolve [ segment ] blocks
    expr = normed_expr
    tokens = expr.split(" ")
    result_tokens: List[str] = []
    in_bracket = False
    seg_tokens: List[str] = []

    for tok in tokens:
        if tok == "[":
            in_bracket = True
            seg_tokens = []
        elif tok == "]":
            in_bracket = False
            seg = " ".join(seg_tokens).strip()
            seg_lower = seg.lower()
            if seg_lower in ent_map:
                result_tokens.append(ent_map[seg_lower])
            elif seg_lower in rel_map:
                result_tokens.append(rel_map[seg_lower])
            else:
                # fallback: try to reverse-textualize as relation
                # e.g. "sports , sports team , championships" → "sports.sports_team.championships"
                raw = seg_lower.replace(" , ", ".").replace(" ", "_")
                result_tokens.append(raw)
        elif in_bracket:
            seg_tokens.append(tok)
        else:
            result_tokens.append(tok)

    return " ".join(result_tokens).replace("( ", "(").replace(" )", ")")


def sexpr_to_answer(sexpr: str, record: Optional[Dict[str, Any]] = None,
                    endpoint: str = SPARQL_ENDPOINT) -> Optional[List[str]]:
    """Convert normed s-expression → denormalize → SPARQL → execute."""
    if str(GMT_KBQA_ROOT) not in sys.path:
        sys.path.insert(0, str(GMT_KBQA_ROOT))
    try:
        from executor.logic_form_util import lisp_to_sparql
        denormed = denormalize_sexpr(sexpr, record or {}) if record else sexpr
        sparql_query = lisp_to_sparql(denormed)
        return sparql_execute(sparql_query, endpoint)
    except Exception:
        return None


def answers_from_candidates(candidates: List[str],
                            record: Optional[Dict[str, Any]] = None) -> List[str]:
    """Try each candidate s-expression until one executes and returns answers."""
    for sexpr in candidates:
        ans = sexpr_to_answer(sexpr.strip(), record)
        if ans:
            return ans
    return []


def normed_sexpr_em(pred: str, gold: str) -> float:
    """Exact match on normed s-expressions (both lowercased, whitespace-normalised)."""
    return float(pred.strip().lower() == gold.strip().lower())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gmt_gen_data(gen_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load GMT-KBQA generation data; return dict {ID: record}."""
    with gen_file.open() as f:
        records = json.load(f)
    return {r["ID"]: r for r in records}


def _textualize_relation(r: str) -> str:
    """Mirror GMT-KBQA gen_mtl_dataset._textualize_relation: replace _ → ' ' and . → ' , '."""
    if "_" in r:
        r = r.replace("_", " ")
    if "." in r:
        r = r.replace(".", " , ")
    return r


def build_input_text(record: Dict[str, Any], question_override: Optional[str] = None,
                     top_rels: int = 10) -> str:
    """
    Reconstruct the GMT-KBQA model input string exactly as used during training.

    Format (from GMT-KBQA/generation/models/T5_models_final.py lines 255-263):
      "translate to s-expression: {question} [rel] {rel1} [rel] {rel2} ..."
    All lowercased; relations from cand_relation_list filtered to logit > 0, top top_rels kept.
    """
    q = question_override or record.get("question", "")
    # Candidate relations: each entry is [relation_id, logit, textual_form]
    cand_rels = record.get("cand_relation_list") or []
    # Keep relations with logit > 0, sorted descending by logit
    pos_rels = sorted(
        [(r[0], r[1]) for r in cand_rels if len(r) >= 2 and r[1] > 0],
        key=lambda x: -x[1]
    )[:top_rels]
    src = "translate to s-expression: " + q
    for rel_id, _ in pos_rels:
        src += " [rel] " + _textualize_relation(str(rel_id))
    return src.lower()


def load_perturbations_cwq(pert_file: Path) -> Dict[str, str]:
    """Returns {ID: perturbed_question}"""
    with pert_file.open() as f:
        records = json.load(f)
    return {r["ID"]: str(r.get("question") or "").strip() for r in records
            if r.get("ID") and r.get("question")}


def load_perturbations_webqsp(pert_file: Path) -> Dict[str, str]:
    """Returns {QuestionId: perturbed_ProcessedQuestion}"""
    with pert_file.open() as f:
        data = json.load(f)
    questions = data["Questions"] if isinstance(data, dict) else data
    return {r["QuestionId"]: str(r.get("ProcessedQuestion") or "").strip()
            for r in questions if r.get("QuestionId") and r.get("ProcessedQuestion")}


def load_gold_answers_cwq(gen_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Gold answers are MIDs in 'answer' field of generation data."""
    return {qid: r.get("answer", []) for qid, r in gen_data.items()}


def load_gold_answers_webqsp(gen_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """WebQSP gold answers from Parses field."""
    out = {}
    for qid, r in gen_data.items():
        answers = []
        for parse in r.get("Parses", []):
            for a in parse.get("Answers", []):
                name = a.get("EntityName") or a.get("AnswerArgument", "")
                if name:
                    answers.append(str(name))
        # Also include answer MIDs if present
        if "answer" in r:
            answers.extend(str(a) for a in r["answer"])
        out[qid] = list(set(answers))
    return out


# ---------------------------------------------------------------------------
# Per-type evaluation
# ---------------------------------------------------------------------------

def evaluate_type(
    pert_type: str,
    variant: str,
    dataset: str,
    gen_data: Dict[str, Dict[str, Any]],
    gold_answers: Dict[str, List[str]],
    tok, model,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    # Load perturbations
    if dataset == "CWQ":
        pert_file = PERT_ROOT / "CWQ" / variant / \
                    f"ComplexWebQuestions_test_pert_{pert_type}.json"
        load_fn = load_perturbations_cwq
    else:
        pert_file = PERT_ROOT / "WebQSP" / variant / \
                    f"WebQSP.test_pert_{pert_type}.json"
        load_fn = load_perturbations_webqsp

    if not pert_file.exists():
        return {"skipped": True, "reason": f"not found: {pert_file}"}

    pert_qs = load_fn(pert_file)
    common_ids = sorted(set(gen_data) & set(pert_qs) & set(gold_answers))
    if not common_ids:
        return {"skipped": True, "reason": "no overlapping IDs"}

    print(f"    {len(common_ids)} questions, generating s-expressions...")

    # Build input texts
    orig_inputs = [build_input_text(gen_data[qid]) for qid in common_ids]
    pert_inputs = [build_input_text(gen_data[qid], pert_qs[qid]) for qid in common_ids]

    # Generate
    print(f"    Generating (clean)...")
    orig_candidates = generate_sexprs_batch(
        tok, model, orig_inputs, beam_size=args.beam_size,
        batch_size=args.batch_size, device=args.device
    )
    print(f"    Generating (perturbed)...")
    pert_candidates = generate_sexprs_batch(
        tok, model, pert_inputs, beam_size=args.beam_size,
        batch_size=args.batch_size, device=args.device
    )

    # Execute SPARQL in parallel and score (normed sexpr EM + execution EM)
    per_q = []
    em_c_all, em_p_all, f1_c_all, f1_p_all = [], [], [], []
    sexpr_em_c_all, sexpr_em_p_all = [], []

    sparql_workers = getattr(args, "sparql_workers", 16)
    print(f"    Executing SPARQL + scoring ({len(common_ids)} q, workers={sparql_workers})...")

    def _score_one(idx_qid):
        idx, qid = idx_qid
        rec = gen_data[qid]
        gold = gold_answers.get(qid, [])
        gold_normed = rec.get("normed_sexpr", "").strip().lower()

        orig_ans = answers_from_candidates(orig_candidates[idx], rec)
        pert_ans = answers_from_candidates(pert_candidates[idx], rec)
        orig_pred_str = orig_candidates[idx][0] if orig_candidates[idx] else ""
        pert_pred_str = pert_candidates[idx][0] if pert_candidates[idx] else ""

        # Normed s-expression EM (primary — no SPARQL needed)
        sx_em_c = normed_sexpr_em(orig_pred_str, gold_normed)
        sx_em_p = normed_sexpr_em(pert_pred_str, gold_normed)

        # Execution EM (secondary — requires SPARQL + denormalization)
        if gold and isinstance(gold[0], str) and gold[0].startswith("m."):
            em_c = float(any(g in orig_ans for g in gold)) if orig_ans else 0.0
            em_p = float(any(g in pert_ans for g in gold)) if pert_ans else 0.0
        else:
            orig_str = " ".join(orig_ans) if orig_ans else orig_pred_str
            pert_str = " ".join(pert_ans) if pert_ans else pert_pred_str
            em_c = exact_match_kgqa(orig_str, gold)
            em_p = exact_match_kgqa(pert_str, gold)
        f1_c = f1_kgqa(" ".join(orig_ans), gold) if orig_ans else 0.0
        f1_p = f1_kgqa(" ".join(pert_ans), gold) if pert_ans else 0.0

        return idx, {
            "id": qid,
            "sexpr_em_clean": sx_em_c, "sexpr_em_pert": sx_em_p,
            "em_clean": em_c, "em_pert": em_p,
            "f1_clean": f1_c, "f1_pert": f1_p,
            "orig_sexpr": orig_pred_str, "pert_sexpr": pert_pred_str,
            "gold_normed_sexpr": gold_normed,
            "orig_ans": orig_ans[:5], "pert_ans": pert_ans[:5],
        }

    results_map: Dict[int, Dict] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=sparql_workers) as pool:
        futures = {pool.submit(_score_one, (i, qid)): i
                   for i, qid in enumerate(common_ids)}
        for fut in as_completed(futures):
            idx, rec = fut.result()
            results_map[idx] = rec
            completed += 1
            if completed % 100 == 0:
                print(f"      {completed}/{len(common_ids)}", flush=True)

    for idx in range(len(common_ids)):
        rec = results_map[idx]
        em_c_all.append(rec["em_clean"]); em_p_all.append(rec["em_pert"])
        f1_c_all.append(rec["f1_clean"]); f1_p_all.append(rec["f1_pert"])
        sexpr_em_c_all.append(rec["sexpr_em_clean"])
        sexpr_em_p_all.append(rec["sexpr_em_pert"])
        per_q.append(rec)

    dag_em = [c - p for c, p in zip(em_c_all, em_p_all)]
    dag_f1 = [c - p for c, p in zip(f1_c_all, f1_p_all)]
    dag_sx = [c - p for c, p in zip(sexpr_em_c_all, sexpr_em_p_all)]
    return {
        "condition": PERT_TO_CONDITION.get(pert_type, "?"),
        "pert_type": pert_type, "n": len(common_ids),
        "sexpr_EM_clean": mean_ci(sexpr_em_c_all, args.n_boot),
        "sexpr_EM_pert":  mean_ci(sexpr_em_p_all, args.n_boot),
        "delta_AG_sexpr": mean_ci(dag_sx, args.n_boot),
        "EM_clean": mean_ci(em_c_all, args.n_boot),
        "EM_pert":  mean_ci(em_p_all, args.n_boot),
        "delta_AG_EM": mean_ci(dag_em, args.n_boot),
        "F1_clean": mean_ci(f1_c_all, args.n_boot),
        "F1_pert":  mean_ci(f1_p_all, args.n_boot),
        "delta_AG_F1": mean_ci(dag_f1, args.n_boot),
        "per_question": per_q,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1/E2: GMT-KBQA stage-wise attack")
    p.add_argument("--dataset", choices=["CWQ", "WebQSP"], default="CWQ")
    p.add_argument("--variant", default="greedy")
    p.add_argument("--types", nargs="+", default=PERT_TYPES)
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / "e1_cwq_gmt_kbqa")
    p.add_argument("--gmt-root", type=Path, default=GMT_KBQA_ROOT)
    p.add_argument("--beam-size", type=int, default=10,
                   help="Beam size for T5 generation (10 is fast; 50 for full eval)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu", default="0,1")
    p.add_argument("--sparql-workers", type=int, default=16,
                   help="Parallel threads for SPARQL execution (default: 16)")
    p.add_argument("--resume", action="store_true",
                   help="Skip p-types whose per_question.json already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set out-dir based on dataset if not overridden
    if args.out_dir == REPO_ROOT / "results" / "e1_cwq_gmt_kbqa":
        if args.dataset == "WebQSP":
            args.out_dir = REPO_ROOT / "results" / "e2_webqsp_gmt_kbqa"

    (args.out_dir / args.variant).mkdir(parents=True, exist_ok=True)

    # Add GMT-KBQA root to sys.path
    if str(args.gmt_root) not in sys.path:
        sys.path.insert(0, str(args.gmt_root))

    # Load model
    if args.dataset == "CWQ":
        model_dir = GMT_DATA_ROOT / "exps" / "CWQ_GMT_KBQA" / "model_saved"
        gen_file = GMT_DATA_ROOT / "data" / "CWQ_uploaded" / "generation" / "merged" / "CWQ_test.json"
    else:
        model_dir = GMT_DATA_ROOT / "exps" / "WebQSP_GMT_KBQA" / "model_saved"
        gen_file = GMT_DATA_ROOT / "data" / "WebQSP_uploaded" / "generation" / "merged" / "WebQSP_test.json"

    print(f"Loading GMT-KBQA model from {model_dir}...")
    tok, model = load_gmt_model(model_dir, args.device)
    print("  Model loaded.")

    print(f"Loading generation data from {gen_file}...")
    gen_data = load_gmt_gen_data(gen_file)
    print(f"  {len(gen_data)} records")

    if args.dataset == "CWQ":
        gold_answers = load_gold_answers_cwq(gen_data)
    else:
        gold_answers = load_gold_answers_webqsp(gen_data)

    # Verify Virtuoso
    test_ans = sparql_execute(
        "SELECT ?name WHERE { <http://rdf.freebase.com/ns/m.0d05l6> "
        "<http://rdf.freebase.com/ns/type.object.name> ?name } LIMIT 1"
    )
    if test_ans:
        print(f"  Virtuoso OK: m.0d05l6 → {test_ans}")
    else:
        print("  WARNING: Virtuoso returned no results — answers will be empty!")

    all_results: Dict[str, Any] = {
        "variant": args.variant, "model": "GMT-KBQA (T5-base)",
        "dataset": args.dataset, "beam_size": args.beam_size,
        "sparql_endpoint": SPARQL_ENDPOINT,
    }

    for pert_type in args.types:
        out_file = args.out_dir / args.variant / f"{pert_type}_per_question.json"
        if args.resume and out_file.exists():
            print(f"\n[{pert_type}] — skipped (resume: {out_file.name} exists)")
            with out_file.open() as f:
                pq = json.load(f)
            em_c = [r["em_clean"] for r in pq]
            em_p = [r["em_pert"]  for r in pq]
            f1_c = [r["f1_clean"] for r in pq]
            f1_p = [r["f1_pert"]  for r in pq]
            sx_c = [r.get("sexpr_em_clean", 0.0) for r in pq]
            sx_p = [r.get("sexpr_em_pert",  0.0) for r in pq]
            dag_em = [c - p for c, p in zip(em_c, em_p)]
            dag_f1 = [c - p for c, p in zip(f1_c, f1_p)]
            dag_sx = [c - p for c, p in zip(sx_c, sx_p)]
            all_results[pert_type] = {
                "condition": PERT_TO_CONDITION.get(pert_type, "?"),
                "pert_type": pert_type, "n": len(pq),
                "sexpr_EM_clean": mean_ci(sx_c, args.n_boot),
                "sexpr_EM_pert":  mean_ci(sx_p, args.n_boot),
                "delta_AG_sexpr": mean_ci(dag_sx, args.n_boot),
                "EM_clean": mean_ci(em_c, args.n_boot),
                "EM_pert":  mean_ci(em_p, args.n_boot),
                "delta_AG_EM": mean_ci(dag_em, args.n_boot),
                "F1_clean": mean_ci(f1_c, args.n_boot),
                "F1_pert":  mean_ci(f1_p, args.n_boot),
                "delta_AG_F1": mean_ci(dag_f1, args.n_boot),
            }
            sx_cl = all_results[pert_type]["sexpr_EM_clean"]
            sx_pt = all_results[pert_type]["sexpr_EM_pert"]
            dag_sx_r = all_results[pert_type]["delta_AG_sexpr"]
            print(f"  sexpr_EM_clean={sx_cl['mean']:.3f}  sexpr_EM_pert={sx_pt['mean']:.3f}  "
                  f"ΔAG_sx={dag_sx_r['mean']:.3f} (loaded from disk)")
            continue

        print(f"\n[{pert_type}]")
        res = evaluate_type(
            pert_type, args.variant, args.dataset,
            gen_data, gold_answers, tok, model, args
        )
        all_results[pert_type] = {k: v for k, v in res.items() if k != "per_question"}
        if not res.get("skipped"):
            save_results(out_file, res["per_question"])
            sx = res["delta_AG_sexpr"]
            print(f"  sexpr_EM_clean={res['sexpr_EM_clean']['mean']:.3f}  "
                  f"sexpr_EM_pert={res['sexpr_EM_pert']['mean']:.3f}  "
                  f"ΔAG_sx={sx['mean']:.3f} [{sx['ci_lo']:.3f}, {sx['ci_hi']:.3f}]")

    # Summary table
    print("\n" + "=" * 75)
    print(f"Variant: {args.variant} | Model: GMT-KBQA (T5) | Dataset: {args.dataset}")
    print(f"{'Type':<30} {'Cond':<5} {'EM_cl':>7} {'EM_pt':>7} {'ΔAG':>7}")
    print("-" * 75)
    for pert_type in args.types:
        r = all_results.get(pert_type, {})
        if r.get("skipped"):
            print(f"  {pert_type:<28} {'?':<5} {'—':>7} {'—':>7} {'—':>7}  skipped")
            continue
        dag = r["delta_AG_EM"]
        print(f"  {pert_type:<28} {r['condition']:<5}"
              f" {r['EM_clean']['mean']:>7.3f}"
              f" {r['EM_pert']['mean']:>7.3f}"
              f" {dag['mean']:>7.3f}")

    save_results(args.out_dir / args.variant / "summary.json", all_results)
    print(f"\nResults saved to {args.out_dir / args.variant}/")


if __name__ == "__main__":
    main()
