#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
e3_llm_reasoning.py — LLM reasoning on top of GNN-RAG candidate outputs.

Pipeline:
  1. Re-run GNN-RAG inference on clean / perturbed questions → test.info (candidates)
  2. Build RoG-format dataset combining HF graph triples + perturbed question + GNN cands
  3. Run LLM (FineTune-RoG or Llama-8B) for final answer prediction
  4. Evaluate: Hit@1, F1, Accuracy (LLM-level text-name metrics)

Supports:
  --mode dry-run       : clean questions only, compare against published baseline
  --mode pert          : perturbed questions (p1/p4/p7) with saved GNN candidates
  --model rog          : RoG fine-tuned model (rmanluo/RoG)
  --model llama        : Llama-3.1-8B-Instruct (local)
  --dataset cwq|webqsp
  --pert-types p1 p4 p7  (space-separated, default: p1 p4 p7)
  --skip-gnn           : skip GNN re-inference, reuse existing saved test.info
  --dry-run-only       : run dry-run only (no perturbed)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import string
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = REPO_ROOT
GNN_RAG_GNN  = REPO_ROOT / "tmlr/Repo/GNN-RAG/gnn"
GNN_RAG_LLM  = REPO_ROOT / "tmlr/Repo/GNN-RAG/llm"
GNN_DATA     = REPO_ROOT / "external" / "GNN-RAG" / "data"
DATASET_DIR  = REPO_ROOT / "data" / "perturbations"
GNN_INPUT    = REPO_ROOT / "data" / "perturbations" / "gnn_input"
RESULTS_ROOT = REPO_ROOT / "tmlr/results/e3_llm_reasoning"
LLAMA_PATH   = REPO_ROOT / "tmlr/models/Llama-3.1-8B-Instruct"
ROG_PATH     = REPO_ROOT / "tmlr/Repo/reasoning-on-graphs/GCR-Llama-2-7b-chat-hf"  # GCR fine-tuned Llama-2-7B (rmanluo/GCR-Llama-2-7b-chat-hf)

CKPT = {
    "cwq":    GNN_DATA / "pretrain" / "ReaRev_CWQ.ckpt",
    "webqsp": GNN_DATA / "pretrain" / "ReaRev_webqsp.ckpt",
}
DATASET_SUBDIR = {"cwq": "CWQ", "webqsp": "webqsp"}
HF_DATASET = {"cwq": "rmanluo/RoG-cwq", "webqsp": "rmanluo/RoG-webqsp"}

# Existing clean test.info (from original GNN-RAG paper runs)
CLEAN_TEST_INFO = {
    "cwq":    GNN_RAG_LLM / "results/gnn/RoG-cwq/rearev-sbert/test.info",
    "webqsp": GNN_RAG_LLM / "results/gnn/RoG-webqsp/rearev-sbert/test.info",
}

# NOTE: GNN-RAG GNN and LLM src paths conflict on `utils` module.
# GNN path is added only inside the GNN worker subprocess (see run_gnn_worker.py).
# LLM src path is added only at LLM inference time (see run_llm_on_records).

# ---------------------------------------------------------------------------
# Metric helpers (same as evaluate_results.py)
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(c for c in s if c not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def match(s1: str, s2: str) -> bool:
    return normalize(s2) in normalize(s1)


def eval_hit1(prediction: List[str], answer: List[str]) -> float:
    if not prediction:
        return 0.0
    for a in answer:
        if match(prediction[0], a):
            return 1.0
    return 0.0


def eval_f1(prediction: List[str], answer: List[str]) -> Tuple[float, float, float]:
    if not prediction:
        return 0.0, 0.0, 0.0
    prediction_str = " ".join(prediction)
    matched = sum(1 for a in answer if match(prediction_str, a))
    p = matched / len(prediction)
    r = matched / len(answer) if answer else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, p, r


def eval_hit(prediction_str: str, answer: List[str]) -> float:
    return 1.0 if any(match(prediction_str, a) for a in answer) else 0.0


# ---------------------------------------------------------------------------
# Load GNN candidates from saved output file
# ---------------------------------------------------------------------------

def load_gnn_candidates(cands_file: Path) -> Dict[str, List]:
    """Load {qid: [cand_list]} from worker output file (one {id, cand} per line)."""
    result: Dict[str, List] = {}
    with open(cands_file) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                result[obj["id"]] = obj.get("cand", [])
    return result


def load_test_info(info_path: Path, test_json_path: Path) -> Dict[str, List]:
    """Load test.info and test.json together → {qid: [cand_list]}.
    Only works when test.json and test.info have same number of lines."""
    result: Dict[str, List] = {}
    with open(test_json_path) as fj, open(info_path) as fi:
        for line_j, line_i in zip(fj, fi):
            j = json.loads(line_j)
            i = json.loads(line_i)
            qid = j.get("id", "")
            result[qid] = i.get("cand", [])
    return result


# ---------------------------------------------------------------------------
# GNN inference — run in subprocess to avoid utils module conflict
# ---------------------------------------------------------------------------

GNN_WORKER_SCRIPT = Path(__file__).parent / "e3_gnn_worker.py"


def run_gnn_and_get_cands(
    records: List[Dict[str, Any]],
    dataset: str,
    save_info_path: Optional[Path] = None,
) -> Dict[str, List]:
    """Run GNN-RAG ReaRev via subprocess, return {qid: [(mid, score), ...]}."""
    import subprocess

    # Write records to a temp JSONL for the worker process
    tmp_input = Path(tempfile.mktemp(suffix="_e3_input.jsonl"))
    tmp_output = Path(tempfile.mktemp(suffix="_e3_cands.jsonl"))

    try:
        with open(tmp_input, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        cmd = [
            sys.executable,
            str(GNN_WORKER_SCRIPT),
            "--dataset", dataset,
            "--input", str(tmp_input),
            "--output", str(tmp_output),
        ]
        if save_info_path is not None:
            cmd += ["--save-info", str(save_info_path)]

        print(f"  [GNN worker] running: {' '.join(cmd[-6:])}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"  [GNN worker ERROR] exit code {result.returncode}")
            return {}

        cands_map: Dict[str, List] = {}
        if tmp_output.exists():
            with open(tmp_output) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        cands_map[obj["id"]] = obj.get("cand", [])
        return cands_map

    finally:
        for p in [tmp_input, tmp_output]:
            if p.exists():
                p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Load entities_names.json (mid → text name)
# ---------------------------------------------------------------------------

def load_entities_names() -> Dict[str, str]:
    path = GNN_RAG_LLM / "entities_names.json"
    with open(path) as f:
        mid2name = json.load(f)

    # Supplement with entities_sup_names.tsv (~16M entries) for better MID coverage
    sup_path = REPO_ROOT / "tmlr/resources/entities_sup_names.tsv"
    if sup_path.exists():
        with open(sup_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] not in mid2name:
                    mid2name[parts[0]] = parts[1]

    return mid2name


# ---------------------------------------------------------------------------
# Build combined record for LLM input
# ---------------------------------------------------------------------------

def load_rule_paths(jsonl_path: Path) -> Dict[str, List]:
    """Load {qid: predicted_paths} from gen_rule_path predictions_3_False.jsonl."""
    result: Dict[str, List] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                result[obj["id"]] = obj.get("prediction", [])
    return result


def build_llm_records(
    hf_dataset,
    pert_records: Dict[str, Dict],  # {qid: {perturbed_question, answers, gold_entity_map, ...}}
    cands_map: Dict[str, List],     # {qid: [(mid, score), ...]}
    entities_names: Dict[str, str],
    use_perturbed_question: bool = True,
    rule_paths: Optional[Dict[str, List]] = None,  # {qid: predicted_paths} for RA
) -> List[Dict[str, Any]]:
    """Build list of dicts ready for LLM prediction."""

    def resolve_answers(answers) -> List[str]:
        """Convert answers to text names; resolves MIDs via entities_names."""
        texts = []
        for a in answers:
            if isinstance(a, dict):
                # {"kb_id": "m.xxx", "text": "..."}
                text = a.get("text", "").strip()
                if not text:
                    mid = a.get("kb_id", "")
                    text = entities_names.get(mid, mid)
                texts.append(text)
            elif isinstance(a, str):
                if a.startswith("m.") or a.startswith("g."):
                    # MID — resolve to name
                    text = entities_names.get(a, a)
                else:
                    text = a
                texts.append(text)
        return [t for t in texts if t]

    records = []
    for row in hf_dataset:
        qid = row["id"]
        cands = cands_map.get(qid, [])

        # Resolve cand MIDs to names
        cand_names = []
        for c in cands:
            mid = c[0] if isinstance(c, (list, tuple)) else c
            name = entities_names.get(mid, mid)
            cand_names.append(name)

        # Use perturbed question if available, else original
        if use_perturbed_question and qid in pert_records:
            question = pert_records[qid].get("perturbed_question", row["question"])
            answers = pert_records[qid].get("answers", row["answer"])
            answer_texts = resolve_answers(answers) if answers else list(row["answer"])
        else:
            question = row["question"]
            answer_texts = list(row["answer"])  # HF has text answers already

        rec = {
            "id": qid,
            "question": question,
            "answer": answer_texts,
            "q_entity": row.get("q_entity", []),
            "graph": row.get("graph", ""),
            "choices": row.get("choices", []),
            "cand": cand_names if cand_names else None,
            "predicted_paths": rule_paths.get(qid, []) if rule_paths else [],
            "ground_paths": [],
        }
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# LLM inference (RoG or Llama)
# ---------------------------------------------------------------------------

def run_llm_on_records(
    records: List[Dict[str, Any]],
    model_name: str,
    model_path: str,
    prompt_path: Path,
    output_dir: Path,
    force: bool = False,
    add_rule: bool = False,
    batch_size: int = 0,
) -> Dict[str, Any]:
    """Run LLM prediction and evaluation. Returns summary metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / "predictions.jsonl"

    # Load existing predictions if resuming; truncate on force
    processed_ids: set = set()
    if pred_file.exists() and force:
        pred_file.unlink()  # truncate so --force doesn't append duplicates
    elif pred_file.exists():
        with open(pred_file) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    processed_ids.add(d["id"])
                except Exception:
                    pass

    # Add LLM src to path (only here, not at module level to avoid GNN utils conflict)
    llm_src = str(GNN_RAG_LLM / "src")
    if llm_src not in sys.path:
        sys.path.insert(0, llm_src)

    # Change to LLM dir so entities_names.json loads correctly
    orig_cwd = os.getcwd()
    os.chdir(str(GNN_RAG_LLM))

    # Load LLM
    from llms.language_models import get_registed_model
    LLMClass = get_registed_model(model_name)

    class _Args:
        pass

    args = _Args()
    args.model_path = model_path
    args.max_new_tokens = 128  # answer entity names are short
    args.dtype = "bf16"
    args.prompt_path = str(prompt_path)
    args.add_rule = add_rule
    args.use_true = False
    args.cot = False
    args.explain = False
    args.use_random = False
    args.each_line = False
    args.encrypt = False

    from qa_prediction.build_qa_input import PromptBuilder

    model = LLMClass(args)
    model.prepare_for_inference()
    input_builder = PromptBuilder(
        str(prompt_path),
        encrypt=False,
        add_rule=add_rule,
        maximun_token=model.maximun_token,
        tokenize=model.tokenize,
    )

    # Reduce fragmentation for long-context batches (RoG graph triples can be very long)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Build pending list
    if batch_size > 0:
        BATCH_SIZE = batch_size
    else:
        BATCH_SIZE = 8 if model_name == "rog" else 16

    pending = [d for d in records if d["id"] not in processed_ids]
    print(f"  Running LLM on {len(pending)}/{len(records)} records (batch_size={BATCH_SIZE})...")

    hit1_list, f1_list, prec_list, rec_list, hit_list = [], [], [], [], []

    with open(pred_file, "a") as fout:
        from tqdm import tqdm
        # Build all prompts first
        prompts_and_meta = []
        for data in pending:
            try:
                inp = input_builder.process_input(data)
            except Exception as ex:
                inp = data["question"]
            prompts_and_meta.append((data["id"], data["question"], data["answer"], inp))

        # Run batched inference
        for i in tqdm(range(0, len(prompts_and_meta), BATCH_SIZE)):
            batch = prompts_and_meta[i:i+BATCH_SIZE]
            batch_prompts = [b[3] for b in batch]
            try:
                # model.generator supports list input for batching
                model.generator.tokenizer.padding_side = "left"
                if not model.generator.tokenizer.pad_token_id:
                    model.generator.tokenizer.pad_token_id = model.generator.tokenizer.eos_token_id
                outputs = model.generator(
                    batch_prompts,
                    return_full_text=False,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=BATCH_SIZE,
                    pad_token_id=model.generator.tokenizer.eos_token_id,
                )
                pred_texts = [out[0]["generated_text"].strip() if out else "" for out in outputs]
            except Exception as ex:
                print(f"  [WARN] Batch {i//BATCH_SIZE} failed: {ex}")
                # OOM fallback: retry one-by-one at batch_size=1
                import torch
                torch.cuda.empty_cache()
                pred_texts = []
                for prompt in batch_prompts:
                    try:
                        out = model.generator(
                            [prompt],
                            return_full_text=False,
                            max_new_tokens=args.max_new_tokens,
                            batch_size=1,
                            pad_token_id=model.generator.tokenizer.eos_token_id,
                        )
                        pred_texts.append(out[0][0]["generated_text"].strip() if out and out[0] else "")
                    except Exception as ex2:
                        print(f"  [WARN] Single-item fallback failed: {ex2}")
                        pred_texts.append("")

            for (qid, question, answer, _), pred_text in zip(batch, pred_texts):
                result = {
                    "id": qid,
                    "question": question,
                    "prediction": pred_text,
                    "ground_truth": answer,
                }
                fout.write(json.dumps(result) + "\n")
            fout.flush()

    # Evaluate all predictions (resolve any residual MIDs in stored ground_truth)
    print(f"  Evaluating {pred_file}...")
    eval_entities_names = load_entities_names()

    def resolve_gt(answers: List[str]) -> List[str]:
        resolved = []
        for a in answers:
            if isinstance(a, str) and (a.startswith("m.") or a.startswith("g.")):
                resolved.append(eval_entities_names.get(a, a))
            else:
                resolved.append(str(a))
        return [x for x in resolved if x]

    eval_seen_ids: set = set()
    with open(pred_file) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            # Deduplicate by ID — prevents double-counting when a prior run appended
            if d["id"] in eval_seen_ids:
                continue
            eval_seen_ids.add(d["id"])
            pred_text = d.get("prediction", "").strip()
            answer = d.get("ground_truth", [])
            if not isinstance(answer, list):
                answer = [str(answer)]
            answer = resolve_gt(answer)

            # RoG outputs "Reasoning Path:\n...\nAnswer:\nX" — extract after Answer: marker
            # so Hit@1 checks the actual answer, not the reasoning header line.
            answer_match = re.search(r"Answer[s]?:\s*\n?(.*)", pred_text, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                preds = [p for p in answer_text.split("\n") if p.strip()] or [answer_text]
            else:
                preds = [p for p in pred_text.split("\n") if p.strip()]
                if not preds:
                    preds = [pred_text]

            h1 = eval_hit1(preds, answer)
            f1, p, r = eval_f1(preds, answer)
            hit = eval_hit(" ".join(preds), answer)
            hit1_list.append(h1)
            f1_list.append(f1)
            prec_list.append(p)
            rec_list.append(r)
            hit_list.append(hit)

    n = len(hit1_list)
    metrics = {
        "n": n,
        "hit1": sum(hit1_list) / n * 100 if n else 0.0,
        "f1":   sum(f1_list)   / n * 100 if n else 0.0,
        "precision": sum(prec_list) / n * 100 if n else 0.0,
        "recall":    sum(rec_list)  / n * 100 if n else 0.0,
        "hit":  sum(hit_list)  / n * 100 if n else 0.0,
    }
    print(f"  n={n}  Hit@1={metrics['hit1']:.2f}%  F1={metrics['f1']:.2f}%  Hit={metrics['hit']:.2f}%")

    os.chdir(orig_cwd)

    # Save eval summary
    with open(output_dir / "eval_result.txt", "w") as f:
        f.write(f"Hit@1: {metrics['hit1']:.2f}%  F1: {metrics['f1']:.2f}%  Hit: {metrics['hit']:.2f}%  "
                f"Precision: {metrics['precision']:.2f}%  Recall: {metrics['recall']:.2f}%  n={n}\n")

    return metrics


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def load_pert_dataset(dataset: str, ptype: str) -> Dict[str, Dict]:
    """Load cwq/webqsp perturbation dataset for a p-type → {qid: record}."""
    path = DATASET_DIR / f"{dataset}_{ptype}.json"
    with open(path) as f:
        records = json.load(f)
    return {r["id"]: r for r in records}


def load_gnn_subgraph_records(dataset: str, ptype: str, subgraph_dir: Optional[Path] = None) -> List[Dict]:
    """Load test_subgraph.json for GNN-RAG inference."""
    if subgraph_dir is not None:
        path = subgraph_dir / f"{dataset}_{ptype}_graftnetv3" / "test_subgraph.json"
    else:
        path = GNN_INPUT / f"{dataset}_{ptype}" / "test_subgraph.json"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset",     choices=["cwq", "webqsp"], default="cwq")
    parser.add_argument("--model",       choices=["rog", "llama"],  default="rog",
                        help="rog = FineTune-RoG (rmanluo/RoG); llama = Llama-3.1-8B-Instruct")
    parser.add_argument("--pert-types",  nargs="+", default=["p1", "p4", "p7"],
                        help="Perturbation types to evaluate (space-separated)")
    parser.add_argument("--mode",        choices=["dry-run", "pert", "all"], default="all",
                        help="dry-run=clean only, pert=perturbed only, all=both")
    parser.add_argument("--skip-gnn",    action="store_true",
                        help="(deprecated) GNN skip now auto-detected from saved cands.jsonl")
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite existing predictions")
    parser.add_argument("--gpu",         type=str, default="",
                        help="CUDA_VISIBLE_DEVICES (set before calling this script or pass here)")
    parser.add_argument("--add-rule",    action="store_true",
                        help="RA mode: inject RoG rule paths (predicted_paths) into prompt alongside GNN cands")
    parser.add_argument("--batch-size",  type=int, default=0,
                        help="Batch size for LLM inference (0=auto: 8 for rog, 16 for llama)")
    parser.add_argument("--subgraph-dir", type=Path, default=None,
                        help="Override GNN input dir (default: data/perturbations/gnn_input/{ds}_{ptype}). "
                             "Use for fixed/alternate subgraph builds, e.g. subgraph_var/gnn_input")
    parser.add_argument("--cand-tag",    type=str, default="gnn_candidates",
                        help="Tag for GNN candidates subdir under RESULTS_ROOT/{dataset}/. "
                             "Use a unique tag (e.g. gnn_candidates_fixed) to avoid overwriting existing cands.")
    parser.add_argument("--out-tag",     type=str, default=None,
                        help="Tag appended to model_label_ra for output dirs (e.g. '_fixed'). "
                             "Keeps fixed-subgraph results separate from originals.")
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cand_tag = args.cand_tag  # e.g. "gnn_candidates" or "gnn_candidates_fixed"
    out_tag = args.out_tag or ""  # e.g. "_fixed" or ""

    dataset = args.dataset
    model_name = args.model
    if model_name == "rog":
        model_path = str(ROG_PATH) if ROG_PATH.exists() else "rmanluo/RoG"
    else:
        model_path = str(LLAMA_PATH)
    model_label = "FineTune-RoG" if model_name == "rog" else "Llama-3.1-8B"
    # Use Llama-3 chat format for llama model; llama2 format for rog
    if model_name == "llama":
        prompt_path = GNN_RAG_LLM / "prompts" / "llama3_predict.txt"
    else:
        prompt_path = GNN_RAG_LLM / "prompts" / "llama2_predict.txt"

    print(f"\n{'='*60}")
    print(f"E3 LLM Reasoning: dataset={dataset}  model={model_label}")
    print(f"{'='*60}")

    # Load HF dataset
    print(f"\nLoading HF dataset: {HF_DATASET[dataset]}")
    from datasets import load_dataset as hf_load_dataset
    hf_ds = hf_load_dataset(HF_DATASET[dataset], split="test")
    print(f"  {len(hf_ds)} examples")

    # Load entities_names
    entities_names = load_entities_names()
    print(f"  entities_names: {len(entities_names)} entries")

    summary: Dict[str, Any] = {"dataset": dataset, "model": model_label, "results": {}}

    # Load clean RoG rule paths for RA (reused for all pert types)
    rule_paths: Optional[Dict[str, List]] = None
    if args.add_rule:
        rule_path_file = GNN_RAG_LLM / "results" / "gen_rule_path" / f"RoG-{dataset}" / "RoG" / "test" / "predictions_3_False.jsonl"
        if rule_path_file.exists():
            rule_paths = load_rule_paths(rule_path_file)
            print(f"  Loaded {len(rule_paths)} RoG rule paths from {rule_path_file}")
        else:
            print(f"  [WARN] Rule path file not found: {rule_path_file} — RA disabled")
            args.add_rule = False

    ra_suffix = "-RA" if args.add_rule else ""
    model_label_ra = model_label + ra_suffix + out_tag

    # ------------------------------------------------------------------
    # DRY-RUN: clean questions, existing test.info
    # ------------------------------------------------------------------
    if args.mode in ("dry-run", "all"):
        print(f"\n--- DRY RUN (clean questions, existing test.info) ---")
        clean_info = CLEAN_TEST_INFO[dataset]
        clean_json = clean_info.parent / "test.json"

        if not clean_info.exists():
            print(f"[WARN] Clean test.info not found: {clean_info}")
        else:
            cands_map = load_test_info(clean_info, clean_json)
            print(f"  Loaded {len(cands_map)} GNN candidate sets from clean test.info")

            records = build_llm_records(
                hf_dataset=hf_ds,
                pert_records={},
                cands_map=cands_map,
                entities_names=entities_names,
                use_perturbed_question=False,
                rule_paths=rule_paths,
            )
            out_dir = RESULTS_ROOT / dataset / model_label_ra / "clean"
            metrics = run_llm_on_records(
                records, model_name, model_path, prompt_path, out_dir,
                force=args.force, add_rule=args.add_rule, batch_size=args.batch_size,
            )
            summary["results"]["clean"] = metrics
            print(f"\nDry-run result ({model_label}, {dataset}, clean):")
            print(f"  Hit@1={metrics['hit1']:.2f}%  F1={metrics['f1']:.2f}%")
            if dataset == "cwq":
                print(f"  Baseline (published): Hit1=61.74%, F1=59.43%")
            else:
                print(f"  Baseline (published): Hit1=80.59%, F1=71.28%")

    # ------------------------------------------------------------------
    # PERTURBED: re-run GNN inference, then LLM
    # ------------------------------------------------------------------
    if args.mode in ("pert", "all"):
        for ptype in args.pert_types:
            print(f"\n--- PERTURBED: {dataset} {ptype} ({model_label}) ---")

            # Load perturbed dataset
            pert_recs = load_pert_dataset(dataset, ptype)
            print(f"  {len(pert_recs)} perturbed questions")

            # GNN candidates saved in model-agnostic location (shared across model runs)
            gnn_cand_dir = RESULTS_ROOT / dataset / cand_tag / f"pert_{ptype}"
            cands_file = gnn_cand_dir / "cands.jsonl"  # {id, cand} per line, all records

            if cands_file.exists() and cands_file.stat().st_size > 0:
                print(f"  Reusing saved GNN candidates: {cands_file}")
                cands_map = load_gnn_candidates(cands_file)
            else:
                print(f"  Loading GNN subgraph records for {dataset} {ptype}...")
                sg_records = load_gnn_subgraph_records(dataset, ptype, args.subgraph_dir)
                print(f"  {len(sg_records)} subgraph records → running GNN inference...")
                gnn_cand_dir.mkdir(parents=True, exist_ok=True)
                cands_map = run_gnn_and_get_cands(
                    sg_records,
                    dataset,
                    save_info_path=gnn_cand_dir / "test.info",
                )
                # Save cands.jsonl for future reuse
                with open(cands_file, "w") as f:
                    for qid, cand in cands_map.items():
                        f.write(json.dumps({"id": qid, "cand": cand}) + "\n")

            n_with_cands = sum(1 for v in cands_map.values() if v)
            print(f"  {n_with_cands}/{len(cands_map)} questions have GNN candidates")

            # Build LLM records with perturbed questions
            records = build_llm_records(
                hf_dataset=hf_ds,
                pert_records=pert_recs,
                cands_map=cands_map,
                entities_names=entities_names,
                use_perturbed_question=True,
                rule_paths=rule_paths,
            )

            out_dir = RESULTS_ROOT / dataset / model_label_ra / f"pert_{ptype}"
            metrics = run_llm_on_records(
                records, model_name, model_path, prompt_path, out_dir,
                force=args.force, add_rule=args.add_rule, batch_size=args.batch_size,
            )
            summary["results"][f"pert_{ptype}"] = metrics
            print(f"\nPerturbed {ptype} ({model_label}, {dataset}):")
            print(f"  Hit@1={metrics['hit1']:.2f}%  F1={metrics['f1']:.2f}%")

    # Save summary
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_ROOT / f"{dataset}_{model_label_ra}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    print(f"\n{'='*60}")
    print(f"E3 COMPLETE — {dataset} / {model_label}")
    for k, v in summary["results"].items():
        print(f"  {k}: Hit@1={v['hit1']:.2f}%  F1={v['f1']:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()