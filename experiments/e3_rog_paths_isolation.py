#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
e3_rog_paths_isolation.py — Condition B isolation experiment (RoG+paths).

Scientific question: Does the RoG+RA gain come from RA fine-tuning, or from
injecting relation paths at inference (which any model might benefit from)?

Three conditions:
  A — FineTune-RoG  + no paths   (existing cwq_FineTune-RoG_summary.json)
  B — FineTune-RoG  + pred paths  <-- THIS SCRIPT
  C — FineTune-RoG-RA + pred paths (existing cwq_FineTune-RoG-RA_summary.json)

Condition B uses the FineTune-RoG checkpoint (same as A) but injects the
same RoG predicted rule paths (predictions_3_False.jsonl) that condition C
uses at inference — without any RA fine-tuning signal.

Usage:
  # Sanity dry-run on full clean set (verify prompt format before perturbed):
  python e3_rog_paths_isolation.py --dataset cwq --mode dry-run

  # Perturbed p4 + p7 (CWQ):
  python e3_rog_paths_isolation.py --dataset cwq --pert-types p4 p7 --mode pert

  # Both clean + perturbed:
  python e3_rog_paths_isolation.py --dataset cwq --mode all
  python e3_rog_paths_isolation.py --dataset webqsp --mode all

  # Preview prompts for first N samples (no LLM inference, spot-check format):
  python e3_rog_paths_isolation.py --dataset cwq --mode dry-run --preview-n 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import shared infrastructure — avoids duplicating 700 lines of e3_llm_reasoning
sys.path.insert(0, str(Path(__file__).parent))
from e3_llm_reasoning import (
    REPO_ROOT,
    GNN_RAG_LLM,
    RESULTS_ROOT,
    ROG_PATH,
    HF_DATASET,
    CLEAN_TEST_INFO,
    load_gnn_candidates,
    load_test_info,
    load_entities_names,
    load_rule_paths,
    build_llm_records,
    run_llm_on_records,
    load_pert_dataset,
)

MODEL_LABEL = "FineTune-RoG-PathOnly"  # Condition B — distinct from A and C
MODEL_NAME = "rog"
MODEL_PATH = str(ROG_PATH) if ROG_PATH.exists() else "rmanluo/RoG"
PROMPT_PATH = GNN_RAG_LLM / "prompts" / "llama2_predict.txt"

# Expected clean hit@1 baselines for sanity reference
CLEAN_BASELINES = {
    "cwq":    {"cond_a": 56.36, "published_rog": 61.74},
    "webqsp": {"cond_a": None,  "published_rog": 80.59},
}


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _load_summary(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return json.load(f).get("results", {})
    return {}


def print_comparison(dataset: str, b_summary: Dict[str, Any]) -> None:
    a = _load_summary(RESULTS_ROOT / f"{dataset}_FineTune-RoG_summary.json")
    c = _load_summary(RESULTS_ROOT / f"{dataset}_FineTune-RoG-RA_summary.json")
    b = b_summary

    def fmt(d: dict, key: str) -> str:
        v = d.get(key, {}).get("hit1")
        return f"{v:.2f}%" if v is not None else "—"

    print(f"\n{'='*65}")
    print(f"ISOLATION COMPARISON — {dataset.upper()}")
    print(f"{'='*65}")
    print(f"  {'Condition':<30} {'clean':>8} {'p4':>8} {'p7':>8}")
    print(f"  {'-'*56}")
    for label, res in [
        ("A  FineTune-RoG (no paths)", a),
        ("B  FineTune-RoG-PathOnly  ← new", b),
        ("C  FineTune-RoG-RA (paths+ft)", c),
    ]:
        print(f"  {label:<30} {fmt(res,'clean'):>8} {fmt(res,'pert_p4'):>8} {fmt(res,'pert_p7'):>8}")
    print(f"{'='*65}")
    print("  B≈A → gain is purely from RA fine-tuning")
    print("  B≈C → path injection is the key mechanism")
    print("  A<B<C → both contribute\n")


# ---------------------------------------------------------------------------
# Prompt preview (format verification, no GPU needed)
# ---------------------------------------------------------------------------

def preview_prompts(records: List[Dict], n: int, dataset: str) -> None:
    """Build and print prompts for the first n records using the LLM src stack."""
    llm_src = str(GNN_RAG_LLM / "src")
    if llm_src not in sys.path:
        sys.path.insert(0, llm_src)
    orig_cwd = os.getcwd()
    os.chdir(str(GNN_RAG_LLM))

    try:
        from qa_prediction.build_qa_input import PromptBuilder
        builder = PromptBuilder(
            str(PROMPT_PATH),
            encrypt=False,
            add_rule=True,
            maximun_token=4096,
            tokenize=lambda x: len(x),
        )
        print(f"\n{'='*65}")
        print(f"PROMPT PREVIEW — first {n} records ({dataset})")
        print(f"{'='*65}")
        for i, rec in enumerate(records[:n]):
            try:
                prompt = builder.process_input(rec)
            except Exception as ex:
                prompt = f"[ERROR building prompt: {ex}]"
            paths_loaded = len(rec.get("predicted_paths", []))
            print(f"\n--- Record {i+1}  id={rec['id']} ---")
            print(f"  predicted_paths loaded: {paths_loaded}")
            print(f"  answer: {rec['answer']}")
            print(f"  PROMPT:\n{prompt}")
            print()
    finally:
        os.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset",    choices=["cwq", "webqsp"], default="cwq")
    parser.add_argument("--pert-types", nargs="+", default=["p4", "p7"])
    parser.add_argument("--mode",       choices=["dry-run", "pert", "all"], default="dry-run")
    parser.add_argument("--force",      action="store_true",
                        help="Overwrite existing predictions")
    parser.add_argument("--gpu",        type=str, default="",
                        help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="LLM batch size (0=auto: 8 for rog)")
    parser.add_argument("--preview-n",  type=int, default=0,
                        help="Print first N prompts and exit (no LLM inference)")
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataset = args.dataset

    print(f"\n{'='*65}")
    print(f"E3 Isolation — Condition B: {MODEL_LABEL}")
    print(f"  dataset   : {dataset}")
    print(f"  checkpoint: FineTune-RoG (rmanluo/RoG checkpoint)")
    print(f"  prompt    : RA format — predicted rule paths always injected")
    print(f"{'='*65}")

    # HF dataset
    print(f"\nLoading HF dataset: {HF_DATASET[dataset]}")
    from datasets import load_dataset as hf_load_dataset
    hf_ds = hf_load_dataset(HF_DATASET[dataset], split="test")
    print(f"  {len(hf_ds)} examples")

    # Entity name lookup
    entities_names = load_entities_names()
    print(f"  entities_names: {len(entities_names)} entries")

    # Rule paths (always required for Condition B)
    rule_path_file = (
        GNN_RAG_LLM / "results" / "gen_rule_path"
        / f"RoG-{dataset}" / "RoG" / "test" / "predictions_3_False.jsonl"
    )
    if not rule_path_file.exists():
        print(f"[ERROR] Rule path file not found: {rule_path_file}")
        sys.exit(1)
    rule_paths = load_rule_paths(rule_path_file)
    print(f"  rule paths loaded: {len(rule_paths)} records from predictions_3_False.jsonl")

    summary_path = RESULTS_ROOT / f"{dataset}_{MODEL_LABEL}_summary.json"
    summary: Dict[str, Any] = {"dataset": dataset, "model": MODEL_LABEL, "results": {}}
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                existing = json.load(f)
            summary["results"].update(existing.get("results", {}))
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # DRY-RUN: clean questions
    # -----------------------------------------------------------------------
    if args.mode in ("dry-run", "all"):
        print(f"\n--- DRY-RUN: clean questions (sanity check) ---")
        clean_info = CLEAN_TEST_INFO[dataset]
        clean_json = clean_info.parent / "test.json"

        if not clean_info.exists():
            print(f"[WARN] Clean test.info not found: {clean_info} — skipping dry-run")
        else:
            cands_map = load_test_info(clean_info, clean_json)
            print(f"  {len(cands_map)} GNN candidate sets loaded from clean test.info")

            records = build_llm_records(
                hf_dataset=hf_ds,
                pert_records={},
                cands_map=cands_map,
                entities_names=entities_names,
                use_perturbed_question=False,
                rule_paths=rule_paths,
            )

            # Prompt preview mode — exit before running GPU inference
            if args.preview_n > 0:
                preview_prompts(records, args.preview_n, dataset)
                print("[preview-n mode] Exiting without LLM inference.")
                return

            n_with_paths = sum(1 for r in records if r.get("predicted_paths"))
            print(f"  {n_with_paths}/{len(records)} records have rule paths")

            out_dir = RESULTS_ROOT / dataset / MODEL_LABEL / "clean"
            metrics = run_llm_on_records(
                records, MODEL_NAME, MODEL_PATH, PROMPT_PATH, out_dir,
                force=args.force, add_rule=True, batch_size=args.batch_size,
            )
            summary["results"]["clean"] = metrics

            # Sanity reference
            bl = CLEAN_BASELINES[dataset]
            print(f"\n  Clean hit@1 = {metrics['hit1']:.2f}%")
            if bl["cond_a"]:
                delta = metrics["hit1"] - bl["cond_a"]
                sign = "+" if delta >= 0 else ""
                print(f"  vs Condition A clean: {bl['cond_a']:.2f}%  (delta {sign}{delta:.2f}pp)")
                if delta < -1.0:
                    print("  [WARN] Large drop vs Condition A — check that rule paths are loading correctly.")
            print(f"  Published RoG clean baseline: {bl['published_rog']:.2f}%")

    # -----------------------------------------------------------------------
    # PERTURBED
    # -----------------------------------------------------------------------
    if args.mode in ("pert", "all"):
        for ptype in args.pert_types:
            print(f"\n--- PERTURBED: {dataset} {ptype} (Condition B) ---")

            pert_recs = load_pert_dataset(dataset, ptype)
            print(f"  {len(pert_recs)} perturbed questions")

            gnn_cand_dir = RESULTS_ROOT / dataset / "gnn_candidates" / f"pert_{ptype}"
            cands_file = gnn_cand_dir / "cands.jsonl"
            if not cands_file.exists() or cands_file.stat().st_size == 0:
                print(f"[ERROR] GNN candidates not found: {cands_file}")
                print(f"  Run e3_llm_reasoning.py --model rog --mode pert first to cache candidates.")
                continue

            cands_map = load_gnn_candidates(cands_file)
            n_with_cands = sum(1 for v in cands_map.values() if v)
            print(f"  {n_with_cands}/{len(cands_map)} questions have GNN candidates")

            records = build_llm_records(
                hf_dataset=hf_ds,
                pert_records=pert_recs,
                cands_map=cands_map,
                entities_names=entities_names,
                use_perturbed_question=True,
                rule_paths=rule_paths,
            )

            n_with_paths = sum(1 for r in records if r.get("predicted_paths"))
            print(f"  {n_with_paths}/{len(records)} records have rule paths")

            out_dir = RESULTS_ROOT / dataset / MODEL_LABEL / f"pert_{ptype}"
            metrics = run_llm_on_records(
                records, MODEL_NAME, MODEL_PATH, PROMPT_PATH, out_dir,
                force=args.force, add_rule=True, batch_size=args.batch_size,
            )
            summary["results"][f"pert_{ptype}"] = metrics

    # -----------------------------------------------------------------------
    # Save summary and print comparison
    # -----------------------------------------------------------------------
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    print_comparison(dataset, summary["results"])


if __name__ == "__main__":
    main()
