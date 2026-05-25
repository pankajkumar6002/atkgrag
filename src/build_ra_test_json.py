#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_ra_test_json.py
Build test.json files alongside each pert test.info so that predict_answer.py
load_gnn_rag() can zip them correctly for RA runs.

test.json must:
  - have exactly the same number of lines as test.info
  - be in the same order (positional match)
  - contain: id, question (perturbed), q_entity, a_entity, graph, answer, choices

Strategy:
  1. Load HF dataset (rmanluo/RoG-webqsp or RoG-cwq) for graph/entity/answer fields
  2. Load perturbed questions from data/perturbations/{dataset}_{ptype}.json
  3. Read test.info to get the ordered list of IDs
  4. For each line in test.info, write a matching test.json line with perturbed question

For IDs in test.info that have no perturbed question (p1 partial coverage),
fall back to original question.
"""
import json
import sys
import argparse
from pathlib import Path

REPO_ROOT  = REPO_ROOT
PERT_DIR   = REPO_ROOT / "data" / "perturbations"
CANDS_ROOT = REPO_ROOT / "tmlr/results/e3_llm_reasoning"

def load_hf_dataset(dataset: str):
    from datasets import load_dataset
    hf_name = f"rmanluo/RoG-{dataset}" if dataset != "cwq" else "rmanluo/RoG-cwq"
    ds = load_dataset(hf_name, split="test")
    return {r["id"]: r for r in ds}

def load_pert_questions(dataset: str, ptype: str):
    path = PERT_DIR / f"{dataset}_{ptype}.json"
    with open(path) as f:
        records = json.load(f)
    return {r["id"]: r["perturbed_question"] for r in records}

def build_test_json(dataset: str, ptype: str, force: bool = False):
    info_path = CANDS_ROOT / dataset / "gnn_candidates" / f"pert_{ptype}" / "test.info"
    out_path  = info_path.parent / "test.json"

    if out_path.exists() and not force:
        print(f"  {out_path} already exists — skipping (use --force to overwrite)")
        return

    print(f"  Loading HF dataset rmanluo/RoG-{dataset}...")
    hf = load_hf_dataset(dataset)

    print(f"  Loading perturbed questions for {dataset} {ptype}...")
    pert_qs = load_pert_questions(dataset, ptype)

    # Read test.info to get ordered IDs
    with open(info_path) as f:
        info_lines = [json.loads(l) for l in f if l.strip()]

    print(f"  Building test.json ({len(info_lines)} lines)...")
    missing_hf = 0
    missing_pert = 0
    with open(out_path, "w") as fout:
        for obj in info_lines:
            qid = obj.get("id")
            if qid not in hf:
                # Fallback: write minimal record
                missing_hf += 1
                rec = {
                    "id": qid,
                    "question": pert_qs.get(qid, ""),
                    "q_entity": [],
                    "a_entity": [],
                    "graph": [],
                    "answer": [],
                    "choices": [],
                }
            else:
                rec = dict(hf[qid])
                # Replace question with perturbed version
                if qid in pert_qs:
                    rec["question"] = pert_qs[qid]
                else:
                    missing_pert += 1  # fall back to original question
            fout.write(json.dumps(rec) + "\n")

    print(f"  Written: {out_path}  (missing_hf={missing_hf}, fallback_to_orig_q={missing_pert})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["webqsp", "cwq"])
    parser.add_argument("--ptypes",   nargs="+", default=["p1", "p4", "p7"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for dataset in args.datasets:
        for ptype in args.ptypes:
            print(f"\n[{dataset} {ptype}]")
            try:
                build_test_json(dataset, ptype, args.force)
            except Exception as e:
                print(f"  ERROR: {e}")

if __name__ == "__main__":
    main()
