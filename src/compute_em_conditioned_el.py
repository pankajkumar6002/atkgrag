#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
compute_em_conditioned_el.py — Task 4 from To_do.md.

Compute downstream EM conditioned on EL success:
  - Full EM: all test questions
  - EL-correct EM: only questions where ELQ correctly linked all GT entities (all_gt=1)
  - EL-any EM: only questions where ELQ linked at least one GT entity (any_gt=1)

This isolates architecture behavior from EL noise.

Format needed (from To_do.md):
  Condition           GNN-RAG CWQ   GMT-KBQA CWQ
  P1, all questions      0.2%         18.3%
  P1, EL correct         ?            ?
  P1, oracle seed        ?            ?

Usage:
  python compute_em_conditioned_el.py --dataset cwq --p-type p1
  python compute_em_conditioned_el.py --all
  python compute_em_conditioned_el.py --dataset cwq --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = REPO_ROOT
TMLR    = REPO_ROOT / "tmlr"
RESULTS = TMLR / "results"

DATASET_DIR = TMLR / "pert" / "dataset"
EL_DIR      = DATASET_DIR / "el_elq"
GNN_DATA    = TMLR / "Repo" / "GNN-RAG" / "data"

GNN_RESULTS_BASE = RESULTS / "e1_cwq_gnn_rag"
GMT_RESULTS_BASE = RESULTS / "e1_cwq_gmt_kbqa"


def load_id2mid(dataset: str) -> Dict[int, str]:
    subdir = "CWQ" if dataset == "cwq" else "webqsp"
    with open(GNN_DATA / subdir / "entities.txt") as f:
        return {i: l.strip() for i, l in enumerate(f) if l.strip()}


# Map p-type → old perturbation name used in result file names
PTYPE_TO_FILENAME = {
    "p1": "adversarial_entity",
    "p2": "entity_alias_substitution",
    "p3": "active_passive_flip",
    "p4": "question_reordering",
    "p5": "relation_synonym",
    "p6": "multi_perturbation",   # if exists
    "orig": "orig",
}


def find_prediction_file(base_dir: Path, dataset: str, p_type: str) -> Optional[Path]:
    """Find the prediction output file for a dataset/p-type combo."""
    if not base_dir.exists():
        return None

    # Try mapped filename first (GMT-KBQA / old naming convention)
    old_name = PTYPE_TO_FILENAME.get(p_type, p_type)
    for subdir in [base_dir, base_dir / "greedy", base_dir / "elq",
                   base_dir / "agent", base_dir / "ela_v1"]:
        if not subdir.exists():
            continue
        for pat in [f"{old_name}_per_question.json",
                    f"*{p_type}*.json", f"*{old_name}*.json",
                    f"predictions_{p_type}.json"]:
            matches = list(subdir.glob(pat))
            if matches:
                return sorted(matches)[-1]
    return None


def load_predictions(pred_file: Path, dataset: str) -> Dict[str, Dict]:
    """Load model predictions file → {qid: {em_pert, em_clean, em, ...}}"""
    try:
        data = json.loads(pred_file.read_text())
        if isinstance(data, list):
            return {r.get("id") or r.get("qid", ""): r for r in data
                    if r.get("id") or r.get("qid")}
        elif isinstance(data, dict):
            if "results" in data:
                return {r.get("id",""): r for r in data["results"]}
            return data
    except Exception as e:
        print(f"  [WARN] Could not load {pred_file.name}: {e}")
    return {}


def get_em(rec: Dict, key: str = "em_pert") -> Optional[bool]:
    """Extract EM from a prediction record — handles multiple key conventions."""
    for k in [key, "em_pert", "em", "exact_match", "EM"]:
        v = rec.get(k)
        if v is not None:
            return bool(v)
    return None


def compute_em(predictions: Dict[str, Dict], question_subset: Set[str],
               em_key: str = "em_pert") -> Tuple[int, int, float]:
    """Compute EM over a subset of questions."""
    n = 0
    n_correct = 0
    for qid in question_subset:
        pred = predictions.get(qid)
        if pred is None:
            continue
        em = get_em(pred, em_key)
        if em is not None:
            n += 1
            if em:
                n_correct += 1
    pct = round(100 * n_correct / max(1, n), 2)
    return n_correct, n, pct


def get_el_subsets(
    el_records: List[Dict],
    orig_records: Dict[str, Dict],
    id2mid: Dict[int, str],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (any_gt_qids, all_gt_qids, no_gt_qids) based on EL correctness."""
    any_gt_qids: Set[str] = set()
    all_gt_qids: Set[str] = set()
    no_gt_qids:  Set[str] = set()

    el_by_id = {r["id"]: r for r in el_records}

    for qid, orig in orig_records.items():
        gt_mids: Set[str] = {id2mid[i] for i in orig.get("gnn_entities", []) if i in id2mid}
        if not gt_mids:
            continue
        linked = el_by_id.get(qid, {}).get("linked_entities", [])
        pred   = {e.get("mid", "") for e in linked if e.get("mid")}

        if gt_mids <= pred:
            all_gt_qids.add(qid)
        if pred & gt_mids:
            any_gt_qids.add(qid)
        else:
            no_gt_qids.add(qid)

    return any_gt_qids, all_gt_qids, no_gt_qids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["cwq", "webqsp"])
    ap.add_argument("--p-type",  choices=["orig", "p1", "p2", "p3", "p4", "p5", "p6"])
    ap.add_argument("--all",     action="store_true")
    ap.add_argument("--out-file", default=None)
    args = ap.parse_args()

    if args.all:
        if args.dataset:
            combos = [(args.dataset, pt)
                      for pt in ["orig", "p1", "p2", "p3", "p4", "p5", "p6"]]
        else:
            combos = [(ds, pt) for ds in ["cwq", "webqsp"]
                      for pt in ["orig", "p1", "p2", "p3", "p4", "p5", "p6"]]
    else:
        combos = [(args.dataset, args.p_type)]

    results = {}

    print("\n=== EM Conditioned on EL Success ===\n")
    print(f"  {'Key':<18} {'Model':<12} "
          f"{'EM_all%':>8} {'EM_any%':>8} {'EM_all_gt%':>11} "
          f"{'n_all':>7} {'n_any':>7} {'n_allgt':>8}")
    print("  " + "-" * 90)

    for dataset, p_type in combos:
        key = f"{dataset}_{p_type}"
        id2mid = load_id2mid(dataset)

        orig_path = DATASET_DIR / f"{dataset}_orig.json"
        if not orig_path.exists():
            continue
        orig_by_id = {r["id"]: r for r in json.load(open(orig_path))}

        # Load ELQ
        elq_path = EL_DIR / f"{dataset}_{p_type}_el.json"
        if not elq_path.exists():
            print(f"  {key}: ELQ file not found — skip")
            continue
        el_recs = json.load(open(elq_path))
        any_gt_qs, all_gt_qs, no_gt_qs = get_el_subsets(el_recs, orig_by_id, id2mid)
        all_qs = set(orig_by_id.keys())

        # Find GNN-RAG predictions
        gnn_base = RESULTS / f"e1_{dataset}_gnn_rag"
        if dataset == "webqsp":
            gnn_base = RESULTS / f"e2_{dataset}_gnn_rag"
        gnn_pred_file = find_prediction_file(gnn_base, dataset, p_type)
        if gnn_pred_file:
            gnn_preds = load_predictions(gnn_pred_file, dataset)
            if gnn_preds:
                nc_all, n_all, em_all = compute_em(gnn_preds, all_qs & set(gnn_preds))
                nc_any, n_any, em_any = compute_em(gnn_preds, any_gt_qs & set(gnn_preds))
                nc_agt, n_agt, em_agt = compute_em(gnn_preds, all_gt_qs & set(gnn_preds))
                print(f"  {key:<18} {'GNN-RAG':<12} "
                      f"{em_all:>7.1f}% {em_any:>7.1f}% {em_agt:>10.1f}% "
                      f"{n_all:>7} {n_any:>7} {n_agt:>8}")
                results.setdefault(key, {})["gnn_rag"] = {
                    "em_all_pct": em_all, "em_any_gt_pct": em_any, "em_all_gt_pct": em_agt,
                    "n_all": n_all, "n_any_gt": n_any, "n_all_gt": n_agt,
                }
            else:
                print(f"  {key:<18} {'GNN-RAG':<12}  [no predictions in file]")
        else:
            print(f"  {key:<18} {'GNN-RAG':<12}  [no pred file in {gnn_base.name}]")

        # Find GMT-KBQA predictions
        gmt_base = RESULTS / f"e1_{dataset}_gmt_kbqa"
        if dataset == "webqsp":
            gmt_base = RESULTS / f"e2_{dataset}_gmt_kbqa"
        gmt_pred_file = find_prediction_file(gmt_base, dataset, p_type)
        if gmt_pred_file:
            gmt_preds = load_predictions(gmt_pred_file, dataset)
            if gmt_preds:
                nc_all, n_all, em_all = compute_em(gmt_preds, all_qs & set(gmt_preds))
                nc_any, n_any, em_any = compute_em(gmt_preds, any_gt_qs & set(gmt_preds))
                nc_agt, n_agt, em_agt = compute_em(gmt_preds, all_gt_qs & set(gmt_preds))
                print(f"  {key:<18} {'GMT-KBQA':<12} "
                      f"{em_all:>7.1f}% {em_any:>7.1f}% {em_agt:>10.1f}% "
                      f"{n_all:>7} {n_any:>7} {n_agt:>8}")
                results.setdefault(key, {})["gmt_kbqa"] = {
                    "em_all_pct": em_all, "em_any_gt_pct": em_any, "em_all_gt_pct": em_agt,
                    "n_all": n_all, "n_any_gt": n_any, "n_all_gt": n_agt,
                }
            else:
                print(f"  {key:<18} {'GMT-KBQA':<12}  [no predictions in file]")
        else:
            print(f"  {key:<18} {'GMT-KBQA':<12}  [no pred file in {gmt_base.name}]")

    if results:
        out = Path(args.out_file) if args.out_file else (RESULTS / "em_conditioned_el.json")
        out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved → {out}")
    else:
        print("\n[No results — run GNN-RAG / GMT-KBQA on perturbed questions first]")
        print("See: tmlr/scripts/experiments/e1_cwq_gnn_rag.py --pert-type p1")


if __name__ == "__main__":
    main()