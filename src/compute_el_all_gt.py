#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
compute_el_all_gt.py — Add all_gt metric to el_linking_results.json.

For multi-hop questions (especially CWQ), a question may have 2+ GT seed entities.
any_gt counts questions where AT LEAST ONE GT seed was linked.
all_gt counts questions where ALL GT seeds were linked.

The gap between any_gt and all_gt reveals how often ELQ misses secondary entities
in multi-hop chains — a critical failure mode for compositional reasoning.

Updates el_linking_results.json in-place and prints a comparison table.

Usage:
  python compute_el_all_gt.py
  python compute_el_all_gt.py --dataset cwq --p-type p1   # single combo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

REPO_ROOT = REPO_ROOT
TMLR    = REPO_ROOT / "tmlr"
RESULTS = TMLR / "results"

DATASET_DIR = TMLR / "pert" / "dataset"
EL_DIR      = DATASET_DIR / "el_elq"
ELA_DIR     = DATASET_DIR / "el_agent"
GNN_DATA    = TMLR / "Repo" / "GNN-RAG" / "data"

EL_RESULTS_PATH = RESULTS / "el_linking_results.json"


def load_id2mid(dataset: str) -> Dict[int, str]:
    subdir = "CWQ" if dataset == "cwq" else "webqsp"
    with open(GNN_DATA / subdir / "entities.txt") as f:
        return {i: l.strip() for i, l in enumerate(f) if l.strip()}


def compute_all_gt_for_split(
    el_records: List[Dict],
    orig_records: Dict[str, Dict],
    id2mid: Dict[int, str],
    n_gt_threshold: int = 1,  # only multi-gt questions
) -> Dict:
    """Compute all_gt, any_gt, multi_hop_any_gt, multi_hop_all_gt stats.

    all_gt = fraction of questions where ALL GT MIDs appear in linked entities.
    Multi-hop subset = questions with gnn_entities count > 1.
    """
    n_total    = 0
    n_any_gt   = 0
    n_all_gt   = 0
    n_top1_gt  = 0

    # Multi-hop (multi-entity) stats
    n_multi    = 0
    n_multi_any  = 0
    n_multi_all  = 0
    n_multi_top1 = 0

    id2rec = {r["id"]: r for r in el_records}

    for qid, orig in orig_records.items():
        gt_indices = orig.get("gnn_entities", [])
        gt_mids: Set[str] = {id2mid[i] for i in gt_indices if i in id2mid}
        if not gt_mids:
            continue

        linked = id2rec.get(qid, {}).get("linked_entities", [])
        pred_mids = {e.get("mid", "") for e in linked if e.get("mid")}
        top1_mid  = None
        if linked:
            top = max(linked, key=lambda e: e.get("score", 0.0))
            top1_mid = top.get("mid") or None

        any_hit  = bool(pred_mids & gt_mids)
        all_hit  = bool(gt_mids and gt_mids <= pred_mids)
        top1_hit = bool(top1_mid and top1_mid in gt_mids)

        n_total  += 1
        if any_hit:  n_any_gt  += 1
        if all_hit:  n_all_gt  += 1
        if top1_hit: n_top1_gt += 1

        # Multi-hop subset (>1 GT seed)
        if len(gt_mids) > 1:
            n_multi += 1
            if any_hit:  n_multi_any  += 1
            if all_hit:  n_multi_all  += 1
            if top1_hit: n_multi_top1 += 1

    def pct(a, b): return round(100 * a / max(1, b), 2)

    return {
        "n_total":           n_total,
        "any_gt":            n_any_gt,
        "any_gt_pct":        pct(n_any_gt, n_total),
        "top1_gt":           n_top1_gt,
        "top1_gt_pct":       pct(n_top1_gt, n_total),
        "all_gt":            n_all_gt,
        "all_gt_pct":        pct(n_all_gt, n_total),
        "n_multi_hop":       n_multi,
        "multi_any_gt":      n_multi_any,
        "multi_any_gt_pct":  pct(n_multi_any, n_multi),
        "multi_all_gt":      n_multi_all,
        "multi_all_gt_pct":  pct(n_multi_all, n_multi),
        "multi_top1_gt":     n_multi_top1,
        "multi_top1_gt_pct": pct(n_multi_top1, n_multi),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["cwq", "webqsp"])
    ap.add_argument("--p-type", choices=["orig", "p1", "p2", "p3", "p4", "p5", "p6"])
    ap.add_argument("--no-update", action="store_true",
                    help="Print results only, do not update el_linking_results.json")
    args = ap.parse_args()

    # Determine combos
    datasets = ["cwq", "webqsp"] if not args.dataset else [args.dataset]
    p_types  = ["orig", "p1", "p2", "p3", "p4", "p5", "p6"] if not args.p_type else [args.p_type]

    # Load existing el_linking_results.json
    el_results: Dict = {}
    if EL_RESULTS_PATH.exists():
        el_results = json.load(open(EL_RESULTS_PATH))

    print("\n=== all_gt Metric Computation ===\n")
    print(f"{'DS':<8} {'PT':<5}  {'SRC':<7}  "
          f"{'any_gt%':>8} {'top1%':>6} {'all_gt%':>8}  "
          f"{'mAny%':>6} {'mAll%':>6} (multi-hop subset)")
    print("-" * 80)

    for dataset in datasets:
        id2mid = load_id2mid(dataset)
        orig_path = DATASET_DIR / f"{dataset}_orig.json"
        if not orig_path.exists():
            print(f"  [SKIP] {orig_path} not found")
            continue
        orig_by_id = {r["id"]: r for r in json.load(open(orig_path))}

        for p_type in p_types:
            for src, src_dir, src_key in [("elq_v2", EL_DIR, "elq_v2"),
                                           ("ela_v2", ELA_DIR, "ela_v2")]:
                el_path = src_dir / f"{dataset}_{p_type}_el.json"
                if not el_path.exists():
                    continue

                el_recs = json.load(open(el_path))
                stats   = compute_all_gt_for_split(el_recs, orig_by_id, id2mid)

                print(f"{dataset:<8} {p_type:<5}  {src:<7}  "
                      f"{stats['any_gt_pct']:>7.1f}% {stats['top1_gt_pct']:>5.1f}% "
                      f"{stats['all_gt_pct']:>7.1f}%  "
                      f"{stats.get('multi_any_gt_pct',0):>5.1f}% "
                      f"{stats.get('multi_all_gt_pct',0):>5.1f}%"
                      f"  (n_multi={stats['n_multi_hop']})")

                # Update el_linking_results.json in-place
                if not args.no_update and dataset in el_results and p_type in el_results[dataset]:
                    entry = el_results[dataset][p_type].setdefault(src_key, {})
                    entry.update({
                        "all_gt":            stats["all_gt"],
                        "all_gt_pct":        stats["all_gt_pct"],
                        "n_multi_hop":       stats["n_multi_hop"],
                        "multi_any_gt_pct":  stats["multi_any_gt_pct"],
                        "multi_all_gt_pct":  stats["multi_all_gt_pct"],
                        "multi_top1_gt_pct": stats["multi_top1_gt_pct"],
                    })

    if not args.no_update and EL_RESULTS_PATH.exists():
        EL_RESULTS_PATH.write_text(json.dumps(el_results, indent=2))
        print(f"\nUpdated → {EL_RESULTS_PATH}")
    else:
        print("\n[no-update mode — nothing written]")

    # Summary: multi-hop all_gt gap
    print("\n── all_gt vs any_gt gap (multi-hop, CWQ) ──")
    for p_type in ["orig", "p1", "p2", "p3", "p4", "p5", "p6"]:
        try:
            e = el_results.get("cwq", {}).get(p_type, {}).get("elq_v2", {})
            gap = e.get("multi_any_gt_pct", 0) - e.get("multi_all_gt_pct", 0)
            print(f"  cwq {p_type}: any={e.get('multi_any_gt_pct',0):.1f}%  "
                  f"all={e.get('multi_all_gt_pct',0):.1f}%  gap={gap:.1f}pp")
        except Exception:
            pass


if __name__ == "__main__":
    main()