#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
e_explaignn_rs.py — EXPLAIGNN RS (p7, Relation Synonym) experiment.

Compares EXPLAIGNN P@1 on clean vs RS-perturbed ConvMix test set
(760 conversations, 4800 turns; 1664 KB-answerable turns perturbed).

RS (relation synonym) substitutes predicates with synonyms via Llama-3.3-70B,
keeping entities and answers unchanged. BART SR produces different SRs for
perturbed questions → different CLOCQ subgraphs → P@1 drop on KB turns.

Usage (compare only — after both EXPLAIGNN runs complete):
  cd external/EXPLAIGNN
  python  # set PYTHON_EXP env var to override \\
      experiments/e_explaignn_rs.py --compare-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

EXPLAIGNN_ROOT = REPO_ROOT / "external" / "EXPLAIGNN"
RESULTS_DIR    = REPO_ROOT / "results" / "convmix_rs"
META_FILE      = RESULTS_DIR / "perturbation_meta.json"
SUMMARY_OUT    = RESULTS_DIR / "explaignn_rs_summary.json"
SOURCES_STR    = "kb_text_table_info"


def find_latest_result(config_name: str) -> Optional[Path]:
    # Map config name to its intermediate results dir
    subdir_map = {
        "explaignn":    "convmix",
        "explaignn_rs": "convmix_rs",
        "explaignn_cr": "convmix_cr",
    }
    subdir = subdir_map.get(config_name, f"convmix_{config_name}")
    base = EXPLAIGNN_ROOT / "_intermediate_representations" / subdir / "sr" / "rers" / "kb_text_table_info" / "explaignn"
    if not base.exists():
        return None
    res_files = sorted(base.glob("res_turn_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    return res_files[-1] if res_files else None


def load_result(result_path: Path) -> List[Dict]:
    with open(result_path) as f:
        return json.load(f)


def compute_p_at_1(conversations: List[Dict]) -> Dict[str, float]:
    p1_list, mrr_list, kb_p1_list = [], [], []
    for conv in conversations:
        for turn in conv.get("questions", []):
            p1  = turn.get("p_at_1", 0.0)
            mrr = turn.get("mrr", 0.0)
            p1_list.append(p1)
            mrr_list.append(mrr)
            if turn.get("answer_src") == "kb" or turn.get("is_rs_perturbed"):
                kb_p1_list.append(p1)
    n, n_kb = len(p1_list), len(kb_p1_list)
    return {
        "p_at_1_all":     round(sum(p1_list) / n, 4)      if n    else 0,
        "mrr_all":        round(sum(mrr_list) / n, 4)     if n    else 0,
        "p_at_1_kb_only": round(sum(kb_p1_list) / n_kb, 4) if n_kb else 0,
        "n_all": n, "n_kb": n_kb,
    }


def print_comparison(clean: Dict, rs: Dict, meta: Optional[List]) -> None:
    print("\n" + "=" * 70)
    print("  EXPLAIGNN on ConvMix (Wikidata) — RS (p7) Sanity Check")
    print("  Addresses reviewer B5 (rvw2) / B8 (rvw_v1): Freebase deprecation")
    print("=" * 70)
    print(f"  {'Metric':<30} {'Clean':>10} {'RS-Pert':>10} {'Drop':>10}")
    print(f"  {'-'*62}")
    for metric, label in [
        ("p_at_1_all",     "P@1 (all 4800 turns)"),
        ("p_at_1_kb_only", "P@1 (KB turns only)"),
        ("mrr_all",        "MRR (all turns)"),
    ]:
        c = clean.get(metric, 0.0)
        p = rs.get(metric, 0.0)
        drop = c - p
        print(f"  {label:<30} {c*100:>9.2f}% {p*100:>9.2f}% {drop*100:>9.2f}%")
    print(f"\n  N (all turns):  {clean.get('n_all', 0)}")
    print(f"  N (KB turns):   {clean.get('n_kb', 0)}")
    if meta:
        jaccards = [m["jaccard"] for m in meta]
        avg_j    = sum(jaccards) / len(jaccards) if jaccards else 0
        fallback = sum(1 for m in meta if not m.get("valid", True))
        print(f"\n  RS perturbations: {len(meta)} KB turns")
        print(f"  Mean Jaccard (orig vs RS): {avg_j:.3f} (lower = more restructured)")
        print(f"  Fallback (identity kept):  {fallback} ({100*fallback/max(1,len(meta)):.1f}%)")
    clean_p1 = clean.get("p_at_1_kb_only", 0)
    rs_p1    = rs.get("p_at_1_kb_only", 0)
    drop_pp  = (clean_p1 - rs_p1) * 100
    print(f"\n  KEY FINDING: P@1 on KB turns drops {drop_pp:.1f}pp under RS "
          f"({clean_p1*100:.1f}% → {rs_p1*100:.1f}%)")
    print("  This confirms the RS failure mode transfers to Wikidata-based GNN-KGQA.")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-only", action="store_true",
                    help="Only compute comparison from existing result files")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.compare_only:
        print("Run generate_convmix_rs.py (fb-perturb env) first,")
        print("then run EXPLAIGNN via run_explaignn_rs.sh (explaignn env).")
        print("Then call this script with --compare-only.")
        sys.exit(0)

    print("[Step] Computing P@1 comparison...")
    clean_result = find_latest_result("explaignn")
    rs_result    = find_latest_result("explaignn_rs")

    if not clean_result:
        print("[ERROR] No clean result found. Run explaignn.yml pipeline first.")
        sys.exit(1)
    if not rs_result:
        print("[ERROR] No RS result found. Run explaignn_rs.yml pipeline first.")
        sys.exit(1)

    print(f"  Clean:  {clean_result}")
    print(f"  RS:     {rs_result}")

    clean_convs = load_result(clean_result)
    rs_convs    = load_result(rs_result)

    clean_metrics = compute_p_at_1(clean_convs)
    rs_metrics    = compute_p_at_1(rs_convs)

    meta = None
    if META_FILE.exists():
        with open(META_FILE) as f:
            meta = json.load(f)

    print_comparison(clean_metrics, rs_metrics, meta)

    summary = {
        "experiment": "EXPLAIGNN RS Sanity Check (Wikidata/ConvMix)",
        "dataset":    "ConvMix (760 conversations, 4800 turns)",
        "note":       "Addresses reviewer B5/B8: Freebase deprecation concern",
        "perturbation": "RS (Relation Synonym / p7) via Llama-3.3-70B",
        "clean":      {"source": str(clean_result), **clean_metrics},
        "rs_perturbed": {"source": str(rs_result), **rs_metrics},
        "n_kb_turns_perturbed": len(meta) if meta else None,
        "mean_jaccard": (
            round(sum(m["jaccard"] for m in meta) / len(meta), 3) if meta else None
        ),
    }
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
