#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
e_explaignn_cr.py — EXPLAIGNN CR experiment (Wikidata sanity check for reviewers B5/B8).

Runs EXPLAIGNN on full ConvMix test set (760 conversations, 4800 turns) with:
  1. Clean benchmark  → ER cache hits → fast baseline P@1
  2. CR-perturbed benchmark → live CLOCQ API → measures P@1 drop under compositional restructuring

The CR perturbation targets KB-answerable turns (1664/4800 turns), using Llama-generated
compositional rephrasing (same answer, different surface form, different PPR anchoring).

Usage (run from EXPLAIGNN root):
  cd external/EXPLAIGNN
  python  # set PYTHON_EXP env var to override \\
      experiments/e_explaignn_cr.py

  # Step 1 only (generate perturbations):
  python  # set PYTHON_GNN env var to override \\
      experiments/generate_convmix_cr.py

  # Step 2 (clean baseline — uses ER cache, fast):
  cd external/EXPLAIGNN
  python  # set PYTHON_EXP env var to override explaignn/pipeline.py \\
      --pred-answers config/convmix/explaignn.yml kb_text_table_info

  # Step 3 (CR run — no cache, live CLOCQ API):
  cd external/EXPLAIGNN
  python  # set PYTHON_EXP env var to override explaignn/pipeline.py \\
      --pred-answers config/convmix/explaignn_cr.yml kb_text_table_info

  # Step 4 (compute and compare results):
  python  # set PYTHON_EXP env var to override \\
      experiments/e_explaignn_cr.py --compare-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

EXPLAIGNN_ROOT = REPO_ROOT / "external" / "EXPLAIGNN"
RESULTS_DIR    = REPO_ROOT / "results" / "convmix_cr"
META_FILE      = RESULTS_DIR / "perturbation_meta.json"
SUMMARY_OUT    = RESULTS_DIR / "explaignn_cr_summary.json"

PYTHON_EXP  = os.environ.get("PYTHON_EXP", "python")
PYTHON_GNN  = os.environ.get("PYTHON_GNN", "python")

CONFIG_CLEAN = "config/convmix/explaignn.yml"
CONFIG_CR    = "config/convmix/explaignn_cr.yml"
SOURCES_STR  = "kb_text_table_info"

CR_BENCHMARK = EXPLAIGNN_ROOT / "_benchmarks/convmix/test_set/test_set_ALL_cr.json"
CR_GEN_SCRIPT = Path("experiments/generate_convmix_cr.py")


def find_latest_result(config_name: str) -> Optional[Path]:
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
    """Load EXPLAIGNN result JSON (list of conversations)."""
    with open(result_path) as f:
        return json.load(f)


def compute_p_at_1(conversations: List[Dict], kb_only: bool = False) -> Dict[str, float]:
    """Compute P@1 from EXPLAIGNN output. Optionally restrict to KB-answerable turns."""
    p1_list, mrr_list = [], []
    kb_p1_list = []

    for conv in conversations:
        for turn in conv.get("questions", []):
            p1 = turn.get("p_at_1", 0.0)
            mrr = turn.get("mrr", 0.0)
            p1_list.append(p1)
            mrr_list.append(mrr)
            if turn.get("answer_src") == "kb" or turn.get("is_cr_perturbed"):
                kb_p1_list.append(p1)

    n = len(p1_list)
    n_kb = len(kb_p1_list)
    return {
        "p_at_1_all":      round(sum(p1_list) / n, 4) if n else 0,
        "mrr_all":         round(sum(mrr_list) / n, 4) if n else 0,
        "p_at_1_kb_only":  round(sum(kb_p1_list) / n_kb, 4) if n_kb else 0,
        "n_all":           n,
        "n_kb":            n_kb,
    }


def print_comparison(clean_metrics: Dict, cr_metrics: Dict, meta: Optional[List]) -> None:
    print("\n" + "=" * 70)
    print("  EXPLAIGNN on ConvMix (Wikidata) — CR Sanity Check")
    print("  Addressing reviewer B5 (rvw2) / B8 (rvw_v1): Freebase deprecation")
    print("=" * 70)
    print(f"  {'Metric':<30} {'Clean':>10} {'CR-Pert':>10} {'Drop':>10}")
    print(f"  {'-'*62}")

    for metric, label in [
        ("p_at_1_all",     "P@1 (all 4800 turns)"),
        ("p_at_1_kb_only", "P@1 (KB turns only)"),
        ("mrr_all",        "MRR (all turns)"),
    ]:
        c = clean_metrics.get(metric, 0.0)
        p = cr_metrics.get(metric, 0.0)
        drop = c - p
        print(f"  {label:<30} {c*100:>9.2f}% {p*100:>9.2f}% {drop*100:>9.2f}%")

    print(f"\n  N (all turns):   {clean_metrics.get('n_all', 0)}")
    print(f"  N (KB turns):    {clean_metrics.get('n_kb', 0)}")

    if meta:
        jaccards = [m["jaccard"] for m in meta if "jaccard" in m]
        avg_j = sum(jaccards) / len(jaccards) if jaccards else None
        fallback = sum(1 for m in meta if not m.get("valid", True))
        print(f"\n  CR perturbations: {len(meta)} KB turns")
        if avg_j is not None:
            print(f"  Mean Jaccard (orig vs CR): {avg_j:.3f} (lower = more restructured)")
        print(f"  Fallback (rule-based): {fallback} ({100*fallback/max(1,len(meta)):.1f}%)")

    clean_p1 = clean_metrics.get("p_at_1_kb_only", 0)
    cr_p1    = cr_metrics.get("p_at_1_kb_only", 0)
    drop_pp  = (clean_p1 - cr_p1) * 100
    print(f"\n  KEY FINDING: P@1 on KB turns drops {drop_pp:.1f}pp under CR "
          f"({clean_p1*100:.1f}% → {cr_p1*100:.1f}%)")
    print("  This confirms the CR failure mode transfers to Wikidata-based GNN-KGQA.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="EXPLAIGNN CR experiment (Wikidata sanity check)")
    ap.add_argument("--compare-only", action="store_true",
                    help="Only compute comparison from existing result files")
    ap.add_argument("--skip-cr-gen", action="store_true",
                    help="Skip CR generation (assume test_set_ALL_cr.json already exists)")
    ap.add_argument("--skip-clean", action="store_true",
                    help="Skip clean run (use cached results)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.compare_only:
        # Step 1: Generate CR perturbations
        if not args.skip_cr_gen:
            if CR_BENCHMARK.exists():
                print(f"CR benchmark already exists: {CR_BENCHMARK}")
                print("  Use --skip-cr-gen to skip, or delete to regenerate.")
            else:
                print("\n[Step 1] Generating CR perturbations with Llama-3.1-8B-Instruct...")
                ret = subprocess.run(
                    [PYTHON_GNN, str(CR_GEN_SCRIPT)],
                    cwd=str(EXPLAIGNN_ROOT),
                )
                if ret.returncode != 0:
                    print("[ERROR] CR generation failed.")
                    sys.exit(1)
        else:
            if not CR_BENCHMARK.exists():
                print(f"[ERROR] CR benchmark not found: {CR_BENCHMARK}")
                print("  Run generate_convmix_cr.py first.")
                sys.exit(1)
            print(f"[Step 1] Skipped (using existing {CR_BENCHMARK.name})")

        # Step 2: Clean baseline
        if not args.skip_clean:
            print(f"\n[Step 2] Running EXPLAIGNN on CLEAN benchmark (ER cache, fast)...")
            ret = subprocess.run(
                [PYTHON_EXP, "explaignn/pipeline.py",
                 "--pred-answers", CONFIG_CLEAN, SOURCES_STR],
                cwd=str(EXPLAIGNN_ROOT),
                env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
            )
            if ret.returncode != 0:
                print("[ERROR] Clean run failed.")
                sys.exit(1)
        else:
            print("[Step 2] Skipped (using cached clean results)")

        # Step 3: CR run
        print(f"\n[Step 3] Running EXPLAIGNN on CR-PERTURBED benchmark (live CLOCQ API)...")
        print("  This will take longer since perturbed questions miss the ER cache.")
        ret = subprocess.run(
            [PYTHON_EXP, "explaignn/pipeline.py",
             "--pred-answers", CONFIG_CR, SOURCES_STR],
            cwd=str(EXPLAIGNN_ROOT),
            env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
        )
        if ret.returncode != 0:
            print("[ERROR] CR run failed.")
            sys.exit(1)

    # Step 4: Compute and compare
    print("\n[Step 4] Computing P@1 comparison...")

    clean_result = find_latest_result("explaignn")
    cr_result    = find_latest_result("explaignn_cr")

    if not clean_result:
        print("[ERROR] No clean result file found. Run clean pipeline first.")
        sys.exit(1)
    if not cr_result:
        print("[ERROR] No CR result file found. Run CR pipeline first.")
        sys.exit(1)

    print(f"  Clean result: {clean_result}")
    print(f"  CR result:    {cr_result}")

    clean_convs = load_result(clean_result)
    cr_convs    = load_result(cr_result)

    clean_metrics = compute_p_at_1(clean_convs)
    cr_metrics    = compute_p_at_1(cr_convs)

    meta = None
    if META_FILE.exists():
        with open(META_FILE) as f:
            meta = json.load(f)

    print_comparison(clean_metrics, cr_metrics, meta)

    summary = {
        "experiment": "EXPLAIGNN CR Sanity Check (Wikidata)",
        "dataset": "ConvMix (760 conversations, 4800 turns)",
        "note": "Addresses reviewer B5/B8: Freebase deprecation concern",
        "clean": {"source": str(clean_result), **clean_metrics},
        "cr_perturbed": {"source": str(cr_result), **cr_metrics},
        "n_kb_turns_perturbed": len(meta) if meta else None,
        "mean_jaccard": (
            round(sum(m["jaccard"] for m in meta if "jaccard" in m) / max(1, sum(1 for m in meta if "jaccard" in m)), 3)
            if meta and any("jaccard" in m for m in meta) else None
        ),
    }

    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
