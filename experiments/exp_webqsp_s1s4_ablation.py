#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
exp_webqsp_s1s4_ablation.py  — Suggested Experiment #2
========================================================
Complete WebQSP secondary perturbation ablation: build NSM and GraftNet
subgraphs for WebQSP S1-S4 (P2,P3,P5,P6) and run GNN-RAG inference.

Gap in the current paper:
  Appendix Table (tab:secondary) shows S1-S4 with ELQ seeds only for BOTH
  datasets (CWQ and WebQSP). CWQ ablation (NSM + GraftNet) is being added via
  exp_s1s4_subgraph_build.py + exp_s1s4_inference.py. This script adds the
  same for WebQSP, completing the 28-cell picture:
    7 ptypes × 2 datasets × 2 methods = 28 cells

Coverage note (ablation source restriction):
  nsm_gem / nsm_elq / graftnetv3_gem / graftnetv3_elq exist only for p4/p7
  under new_exp/gnn_input/. Running --all-pert-types with these sources silently
  returns {"skipped": True} for p1/p2/p3/p5/p6. This script builds the missing
  WebQSP cells explicitly and warns before each run.

Why WebQSP is different from CWQ:
  - WebQSP has 1-2 hop questions (CWQ has 2-4) → BFS is faster (cap=2000 vs 2500)
  - GEM seed match rate is lower for WebQSP (77.8% vs 97.5% CWQ) for some ptypes
  - Expected: similar collapse pattern but different absolute EM levels
  - WebQSP clean EM: 74.3% (much higher than CWQ 52.9%)

Expected findings:
  WebQSP S1-S4 with NSM/GraftNet likely shows similar collapse to CWQ (~0.5-2% EM),
  consistent with "all secondary types collapse regardless of subgraph variant."
  Exception: S3 (entity insertion) with GraftNet may partially recover on WebQSP
  due to simpler 1-2 hop structure (distractor entity easier to down-weight).

Prerequisite:
  - WebQSP p2/p3/p5/p6 nxhd/pkl files at data/new_seed_nxhd/ (seeded from gold_entity_map)
  - exp_s1s4_subgraph_build.py for CWQ should run first (verifies the pipeline)

Pipeline:
  Stage 1: Run exp_s1s4_subgraph_build.py --dataset webqsp (builds subgraphs)
  Stage 2: Run this script for inference (wraps e2_webqsp_gnn_rag.py)
  Stage 3: Report table → ablate.tex

Usage:
  # Full pipeline: build WebQSP subgraphs then run inference:
  python exp_webqsp_s1s4_ablation.py --ptype all --method both --seed-mode gem

  # Inference only (if build already ran via exp_s1s4_subgraph_build.py --dataset webqsp):
  python exp_webqsp_s1s4_ablation.py --ptype all --inference-only

  # Single cell:
  python exp_webqsp_s1s4_ablation.py --ptype p3 --method graftnetv3 --seed-mode gem

  # Dry run (check what would execute):
  python exp_webqsp_s1s4_ablation.py --ptype all --method both --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = REPO_ROOT
RESULTS_WSP  = REPO_ROOT / "tmlr" / "results" / "e2_webqsp_gnn_rag"
SCRIPTS_DIR  = Path(__file__).parent
E2_SCRIPT    = SCRIPTS_DIR / "e2_webqsp_gnn_rag.py"
BUILD_SCRIPT = SCRIPTS_DIR / "exp_s1s4_subgraph_build.py"
NEW_EXP      = REPO_ROOT / "new_exp"

S1S4_PTYPES  = ["p2", "p3", "p5", "p6"]
METHODS      = ["nsm", "graftnetv3"]
SEED_MODES   = ["gem"]   # ELQ seeds optional; gem is the primary ablation

PTYPE_LABELS = {
    "p2": "S1 — Relation paraphrase (WebQSP)",
    "p3": "S2 — Voice flip (WebQSP)",
    "p5": "S3 — Adversarial entity insertion (WebQSP)",
    "p6": "S4 — Mixed lexical noise (WebQSP)",
}
EL_SOURCE_MAP = {
    ("nsm",         "gem"): "nsm_gem",
    ("nsm",         "elq"): "nsm_elq",
    ("graftnetv3",  "gem"): "graftnetv3_gem",
    ("graftnetv3",  "elq"): "graftnetv3_elq",
}


# ══════════════════════════════════════════════════════════════════════════════
# Coverage check
# ══════════════════════════════════════════════════════════════════════════════

def check_subgraph_ready(ptype: str, method: str, seed_mode: str) -> bool:
    """Return True if the WebQSP subgraph directory exists and is non-empty."""
    el_source = EL_SOURCE_MAP[(method, seed_mode)]
    sg_file = NEW_EXP / "gnn_input" / f"webqsp_{ptype}_{el_source}" / "test_subgraph.json"
    if sg_file.exists():
        n = sum(1 for _ in open(sg_file))
        return n > 0
    return False


def print_coverage_table(ptypes: List[str], methods: List[str],
                          seed_modes: List[str]) -> None:
    """Print a readiness table: which cells are built vs missing."""
    print(f"\n  {'Cell':<30}  {'Subgraph ready?':>16}  {'Records':>8}")
    print(f"  {'-'*60}")
    for ptype in ptypes:
        for method in methods:
            for seed_mode in seed_modes:
                el_source = EL_SOURCE_MAP[(method, seed_mode)]
                tag = f"webqsp_{ptype}_{el_source}"
                sg_file = NEW_EXP / "gnn_input" / tag / "test_subgraph.json"
                if sg_file.exists():
                    n = sum(1 for _ in open(sg_file))
                    status = f"✓ ({n} records)"
                else:
                    status = "✗ MISSING — run build first"
                    n = 0
                print(f"  {tag:<30}  {status}")


# ══════════════════════════════════════════════════════════════════════════════
# Build stage (delegates to exp_s1s4_subgraph_build.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_build(ptype: str, method: str, seed_mode: str,
              dry_run: bool, workers: int, bfs_workers: int) -> bool:
    """Build WebQSP subgraph for one cell. Returns True on success."""
    el_source = EL_SOURCE_MAP[(method, seed_mode)]
    if check_subgraph_ready(ptype, method, seed_mode):
        sg = NEW_EXP / "gnn_input" / f"webqsp_{ptype}_{el_source}" / "test_subgraph.json"
        n = sum(1 for _ in open(sg))
        print(f"  [skip build] webqsp/{ptype}/{el_source}: {n} records exist")
        return True

    cmd = [
        sys.executable, str(BUILD_SCRIPT),
        "--ptype", ptype,
        "--dataset", "webqsp",
        "--method", method,
        "--seed-mode", seed_mode,
        "--workers", str(workers),
        "--bfs-workers", str(bfs_workers),
    ]
    if dry_run:
        print(f"  [DRY RUN build] {' '.join(cmd)}")
        return True

    print(f"  Building: webqsp/{ptype}/{el_source} ...")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        print(f"  Build done in {elapsed/60:.1f}min")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Build FAILED: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Inference stage (delegates to e2_webqsp_gnn_rag.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(ptype: str, el_source: str, force: bool, dry_run: bool) -> bool:
    """Run GNN-RAG WebQSP inference for one (ptype, el_source) cell."""
    out = RESULTS_WSP / f"full_{el_source}" / f"{ptype}_per_question.json"
    if out.exists() and not force:
        n = sum(1 for _ in open(out))
        print(f"  [skip infer] {el_source}/{ptype}: {n} records exist")
        return True

    # Verify subgraph exists before launching GPU inference
    method, seed_mode = el_source.rsplit("_", 1)
    if not check_subgraph_ready(ptype, method, seed_mode):
        print(f"  [BLOCK infer] {el_source}/{ptype}: subgraph not built — run build stage first")
        return False

    cmd = [
        sys.executable, str(E2_SCRIPT),
        "--pert-type", ptype,
        "--el-source", el_source,
        "--mode", "full",
    ]
    if force:
        cmd.append("--force")
    if dry_run:
        print(f"  [DRY RUN infer] {' '.join(cmd)}")
        return True

    print(f"  Inferring: webqsp/{ptype}/{el_source} ...")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        print(f"  Inference done in {elapsed/60:.1f}min → {out}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Inference FAILED: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def load_em_results(ptypes: List[str], el_sources: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load EM results for all (ptype × el_source) cells. Returns nested dict."""
    results: Dict[str, Dict[str, Any]] = {}
    for el_source in el_sources:
        results[el_source] = {}
        for ptype in ptypes:
            pq_file = RESULTS_WSP / f"full_{el_source}" / f"{ptype}_per_question.json"
            if not pq_file.exists():
                results[el_source][ptype] = None
                continue
            with open(pq_file) as f:
                recs = json.load(f)
            em_pert  = [r["em_pert"]  for r in recs]
            em_clean = [r["em_clean"] for r in recs]
            results[el_source][ptype] = {
                "n": len(recs),
                "EM_clean":  round(100 * sum(em_clean) / len(em_clean), 2),
                "EM_pert":   round(100 * sum(em_pert)  / len(em_pert),  2),
                "delta_AG":  round(100 * (sum(em_clean) - sum(em_pert)) / len(em_clean), 2),
            }
    return results


def print_comparison_table(ptypes: List[str], el_sources: List[str],
                            results: Dict[str, Dict[str, Any]]) -> None:
    """Print LaTeX-ready comparison table across subgraph variants."""
    clean_wsp = 74.3
    print(f"\n{'='*80}")
    print(f"  WebQSP S1-S4 Subgraph Ablation  (Clean EM: {clean_wsp}%)")
    print(f"{'='*80}")

    # Header
    header = f"  {'Type':<22}  {'ELQ':>8}"
    for el in el_sources:
        header += f"  {el.replace('_','-'):>16}"
    print(header + "  (EM%)")
    print(f"  {'-'*76}")

    # ELQ baseline from existing full_elq results
    elq_results: Dict[str, Optional[float]] = {}
    for ptype in ptypes:
        pq_file = RESULTS_WSP / "full_elq" / f"{ptype}_per_question.json"
        if pq_file.exists():
            with open(pq_file) as f:
                recs = json.load(f)
            elq_results[ptype] = round(100 * sum(r["em_pert"] for r in recs) / len(recs), 2)
        else:
            elq_results[ptype] = None

    for ptype in ptypes:
        label = PTYPE_LABELS.get(ptype, ptype).split("(")[0].strip()
        elq_em = elq_results.get(ptype)
        row = f"  {label:<22}  {str(elq_em) if elq_em is not None else '—':>8}"
        for el in el_sources:
            cell = results.get(el, {}).get(ptype)
            em = f"{cell['EM_pert']:.2f}" if cell else "—"
            row += f"  {em:>16}"
        print(row)

    print(f"\n  Note: All values are EM% on full WebQSP test set (n=1639).")
    print(f"  ELQ baseline from results/e2_webqsp_gnn_rag/full_elq/.")
    print(f"\n  LaTeX snippet for ablate.tex (WebQSP column extension):")
    for ptype in ptypes:
        label = PTYPE_LABELS.get(ptype, ptype).split(" (")[0]
        vals = []
        for el in el_sources:
            cell = results.get(el, {}).get(ptype)
            vals.append(f"{cell['EM_pert']:.2f}" if cell else "--")
        print(f"  {label} & {' & '.join(vals)} \\\\")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="WebQSP S1-S4 subgraph ablation (Suggested Experiment #2)")
    p.add_argument("--ptype", default="all", choices=S1S4_PTYPES + ["all"],
                   help="Perturbation type (default: all = p2,p3,p5,p6)")
    p.add_argument("--method", default="both", choices=["nsm", "graftnetv3", "both"],
                   help="Subgraph method (default: both)")
    p.add_argument("--seed-mode", default="gem", choices=["gem", "elq", "both"],
                   help="Seed strategy (default: gem = GEM/SPARQL gold entity seeds)")
    p.add_argument("--inference-only", action="store_true",
                   help="Skip build stage (subgraphs already built via exp_s1s4_subgraph_build.py)")
    p.add_argument("--build-only", action="store_true",
                   help="Only run subgraph build, skip inference")
    p.add_argument("--report-only", action="store_true",
                   help="Print results table from existing files, no inference")
    p.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers",     type=int, default=8)
    p.add_argument("--bfs-workers", type=int, default=6)
    args = p.parse_args()

    ptypes     = S1S4_PTYPES if args.ptype == "all" else [args.ptype]
    methods    = METHODS if args.method == "both" else [args.method]
    seed_modes = SEED_MODES if args.seed_mode == "both" else [args.seed_mode]
    el_sources = [EL_SOURCE_MAP[(m, s)] for m in methods for s in seed_modes]

    print(f"WebQSP S1-S4 Ablation (Experiment #2)")
    print(f"  ptypes={ptypes}  methods={methods}  seed_modes={seed_modes}")

    # ── Coverage summary ────────────────────────────────────────────────────
    print(f"\nSubgraph readiness check:")
    print_coverage_table(ptypes, methods, seed_modes)

    if args.report_only:
        results = load_em_results(ptypes, el_sources)
        print_comparison_table(ptypes, el_sources, results)
        return

    failed_build, failed_infer = [], []

    # ── Build stage ─────────────────────────────────────────────────────────
    if not args.inference_only:
        print(f"\n{'='*60}")
        print(f"STAGE 1: Subgraph Build (WebQSP, {len(ptypes)*len(methods)*len(seed_modes)} cells)")
        print(f"Estimated: ~2-4h per cell (WebQSP BFS is faster than CWQ)")
        print(f"{'='*60}")
        for ptype in ptypes:
            for method in methods:
                for seed_mode in seed_modes:
                    el_source = EL_SOURCE_MAP[(method, seed_mode)]
                    print(f"\n  [BUILD] webqsp/{ptype}/{el_source}")
                    ok = run_build(ptype, method, seed_mode, args.dry_run,
                                   args.workers, args.bfs_workers)
                    if not ok:
                        failed_build.append(f"webqsp/{ptype}/{el_source}")

    # ── Inference stage ─────────────────────────────────────────────────────
    if not args.build_only:
        print(f"\n{'='*60}")
        print(f"STAGE 2: GNN-RAG Inference (WebQSP, {len(ptypes)*len(el_sources)} cells)")
        print(f"Estimated: ~20-40min per cell (WebQSP is smaller: 1639 questions)")
        print(f"{'='*60}")
        for ptype in ptypes:
            for el_source in el_sources:
                print(f"\n  [INFER] webqsp/{ptype}/{el_source}")
                ok = run_inference(ptype, el_source, args.force, args.dry_run)
                if not ok:
                    failed_infer.append(f"webqsp/{ptype}/{el_source}")

    # ── Results table ────────────────────────────────────────────────────────
    if not args.dry_run and not args.build_only:
        results = load_em_results(ptypes, el_sources)
        print_comparison_table(ptypes, el_sources, results)

        # Save JSON
        out = REPO_ROOT / "tmlr" / "results" / "webqsp_s1s4_ablation.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {out}")

    if failed_build:
        print(f"\n[WARN] Build failures: {failed_build}")
    if failed_infer:
        print(f"\n[WARN] Inference failures: {failed_infer}")

    print(f"\n[PAPER UPDATE]")
    print(f"  Update ablate.tex tab:secondary:")
    print(f"  → Add WebQSP columns for NSM and GraftNet variants")
    print(f"  → 28 total cells: 7 ptypes × 2 datasets × 2 methods")


if __name__ == "__main__":
    main()
