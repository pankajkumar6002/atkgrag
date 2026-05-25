#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
exp_s1s4_subgraph_build.py
===========================
Build NSM and GraftNetV3 subgraphs for secondary perturbation types
S1-S4 (P2, P3, P5, P6) using GEM seeds (gold_entity_map from SPARQL).

Extends build_ablation.py which hard-codes choices=["p1","p4","p7"] in argparse.
This script imports run_cell() directly, bypassing that restriction.

Paper context:
  S1=P2 (relation paraphrase), S2=P3 (voice flip),
  S3=P5 (adversarial entity insertion), S4=P6 (mixed noise).
  Table 7 (Appendix) currently shows only ELQ and AgentFresh for S1-S4.
  This script adds NSM and GraftNet variants to complete the subgraph ablation.

Output directories (written under new_exp/gnn_input/):
  {ds}_{ptype}_nsm_gem/       — NSM PPR, GEM seeds
  {ds}_{ptype}_graftnetv3_gem/ — GraftNetV3 PPR, GEM seeds
  {ds}_{ptype}_nsm_elq/       — NSM PPR, ELQ seeds (PPR-only ablation)
  {ds}_{ptype}_graftnetv3_elq/ — GraftNetV3 PPR, ELQ seeds

Usage:
  # All S1-S4 types, both datasets, both methods, GEM seeds only (recommended first run):
  python exp_s1s4_subgraph_build.py --ptype all --dataset both --method both --seed-mode gem

  # Single cell (fastest for testing):
  python exp_s1s4_subgraph_build.py --ptype p2 --dataset cwq --method nsm --seed-mode gem

  # Add ELQ-seed ablation for p3/p5/p6 (entity mentions intact → ELQ ≈ GEM):
  python exp_s1s4_subgraph_build.py --ptype p3 --dataset both --method both --seed-mode elq

  # Stage control (Stage 1 = BFS, 2 = PPR, 3 = GNN encoding):
  python exp_s1s4_subgraph_build.py --ptype p2 --dataset cwq --method graftnetv3 --seed-mode gem --stage 23

Runtime estimate (4× A100, 8 BFS workers):
  Per cell (one ptype × dataset × method × seed): ~2-6h (Stage 1 BFS dominates).
  All 8 gem cells (p2,p3,p5,p6 × cwq,webqsp × nsm,graftnetv3): ~24-48h total.
  ELQ seeds: faster (97%+ seed overlap with GEM → BFS reuse possible for p3/p5/p6).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Import run_cell from build_ablation (bypasses its argparse ptype restriction) ──
SCRIPTS_ROOT = Path(__file__).parent.parent   # tmlr/scripts/
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

try:
    from build_ablation import run_cell, _ablation_tag, REPO_ROOT, NEW_EXP
except ImportError as e:
    print(f"[ERROR] Cannot import from build_ablation.py: {e}")
    print(f"  Expected location: {SCRIPTS_ROOT / 'build_ablation.py'}")
    sys.exit(1)

# ── Secondary perturbation types only ──────────────────────────────────────────
S1S4_PTYPES = ["p2", "p3", "p5", "p6"]  # S1, S2, S3, S4

PTYPE_LABELS = {
    "p2": "S1 — Relation paraphrase",
    "p3": "S2 — Voice flip",
    "p5": "S3 — Adversarial entity insertion",
    "p6": "S4 — Mixed lexical noise",
}

log = logging.getLogger("s1s4_build")


def check_existing(ds: str, ptype: str, method: str, seed_mode: str) -> bool:
    """Return True if the output gnn_input already exists and is non-empty."""
    tag = _ablation_tag(method, seed_mode, ds, ptype)
    out = NEW_EXP / "gnn_input" / tag / "test_subgraph.json"
    if out.exists():
        n = sum(1 for _ in open(out))
        if n > 0:
            log.info(f"  [skip] {tag}: {n} records already in {out}")
            return True
    return False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = REPO_ROOT / "tmlr" / "logs" / f"s1s4_build_{ts}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    p = argparse.ArgumentParser(description="Build NSM/GraftNet subgraphs for S1-S4 (P2,P3,P5,P6)")
    p.add_argument("--ptype", default="all",
                   choices=S1S4_PTYPES + ["all"],
                   help="Perturbation type (default: all = p2,p3,p5,p6)")
    p.add_argument("--dataset", default="both", choices=["cwq", "webqsp", "both"])
    p.add_argument("--method", default="both", choices=["nsm", "graftnetv3", "both"])
    p.add_argument("--seed-mode", default="gem", choices=["gem", "elq", "both"],
                   help="gem = GEM/SPARQL gold entity seeds; elq = ELQ top-1 seeds")
    p.add_argument("--stage", default="123",
                   help="Stages to run: any combination of 1,2,3 (default: 123)")
    p.add_argument("--workers", type=int, default=8, help="Stage2 PPR parallel workers")
    p.add_argument("--bfs-workers", type=int, default=6,
                   help="Stage1 BFS workers (each loads fb_en.txt ~145GB; max=floor(RAM/145))")
    p.add_argument("--no-resume", action="store_true", help="Re-run even if output exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run without executing")
    args = p.parse_args()

    ptypes   = S1S4_PTYPES if args.ptype == "all" else [args.ptype]
    datasets = ["cwq", "webqsp"] if args.dataset == "both" else [args.dataset]
    methods  = ["nsm", "graftnetv3"] if args.method == "both" else [args.method]
    seeds    = ["gem", "elq"] if args.seed_mode == "both" else [args.seed_mode]
    stages   = list(args.stage)
    resume   = not args.no_resume

    log.info(f"S1-S4 subgraph build: ptypes={ptypes} datasets={datasets} "
             f"methods={methods} seeds={seeds} stages={stages}")
    log.info(f"Log: {log_file}")

    cells = [(m, s, d, p) for p in ptypes for d in datasets
             for m in methods for s in seeds]
    log.info(f"Total cells to run: {len(cells)}")

    t_global = time.time()
    completed, skipped, failed = [], [], []

    for method, seed_mode, ds, ptype in cells:
        tag = _ablation_tag(method, seed_mode, ds, ptype)
        label = PTYPE_LABELS.get(ptype, ptype)
        log.info(f"\n{'='*70}")
        log.info(f"Cell: {tag}  [{label}]")

        if resume and check_existing(ds, ptype, method, seed_mode):
            skipped.append(tag)
            continue

        if args.dry_run:
            log.info(f"  [DRY RUN] would call run_cell({method!r}, {seed_mode!r}, {ds!r}, {ptype!r}, {stages}, ...)")
            continue

        t0 = time.time()
        try:
            run_cell(method, seed_mode, ds, ptype, stages, resume, args.workers, args.bfs_workers)
            elapsed = time.time() - t0
            log.info(f"  Done in {elapsed/60:.1f}min")
            completed.append(tag)
        except Exception as e:
            log.error(f"  FAILED: {e}", exc_info=True)
            failed.append((tag, str(e)))

    log.info(f"\n{'='*70}")
    log.info(f"Build complete in {(time.time()-t_global)/3600:.2f}h")
    log.info(f"  Completed: {len(completed)}")
    log.info(f"  Skipped (already done): {len(skipped)}")
    log.info(f"  Failed: {len(failed)}")

    log.info("\nOutput summary:")
    for method, seed_mode, ds, ptype in cells:
        tag = _ablation_tag(method, seed_mode, ds, ptype)
        out = NEW_EXP / "gnn_input" / tag / "test_subgraph.json"
        if out.exists():
            n = sum(1 for _ in open(out))
            log.info(f"  {tag}: {n} records → {out}")
        else:
            log.info(f"  {tag}: MISSING — {out}")

    if failed:
        log.error("\nFailed cells:")
        for tag, err in failed:
            log.error(f"  {tag}: {err}")

    log.info(f"\nNext step: run exp_s1s4_inference.py to evaluate GNN-RAG EM")
    log.info("  Example: python exp_s1s4_inference.py --ptype all --el-source nsm_gem")


if __name__ == "__main__":
    main()
