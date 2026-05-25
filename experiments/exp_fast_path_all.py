#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
exp_fast_path_all.py  (EXP-4 from review.md + fills Table 5 gaps)
===================================================================
Fast-path gold-MID EM for ALL perturbation types, both datasets.

What this does:
  "Fast mode" = original CLEAN subgraph + PERTURBED question text only.
  Measures GNN instruction decoder sensitivity in isolation from subgraph failure.
  Equivalent to Table 5 in the paper (tab:rearev_hits), but reports gold-MID EM
  (substring match) instead of entity-name Hit@1.

Current Table 5 status in the paper:
  - CR and RS: gold-MID EM_fast (clean subgraph, perturbed question) ✓
  - S2 (P3, voice flip): entity-name Hit@1 only — gold-MID EM_fast MISSING ✗
  - S3 (P5, entity insertion): entity-name Hit@1 only — gold-MID EM_fast MISSING ✗
  - S1 (P2, relation paraphrase): completely MISSING from Table 5 ✗
  - S4 (P6, mixed noise): completely MISSING from Table 5 ✗

This script runs all 7 perturbation types in fast mode to produce a complete
gold-MID EM_fast table for both datasets.

EXP-4 component (perturbed-only EM):
  Filter questions where is_perturbed=True (excludes ~13% fallback originals)
  and report EM on this "genuinely perturbed" subset only. This addresses
  reviewer B8: "perturbed-only EM understates severity by ~13% fallback fraction."

Output:
  results/fast_path_em_cwq.json   — gold-MID EM_fast for all ptypes, CWQ
  results/fast_path_em_webqsp.json — gold-MID EM_fast for all ptypes, WebQSP

LaTeX targets:
  Table 5 (tab:rearev_hits): add S1 and S4 rows with gold-MID EM_fast
  sections/limitations.tex: "Perturbed-only EM (excluding fallback) for CR ELQ is X%"

Usage:
  python exp_fast_path_all.py --dataset both --ptype all      # all ptypes
  python exp_fast_path_all.py --dataset cwq  --ptype p2 p6   # S1, S4 only
  python exp_fast_path_all.py --dataset cwq  --ptype all --exp4  # include perturbed-only
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
RESULTS     = REPO_ROOT / "tmlr" / "results"
PERT_DIR    = REPO_ROOT / "tmlr" / "pert" / "dataset"
SCRIPTS_DIR = Path(__file__).parent
E1_SCRIPT   = SCRIPTS_DIR / "e1_cwq_gnn_rag.py"
E2_SCRIPT   = SCRIPTS_DIR / "e2_webqsp_gnn_rag.py"

ALL_PTYPES = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
PTYPE_LABELS = {
    "p1": "ES  — Entity alias subst.",
    "p2": "S1  — Relation paraphrase",
    "p3": "S2  — Voice flip",
    "p4": "CR  — Comp. restructuring",
    "p5": "S3  — Entity insertion",
    "p6": "S4  — Mixed noise",
    "p7": "RS  — Relation synonym",
}


def run_fast_mode(script: Path, ptype: str, force: bool, dry_run: bool) -> Optional[Path]:
    """Run fast-path inference (clean subgraph + perturbed question). Returns per_question path."""
    ds = "cwq" if "cwq" in script.name else "webqsp"
    out_dir  = RESULTS / f"e{'1' if ds=='cwq' else '2'}_{ds}_gnn_rag" / "fast_elq"
    out_file = out_dir / f"{ptype}_per_question.json"

    if out_file.exists() and not force:
        n = sum(1 for _ in open(out_file))
        print(f"    [skip] fast/{ptype}: {n} records already exist")
        return out_file

    cmd = [sys.executable, str(script),
           "--pert-type", ptype, "--mode", "fast", "--el-source", "elq"]
    if force:
        cmd.append("--force")

    if dry_run:
        print(f"    [DRY RUN] {' '.join(cmd)}")
        return None

    print(f"    Running: python {script.name} --pert-type {ptype} --mode fast")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
        print(f"    Done in {(time.time()-t0)/60:.1f}min → {out_file}")
        return out_file
    except subprocess.CalledProcessError as e:
        print(f"    FAILED (exit {e.returncode})")
        return None


def compute_perturbed_only_em(per_question_file: Path, pert_file: Path) -> Optional[float]:
    """EXP-4: compute EM only on is_perturbed=True questions (excludes fallbacks).
    Returns (perturbed_only_em, n_perturbed, n_total)."""
    if not per_question_file.exists() or not pert_file.exists():
        return None, 0, 0

    with open(per_question_file) as f:
        pq = {r["id"]: r for r in json.load(f)}

    with open(pert_file) as f:
        pert_records = json.load(f)

    # Map qid → is_perturbed
    is_perturbed = {r["id"]: r.get("is_perturbed", True) for r in pert_records}

    em_perturbed_only = []
    for qid, rec in pq.items():
        if is_perturbed.get(qid, True):  # default True if not found
            em_perturbed_only.append(rec["em_pert"])

    if not em_perturbed_only:
        return None, 0, len(pq)

    n_pert = len(em_perturbed_only)
    em_val = 100 * sum(em_perturbed_only) / n_pert
    return em_val, n_pert, len(pq)


def load_fast_results(ds: str, ptypes: List[str]) -> Dict[str, Any]:
    """Load existing fast-path per_question files and compute EM statistics."""
    results = {}
    exp_num = "1" if ds == "cwq" else "2"
    base_dir = RESULTS / f"e{exp_num}_{ds}_gnn_rag" / "fast_elq"

    for ptype in ptypes:
        pq_file = base_dir / f"{ptype}_per_question.json"
        if not pq_file.exists():
            results[ptype] = {"status": "missing"}
            continue
        with open(pq_file) as f:
            records = json.load(f)
        em_pert  = [r["em_pert"] for r in records]
        em_clean = [r["em_clean"] for r in records]
        results[ptype] = {
            "n": len(records),
            "EM_clean": round(100 * sum(em_clean) / len(em_clean), 2),
            "EM_fast":  round(100 * sum(em_pert) / len(em_pert), 2),   # fast-path EM
            "delta_decoder": round(100 * (sum(em_clean) - sum(em_pert)) / len(em_clean), 2),
        }
    return results


def print_table5(ds: str, results: Dict[str, Any], ptypes: List[str],
                 run_exp4: bool, pert_dir: Path) -> None:
    """Print Table 5 format: fast-path EM_fast and Ins. cos. (decoder isolation)."""
    clean_em = {"cwq": 52.9, "webqsp": 74.3}[ds]
    print(f"\n{'='*70}")
    print(f"  Table 5 Fast-Path EM ({ds.upper()}) — GNN instruction decoder isolation")
    print(f"  Clean EM: {clean_em}% (full pipeline)")
    print(f"{'='*70}")
    print(f"  {'Type':<28}  {'EM_fast':>8}  {'Δ_decoder':>10}  {'Note':}")
    print(f"  {'-'*68}")

    for ptype in ptypes:
        r = results.get(ptype, {})
        label = PTYPE_LABELS.get(ptype, ptype)
        if r.get("status") == "missing":
            print(f"  {label:<28}  {'—':>8}  {'—':>10}  [NOT YET RUN]")
            continue
        em_fast = r["EM_fast"]
        delta = round(clean_em - em_fast, 2)  # drop vs. clean pipeline EM
        note = ""
        if ptype in ("p3", "p5"):
            note = "† (entity-name Hit@1 proxy in paper)"
        if ptype in ("p2", "p6"):
            note = "★ NEW (not in paper Table 5)"
        print(f"  {label:<28}  {em_fast:>7.2f}%  {delta:>+9.2f}pp  {note}")

        if run_exp4:
            pert_file = pert_dir / f"{ds}_{ptype}.json"
            pq_file = RESULTS / f"e{'1' if ds=='cwq' else '2'}_{ds}_gnn_rag" / "fast_elq" / f"{ptype}_per_question.json"
            em_pert_only, n_pert, n_total = compute_perturbed_only_em(pq_file, pert_file)
            if em_pert_only is not None:
                fallback_pct = 100 * (1 - n_pert / n_total) if n_total else 0
                print(f"    EXP-4: perturbed-only EM = {em_pert_only:.2f}% "
                      f"(n={n_pert}/{n_total}, {fallback_pct:.1f}% fallback excluded)")


def main() -> None:
    p = argparse.ArgumentParser(description="Fast-path gold-MID EM for all perturbation types")
    p.add_argument("--ptype", nargs="+", default=["p2", "p6"],
                   choices=ALL_PTYPES + ["all"],
                   help="Perturbation type(s) to run. Default: p2 p6 (S1, S4 — missing from Table 5)")
    p.add_argument("--dataset", default="both", choices=["cwq", "webqsp", "both"])
    p.add_argument("--exp4", action="store_true",
                   help="Report perturbed-only EM (EXP-4: exclude ~13%% fallback questions)")
    p.add_argument("--report-only", action="store_true",
                   help="Print existing results without running new inference")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ptypes = ALL_PTYPES if "all" in args.ptype else args.ptype
    datasets = ["cwq", "webqsp"] if args.dataset == "both" else [args.dataset]

    print(f"Fast-path EM: ptypes={ptypes} datasets={datasets}")
    if not args.report_only:
        print(f"Running inference in fast mode (clean subgraph + perturbed question)")

    for ds in datasets:
        script = E1_SCRIPT if ds == "cwq" else E2_SCRIPT
        print(f"\n{'='*70}")
        print(f"Dataset: {ds.upper()}")

        if not args.report_only:
            for ptype in ptypes:
                print(f"\n  [{ptype}] {PTYPE_LABELS.get(ptype, '')}:")
                run_fast_mode(script, ptype, args.force, args.dry_run)

        if not args.dry_run:
            results = load_fast_results(ds, ptypes)
            print_table5(ds, results, ptypes, args.exp4, PERT_DIR)

            # Save JSON
            out_file = RESULTS / f"fast_path_em_{ds}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved: {out_file}")

    print("\n[NEXT STEPS]")
    print("  1. Update Table 5 (tab:rearev_hits) with S1/S4 gold-MID EM_fast rows")
    print("  2. For EXP-4: use perturbed-only EM for sections/limitations.tex")
    print("  3. Note: EM_fast ≈ instruction decoder EM; gap vs clean EM = decoder sensitivity")


if __name__ == "__main__":
    main()