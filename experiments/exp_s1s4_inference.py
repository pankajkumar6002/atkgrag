#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
exp_s1s4_inference.py
=====================
Run GNN-RAG (ReaRev) inference for secondary perturbation types S1-S4 (P2,P3,P5,P6)
across subgraph variants: ELQ (already done), NSM, GraftNetV3 (GEM and ELQ seeds).

Prerequisite: exp_s1s4_subgraph_build.py must have completed (for nsm_gem/graftnetv3_gem cells).
The ELQ results for S1-S4 already exist in results/e1_cwq_gnn_rag/full_elq/ and
results/e2_webqsp_gnn_rag/full_elq/.

What this adds:
  - NSM (GEM seeds): results for p2,p3,p5,p6 with NSM subgraph
  - GraftNetV3 (GEM seeds): results for p2,p3,p5,p6 with GraftNet subgraph
  - Optionally NSM/GraftNet with ELQ seeds for PPR-flavor-only ablation

Output: results/e1_cwq_gnn_rag/full_{el_source}/p{N}_per_question.json (CWQ)
        results/e2_webqsp_gnn_rag/full_{el_source}/p{N}_per_question.json (WebQSP)

LaTeX target: Appendix Table (S1-S4 ablation with NSM, GraftNet variants).
Paper claim to test: "All secondary types collapse under ELQ — does NSM or GraftNet recover?"

Expected results:
  S2 (P3, voice flip): NSM/GraftNet may slightly recover (~2-5% EM) since q-emb PPR
    adapts to active-passive switch. Still well below clean 52.9% → bottleneck is GNN decoder.
  S3 (P5, entity insertion): GraftNet should partially suppress the distractor entity since
    its q-emb cosine PPR de-weights edges not relevant to the (unchanged) question.
  S1 (P2, relation paraphrase): NSM (relation-weighted PPR) may help (improved relation matching).
  S4 (P6, mixed noise): likely no recovery (noise is lexical, no structure to exploit).

Usage:
  # All S1-S4, both datasets, nsm_gem and graftnetv3_gem:
  python exp_s1s4_inference.py --ptype all --dataset both --el-source both_gem

  # Single cell:
  python exp_s1s4_inference.py --ptype p2 --dataset cwq --el-source nsm_gem

  # Include ELQ-seed ablation:
  python exp_s1s4_inference.py --ptype p3 --dataset cwq --el-source all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = REPO_ROOT
RESULTS_CWQ = REPO_ROOT / "tmlr" / "results" / "e1_cwq_gnn_rag"
RESULTS_WSP = REPO_ROOT / "tmlr" / "results" / "e2_webqsp_gnn_rag"
SCRIPTS_DIR = Path(__file__).parent
E1_SCRIPT   = SCRIPTS_DIR / "e1_cwq_gnn_rag.py"
E2_SCRIPT   = SCRIPTS_DIR / "e2_webqsp_gnn_rag.py"

# Use GNN-RAG conda env if available (has PyTorch); fall back to sys.executable
_GNN_RAG_PY = Path(os.environ.get("PYTHON_GNN", "python"))
PYTHON_EXE = str(_GNN_RAG_PY) if _GNN_RAG_PY.exists() else sys.executable

S1S4_PTYPES = ["p2", "p3", "p5", "p6"]

# ── Ablation source coverage guard ─────────────────────────────────────────────
# nsm_elq / nsm_gem / graftnetv3_elq / graftnetv3_gem subgraph directories
# exist ONLY for p4 and p7 under new_exp/gnn_input/.
# Running --all-pert-types with these sources silently returns {"skipped": True}
# for p1/p2/p3/p5/p6 because the directories don't exist yet.
# exp_s1s4_subgraph_build.py produces them for p2/p3/p5/p6; they must complete
# BEFORE this script can run those ptypes with ablation sources.
ABLATION_SOURCES = {"nsm_gem", "nsm_elq", "graftnetv3_gem", "graftnetv3_elq"}
# p4 and p7 already built; p2/p3/p5/p6 require exp_s1s4_subgraph_build.py first
ABLATION_BUILT_PTYPES = {"p4", "p7"}   # pre-existing; do NOT add p2/p3/p5/p6 here

PTYPE_LABELS = {
    "p2": "S1/relation-paraphrase",
    "p3": "S2/voice-flip",
    "p5": "S3/entity-insertion",
    "p6": "S4/mixed-noise",
}
EL_SOURCES = {
    "both_gem": ["nsm_gem", "graftnetv3_gem"],
    "all":      ["nsm_gem", "graftnetv3_gem", "nsm_elq", "graftnetv3_elq"],
    "nsm_gem":         ["nsm_gem"],
    "graftnetv3_gem":  ["graftnetv3_gem"],
    "nsm_elq":         ["nsm_elq"],
    "graftnetv3_elq":  ["graftnetv3_elq"],
}


def check_ablation_source_ready(ptype: str, el_source: str) -> bool:
    """Guard: ablation sources only have pre-built subgraphs for p4/p7.
    For p2/p3/p5/p6, exp_s1s4_subgraph_build.py must complete first.
    Returns True if safe to run, False if the subgraph directory is missing."""
    if el_source not in ABLATION_SOURCES:
        return True   # ELQ / graftnet / nsm use PERT_DIR paths, always present
    if ptype in ABLATION_BUILT_PTYPES:
        return True   # p4, p7: pre-existing ablation cells

    ds = "cwq"  # checked via filesystem; webqsp handled symmetrically
    NEW_EXP = REPO_ROOT
    sg_dir = NEW_EXP / "gnn_input" / f"{ds}_{ptype}_{el_source}"
    sg_file = sg_dir / "test_subgraph.json"
    if sg_file.exists():
        return True   # exp_s1s4_subgraph_build.py already completed this cell

    print(f"  [BLOCK] {ptype}/{el_source}: subgraph NOT BUILT yet.")
    print(f"    Expected: {sg_file}")
    print(f"    Run first: python exp_s1s4_subgraph_build.py "
          f"--ptype {ptype} --dataset both --method {el_source.split('_')[0]} "
          f"--seed-mode {el_source.split('_')[1]}")
    return False


def run_inference_cell(script: Path, ptype: str, el_source: str,
                       force: bool, dry_run: bool) -> bool:
    """Run one (dataset × ptype × el_source) inference cell. Returns True on success."""
    # Guard: ablation sources only cover p4/p7 until exp_s1s4_subgraph_build completes
    if not check_ablation_source_ready(ptype, el_source) and not dry_run:
        return False

    summary_dir = (RESULTS_CWQ if "cwq" in script.name else RESULTS_WSP)
    out = summary_dir / f"full_{el_source}" / f"{ptype}_per_question.json"

    if out.exists() and not force:
        n = sum(1 for _ in open(out))
        print(f"  [skip] {out.name} already exists ({n} records)")
        return True

    cmd = [
        PYTHON_EXE, str(script),
        "--pert-type", ptype,
        "--el-source", el_source,
        "--mode", "full",
    ]
    if force:
        cmd.append("--force")

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True

    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, check=True, text=True,
                                capture_output=False)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed/60:.1f}min → {out}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED (exit {e.returncode}): {e}")
        return False


def load_summary(results_dir: Path, el_source: str, ptypes: List[str]) -> Dict[str, Any]:
    """Collect EM results from existing per_question.json files."""
    summary = {}
    for ptype in ptypes:
        pq_file = results_dir / f"full_{el_source}" / f"{ptype}_per_question.json"
        if not pq_file.exists():
            summary[ptype] = {"status": "missing"}
            continue
        with open(pq_file) as f:
            records = json.load(f)
        em_pert = [r["em_pert"] for r in records]
        em_clean = [r["em_clean"] for r in records]
        summary[ptype] = {
            "n": len(records),
            "EM_clean": round(100 * sum(em_clean) / len(em_clean), 2),
            "EM_pert": round(100 * sum(em_pert) / len(em_pert), 2),
            "delta_AG": round(100 * (sum(em_clean) - sum(em_pert)) / len(em_clean), 2),
        }
    return summary


def print_table(dataset: str, results: Dict[str, Dict[str, Any]], ptypes: List[str],
                el_sources: List[str]) -> None:
    """Print a LaTeX-ready summary table for the given dataset."""
    print(f"\n{'='*80}")
    print(f"  {dataset.upper()} — Secondary Perturbation Subgraph Ablation")
    print(f"  Clean EM: CWQ=52.9%, WebQSP=74.3%")
    print(f"{'='*80}")
    header = f"  {'Type':<22}"
    for el in el_sources:
        header += f"  {el.replace('_','-'):>16}"
    print(header + "  (EM%)")
    print(f"  {'-'*78}")
    for ptype in ptypes:
        row = f"  {PTYPE_LABELS.get(ptype, ptype):<22}"
        for el in el_sources:
            em = results.get(el, {}).get(ptype, {}).get("EM_pert", "—")
            row += f"  {str(em):>16}"
        print(row)


def main() -> None:
    p = argparse.ArgumentParser(description="S1-S4 GNN-RAG inference for NSM/GraftNet variants")
    p.add_argument("--ptype", default="all",
                   choices=S1S4_PTYPES + ["all"],
                   help="Perturbation type (default: all = p2,p3,p5,p6)")
    p.add_argument("--dataset", default="both", choices=["cwq", "webqsp", "both"])
    p.add_argument("--el-source", default="both_gem",
                   choices=list(EL_SOURCES.keys()),
                   help="Subgraph source. both_gem = nsm_gem + graftnetv3_gem (recommended)")
    p.add_argument("--force", action="store_true", help="Re-run even if output exists")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ptypes     = S1S4_PTYPES if args.ptype == "all" else [args.ptype]
    datasets   = ["cwq", "webqsp"] if args.dataset == "both" else [args.dataset]
    el_sources = EL_SOURCES[args.el_source]

    print(f"S1-S4 inference: ptypes={ptypes} datasets={datasets} el_sources={el_sources}")

    # ── Ablation source readiness check ────────────────────────────────────────
    # nsm_gem / nsm_elq / graftnetv3_gem / graftnetv3_elq directories exist only
    # for p4 and p7 in new_exp/gnn_input/ until exp_s1s4_subgraph_build.py completes.
    # Running with these sources for p2/p3/p5/p6 before the build returns
    # {"skipped": True} silently. We surface this explicitly here.
    ablation_sources_requested = [s for s in el_sources if s in ABLATION_SOURCES]
    if ablation_sources_requested:
        unbuilt = [p for p in ptypes if p not in ABLATION_BUILT_PTYPES]
        if unbuilt:
            NEW_EXP = REPO_ROOT
            built_unbuilt = []
            missing_unbuilt = []
            for pt in unbuilt:
                for src in ablation_sources_requested:
                    sg = NEW_EXP / "gnn_input" / f"cwq_{pt}_{src}" / "test_subgraph.json"
                    if sg.exists():
                        built_unbuilt.append(f"{pt}/{src}")
                    else:
                        missing_unbuilt.append(f"{pt}/{src}")
            if missing_unbuilt:
                print(f"\n[COVERAGE WARNING] The following (ptype/el_source) cells have no")
                print(f"  subgraph directory and will be SKIPPED silently by the inference script:")
                for cell in missing_unbuilt:
                    print(f"    {cell}")
                print(f"  → Run exp_s1s4_subgraph_build.py for these ptypes first.")
                print(f"  → Already built: {built_unbuilt or 'none'}")
                print(f"  → Pre-existing (p4/p7): always available\n")

    all_failed = []
    for dataset in datasets:
        script = E1_SCRIPT if dataset == "cwq" else E2_SCRIPT
        results_dir = RESULTS_CWQ if dataset == "cwq" else RESULTS_WSP

        print(f"\n{'='*70}")
        print(f"Dataset: {dataset.upper()}")

        for el_source in el_sources:
            print(f"\n  El-source: {el_source}")
            for ptype in ptypes:
                label = PTYPE_LABELS.get(ptype, ptype)
                print(f"\n  [{dataset}/{ptype}/{el_source}] — {label}")
                ok = run_inference_cell(script, ptype, el_source, args.force, args.dry_run)
                if not ok:
                    all_failed.append(f"{dataset}/{ptype}/{el_source}")

        if not args.dry_run:
            print(f"\n  Results summary for {dataset}:")
            all_results = {el: load_summary(results_dir, el, ptypes) for el in el_sources}
            print_table(dataset, all_results, ptypes, el_sources)

    if all_failed:
        print(f"\n[WARN] Failed cells:")
        for f in all_failed:
            print(f"  {f}")

    if not args.dry_run:
        print("\n[NEXT STEPS]")
        print("  1. Add NSM/GraftNet columns to Appendix Table for S1-S4 (ablate.tex)")
        print("  2. Verify: does GraftNet recover >1% EM for S3 (entity insertion)?")
        print("  3. Check if NSM helps S1 (relation paraphrase) via relation-weighted PPR")
        print("  4. Compare α_pert for new variants vs ELQ using exp_ppr_mass.py")


if __name__ == "__main__":
    main()
