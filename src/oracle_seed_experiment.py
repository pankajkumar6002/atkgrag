#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
oracle_seed_experiment.py — Task 1 from To_do.md.

Inject ground-truth (GT) entity MIDs as seeds directly into GNN-RAG,
bypassing ELQ entirely, to isolate whether EL or architecture is the bottleneck.

This is the SINGLE MOST IMPORTANT EXPERIMENT in the robustness study.
The result determines whether the paper's claim is about EL failure or architecture.

Setup:
  For each P1 perturbed question:
    - Substitute the ELQ seed with the GT seed (gnn_entities[0])
    - Build the subgraph from new_seed_nxhd/{gt_mid}.nxhd
    - Run GNN-RAG with the GT-seeded subgraph
    - Report EM vs P1+ELQ baseline

Expected outcomes:
  Oracle seed → high EM:  EL is the bottleneck → reframe paper around EL failure
  Oracle seed → low EM:   Architecture is brittle → architecture claim holds

Usage:
  python oracle_seed_experiment.py --dataset cwq --p-type p1 --n-sample 100 --dry-run
  python oracle_seed_experiment.py --dataset cwq --p-type p1 --full
  python oracle_seed_experiment.py --all --dry-run

When oracle seeds are provided externally (from substituted entity MIDs):
  python oracle_seed_experiment.py --dataset cwq --p-type p1 \\
      --oracle-seeds tmlr/results/p1_oracle_seeds.json

Oracle seed JSON format:
  { "<qid>": "<oracle_mid>", ... }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = REPO_ROOT
TMLR    = REPO_ROOT / "tmlr"
PREPROC = REPO_ROOT / "Pre-process"
RESULTS = TMLR / "results"

DATASET_DIR = TMLR / "pert" / "dataset"
EL_DIR      = DATASET_DIR / "el_elq"
GNN_DATA    = TMLR / "Repo" / "GNN-RAG" / "data"
GNN_REPO    = TMLR / "Repo" / "GNN-RAG"

CWQ_SEED_DIR = PREPROC / "new_seed_nxhd"
WQ_SEED_DIR  = PREPROC / "webqsp_seed_nxhd"

OUT_DIR = RESULTS / "oracle_seed"
OUT_DIR.mkdir(exist_ok=True)


def load_id2mid(dataset: str) -> Dict[int, str]:
    subdir = "CWQ" if dataset == "cwq" else "webqsp"
    with open(GNN_DATA / subdir / "entities.txt") as f:
        return {i: l.strip() for i, l in enumerate(f) if l.strip()}


def load_mid2id(dataset: str) -> Dict[str, int]:
    return {m: i for i, m in load_id2mid(dataset).items()}


def get_gt_seeds(orig_rec: Dict, id2mid: Dict[int, str]) -> List[str]:
    """Return GT seed MIDs from gnn_entities field."""
    return [id2mid[i] for i in orig_rec.get("gnn_entities", []) if i in id2mid]


def get_substituted_mid_from_sparql(orig_sparql: str, pert_sparql: str) -> Optional[str]:
    """Extract the substituted entity MID from predicted_sparql."""
    import re
    _NS_MID_RE = re.compile(r"ns:(m\.[^\s\.\)>]+|g\.[^\s\.\)>]+)")
    orig_mids = {m.rstrip(")>.,;") for m in _NS_MID_RE.findall(orig_sparql)
                 if len(m) <= 18 and "000000" not in m}
    pert_mids = {m.rstrip(")>.,;") for m in _NS_MID_RE.findall(pert_sparql)
                 if len(m) <= 18 and "000000" not in m}
    new_mids = pert_mids - orig_mids
    return min(new_mids, key=len) if new_mids else None


def build_oracle_seeds_from_sparql(dataset: str, p_type: str) -> Dict[str, str]:
    """Build oracle seed map from predicted_sparql substituted entity MIDs.

    For P1 (adversarial entity substitution), the substituted entity MID
    is the oracle seed — it's the entity that was swapped in.
    For P2-P6 (paraphrase/structure), there's no entity substitution;
    use GT seeds instead.
    """
    ds_path   = DATASET_DIR / f"{dataset}_{p_type}.json"
    orig_path = DATASET_DIR / f"{dataset}_orig.json"

    if not ds_path.exists():
        return {}

    records   = json.load(open(ds_path))
    orig_recs = {r["id"]: r for r in json.load(open(orig_path))}
    id2mid    = load_id2mid(dataset)

    oracle_seeds: Dict[str, str] = {}

    for rec in records:
        qid  = rec["id"]
        orig = orig_recs.get(qid, {})

        # For P1: use substituted entity from predicted_sparql
        if p_type == "p1":
            meta    = rec.get("perturbation_meta", {})
            pert_sp = meta.get("predicted_sparql", "")
            orig_sp = rec.get("sparql", "")
            subst   = get_substituted_mid_from_sparql(orig_sp, pert_sp)
            if subst:
                oracle_seeds[qid] = subst
                continue

        # Fallback / other P-types: use GT seed (first gnn_entity)
        gt = get_gt_seeds(orig, id2mid)
        if gt:
            oracle_seeds[qid] = gt[0]

    return oracle_seeds


def nxhd_exists(dataset: str, mid: str) -> bool:
    """Check if the nxhd file for a MID exists."""
    if dataset == "cwq":
        return (CWQ_SEED_DIR / f"{mid}.nxhd").exists()
    else:
        return (WQ_SEED_DIR / f"{mid}.nxhd").exists()


def audit_oracle_seeds(
    dataset: str,
    p_type: str,
    oracle_seeds: Dict[str, str],
    orig_by_id: Dict[str, Dict],
    id2mid: Dict[int, str],
) -> Dict:
    """Audit oracle seeds: coverage, nxhd availability, GT-seed match rate."""
    n_total  = 0
    n_has_oracle = 0
    n_nxhd_found = 0
    n_same_as_gt = 0
    n_in_elq    = 0

    # Load ELQ for comparison
    elq_path = EL_DIR / f"{dataset}_{p_type}_el.json"
    elq_by_id = {}
    if elq_path.exists():
        elq_by_id = {r["id"]: r for r in json.load(open(elq_path))}

    for qid, orig in orig_by_id.items():
        gt_mids = {id2mid[i] for i in orig.get("gnn_entities", []) if i in id2mid}
        if not gt_mids:
            continue
        n_total += 1

        oracle = oracle_seeds.get(qid)
        if oracle:
            n_has_oracle += 1
            if nxhd_exists(dataset, oracle):
                n_nxhd_found += 1
            if oracle in gt_mids:
                n_same_as_gt += 1

        # Check ELQ
        elq_linked = elq_by_id.get(qid, {}).get("linked_entities", [])
        elq_mids = {e.get("mid","") for e in elq_linked if e.get("mid")}
        if elq_mids & gt_mids:
            n_in_elq += 1

    def pct(a, b): return round(100*a/max(1,b), 2)
    return {
        "n_total":          n_total,
        "n_has_oracle":     n_has_oracle,
        "oracle_coverage":  pct(n_has_oracle, n_total),
        "n_nxhd_found":     n_nxhd_found,
        "nxhd_coverage":    pct(n_nxhd_found, n_has_oracle),
        "n_oracle_is_gt":   n_same_as_gt,
        "oracle_gt_match":  pct(n_same_as_gt, n_has_oracle),
        "elq_any_gt_pct":   pct(n_in_elq, n_total),
    }


def write_gnn_oracle_input(
    dataset: str,
    p_type: str,
    oracle_seeds: Dict[str, str],
    n_sample: Optional[int] = None,
    dry_run: bool = False,
) -> Path:
    """Write the oracle-seeded input JSON for GNN-RAG.

    This replaces the ELQ entity linking output with oracle seeds,
    so GNN-RAG uses the correct ground-truth entity.

    Output format mirrors el_elq output: list of records with
    linked_entities: [{mid: oracle_mid, score: 0.0, mention: "oracle", ...}]
    """
    orig_el_path = EL_DIR / f"{dataset}_{p_type}_el.json"
    if not orig_el_path.exists():
        print(f"  [WARN] ELQ file not found: {orig_el_path}")
        return None

    elq_recs = json.load(open(orig_el_path))
    if n_sample:
        import random
        elq_recs = random.Random(42).sample(elq_recs, min(n_sample, len(elq_recs)))

    oracle_recs = []
    n_replaced = 0
    n_no_oracle = 0

    for rec in elq_recs:
        qid = rec["id"]
        oracle_mid = oracle_seeds.get(qid)
        out_rec = dict(rec)

        if oracle_mid:
            out_rec["linked_entities"] = [{
                "mention":   "oracle",
                "title":     "oracle",
                "entity_id": -1,
                "mid":       oracle_mid,
                "score":     0.0,
                "source":    "oracle_gt",
            }]
            n_replaced += 1
        else:
            n_no_oracle += 1

        oracle_recs.append(out_rec)

    out_path = OUT_DIR / f"{dataset}_{p_type}_oracle_el.json"
    if not dry_run:
        out_path.write_text(json.dumps(oracle_recs, indent=2, ensure_ascii=False))

    print(f"  Oracle EL file: n={len(oracle_recs)} replaced={n_replaced} no_oracle={n_no_oracle}")
    if dry_run:
        print(f"  [DRY RUN] would write → {out_path}")
    else:
        print(f"  Written → {out_path}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",       choices=["cwq", "webqsp"])
    ap.add_argument("--p-type",        default="p1")
    ap.add_argument("--all",           action="store_true")
    ap.add_argument("--n-sample",      type=int, default=None)
    ap.add_argument("--full",          action="store_true")
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--oracle-seeds",  default=None,
                    help="External oracle seeds JSON {qid: mid}. "
                         "If not provided, extract from predicted_sparql (P1) or GT seeds.")
    ap.add_argument("--audit-only",    action="store_true",
                    help="Only print audit stats, do not write oracle EL files.")
    args = ap.parse_args()

    n_sample = None if args.full else args.n_sample

    if args.all:
        if args.dataset:
            combos = [(args.dataset, pt)
                      for pt in ["p1", "p2", "p3", "p4", "p5", "p6"]]
        else:
            combos = [(ds, pt) for ds in ["cwq", "webqsp"]
                      for pt in ["p1", "p2", "p3", "p4", "p5", "p6"]]
    else:
        combos = [(args.dataset, args.p_type)]

    # Load external oracle seeds if provided
    ext_oracle: Optional[Dict[str, str]] = None
    if args.oracle_seeds:
        ext_oracle = json.load(open(args.oracle_seeds))
        print(f"Loaded {len(ext_oracle)} external oracle seeds from {args.oracle_seeds}")

    print(f"\n=== Oracle Seed Experiment ===")
    print(f"Output directory: {OUT_DIR}\n")

    all_audits: Dict = {}

    for dataset, p_type in combos:
        key = f"{dataset}_{p_type}"
        print(f"\n── {key} ──")

        id2mid    = load_id2mid(dataset)
        orig_path = DATASET_DIR / f"{dataset}_orig.json"
        if not orig_path.exists():
            print(f"  [SKIP] {orig_path} not found")
            continue
        orig_by_id = {r["id"]: r for r in json.load(open(orig_path))}

        # Build oracle seeds
        if ext_oracle:
            oracle_seeds = ext_oracle
        else:
            oracle_seeds = build_oracle_seeds_from_sparql(dataset, p_type)
            print(f"  Built {len(oracle_seeds)} oracle seeds from predicted_sparql")

        # Audit
        audit = audit_oracle_seeds(dataset, p_type, oracle_seeds, orig_by_id, id2mid)
        all_audits[key] = audit
        print(f"  Coverage:  {audit['n_has_oracle']}/{audit['n_total']} ({audit['oracle_coverage']:.1f}%)")
        print(f"  nxhd found: {audit['n_nxhd_found']} ({audit['nxhd_coverage']:.1f}%)")
        print(f"  Oracle is GT: {audit['n_oracle_is_gt']} ({audit['oracle_gt_match']:.1f}%)")
        print(f"  ELQ any_gt: {audit['elq_any_gt_pct']:.1f}%")

        if args.audit_only:
            continue

        # Write oracle EL file for GNN-RAG
        write_gnn_oracle_input(dataset, p_type, oracle_seeds,
                                n_sample=n_sample, dry_run=args.dry_run)

    # Summary
    print(f"\n{'='*70}")
    print("Oracle Seed Audit Summary")
    print(f"{'='*70}")
    print(f"  {'Key':<18} {'Coverage':>9} {'nxhd%':>7} {'OracIsGT%':>10} {'ELQ_any%':>9}")
    print("  " + "-"*60)
    for key, a in sorted(all_audits.items()):
        print(f"  {key:<18} {a['oracle_coverage']:>8.1f}% {a['nxhd_coverage']:>6.1f}% "
              f"{a['oracle_gt_match']:>9.1f}% {a['elq_any_gt_pct']:>8.1f}%")

    if not args.dry_run and not args.audit_only:
        (OUT_DIR / "audit.json").write_text(json.dumps(all_audits, indent=2))
        print(f"\nAudit saved → {OUT_DIR / 'audit.json'}")
        print(f"\nNext steps:")
        print(f"  1. Run GNN-RAG with oracle EL: python tmlr/scripts/experiments/e1_cwq_gnn_rag.py \\")
        print(f"       --pert-type p1 --el-source oracle")
        print(f"  2. Compare EM: oracle vs ELQ vs ELA")
        print(f"  3. If oracle EM >> ELQ EM → EL is bottleneck (reframe paper)")


if __name__ == "__main__":
    main()