#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
exp_s1s4_reuse_p4pkl.py
========================
Re-run Stage 2+3 for S1-S4 (P2,P3,P5,P6) using the high-quality P4 pkl files.

P4 pkls were built from pre-existing NSM Stage2 subgraphs (fast-path), giving
rich seed-adjacent triple coverage. S1-S4 fresh BFS pkls have the same seeds
but poor coverage due to the 100k cap filling with distant nodes.

Since all perturbation types share the same base questions and GEM seeds,
P4 pkls are valid inputs for S1-S4 Stage2 PPR runs.

Usage:
  python exp_s1s4_reuse_p4pkl.py --method both --dataset both --workers 16
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).parent.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_ablation import (
    stage2, stage3, _ablation_tag,
    REPO_ROOT, NEW_EXP, PERT_DIR,
    NSM_CAP_PPR, CWQ_CAP_PPR, WEBQSP_CAP_PPR,
    load_elq_seeds,
)
import json

S1S4_PTYPES = ["p2", "p3", "p5", "p6"]
PTYPE_LABELS = {
    "p2": "S1 — Relation paraphrase",
    "p3": "S2 — Voice flip",
    "p5": "S3 — Adversarial entity insertion",
    "p6": "S4 — Mixed lexical noise",
}

log = logging.getLogger("s1s4_reuse_p4pkl")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = REPO_ROOT / "tmlr" / "logs" / f"s1s4_p4reuse_{ts}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    p = argparse.ArgumentParser(description="Rebuild S1-S4 Stage2+3 using P4 pkl files")
    p.add_argument("--ptype", default="all", choices=S1S4_PTYPES + ["all"])
    p.add_argument("--dataset", default="both", choices=["cwq", "webqsp", "both"])
    p.add_argument("--method", default="both", choices=["nsm", "graftnetv3", "both"])
    p.add_argument("--workers", type=int, default=16, help="Stage2 PPR parallel workers")
    p.add_argument("--no-resume", action="store_true", help="Force re-run even if output exists")
    args = p.parse_args()

    ptypes   = S1S4_PTYPES if args.ptype == "all" else [args.ptype]
    datasets = ["cwq", "webqsp"] if args.dataset == "both" else [args.dataset]
    methods  = ["nsm", "graftnetv3"] if args.method == "both" else [args.method]
    resume   = not args.no_resume

    log.info(f"S1-S4 Stage2+3 re-run using P4 pkls: ptypes={ptypes} datasets={datasets} methods={methods}")
    log.info(f"Log: {log_file}")

    for ptype in ptypes:
        for ds in datasets:
            for method in methods:
                seed_mode = "gem"
                tag = _ablation_tag(method, seed_mode, ds, ptype)
                label = PTYPE_LABELS.get(ptype, ptype)

                log.info(f"\n{'='*70}")
                log.info(f"Cell: {tag}  [{label}]")

                # Use P4 pkl dir (high-quality, built from NSM Stage2)
                p4_s1_dir = NEW_EXP / "subgraphs" / f".s1_{ds}_p4_{method}_gem"
                if not p4_s1_dir.exists():
                    log.error(f"  P4 pkl dir missing: {p4_s1_dir} — SKIP")
                    continue
                pkl_count = len(list(p4_s1_dir.glob("*.pkl")))
                log.info(f"  Using P4 pkl dir: {p4_s1_dir} ({pkl_count} pkls)")

                # Load records for this ptype
                pert_file = PERT_DIR / f"{ds}_{ptype}.json"
                if not pert_file.exists():
                    log.warning(f"  Dataset not found: {pert_file} — SKIP")
                    continue
                records = json.loads(pert_file.read_text())
                log.info(f"  [{ds}] {len(records)} records from {pert_file.name}")

                elq_seeds = {}  # gem seed mode — no ELQ needed

                # Output dirs
                sg_dir  = NEW_EXP / "subgraphs" / tag
                gnn_dir = NEW_EXP / "gnn_input" / tag
                cap = (CWQ_CAP_PPR if ds == "cwq" else WEBQSP_CAP_PPR) if method == "graftnetv3" else NSM_CAP_PPR

                # Check if already done
                out_sg = gnn_dir / "test_subgraph.json"
                if resume and out_sg.exists():
                    n = sum(1 for _ in open(out_sg))
                    if n > 0:
                        log.info(f"  [skip] {n} records already in {out_sg}")
                        continue

                # Stage 2: PPR filtering using P4 pkls
                t0 = time.time()
                stage2(method, ds, ptype, seed_mode, records, elq_seeds,
                       p4_s1_dir, sg_dir, cap, False, args.workers)
                log.info(f"  [{ds}] Stage2 done in {time.time()-t0:.1f}s")

                # Stage 3: GNN-RAG integer encoding
                t0 = time.time()
                stage3(ds, ptype, records, sg_dir, gnn_dir)
                log.info(f"  [{ds}] Stage3 done in {time.time()-t0:.1f}s")

    log.info("\nAll cells done. Next: run exp_s1s4_inference.py to evaluate GNN-RAG EM.")
    for ptype in ptypes:
        for ds in datasets:
            for method in methods:
                tag = _ablation_tag(method, "gem", ds, ptype)
                out = NEW_EXP / "gnn_input" / tag / "test_subgraph.json"
                if out.exists():
                    n = sum(1 for _ in open(out))
                    log.info(f"  {tag}: {n} records → {out}")
                else:
                    log.info(f"  {tag}: MISSING — {out}")


if __name__ == "__main__":
    main()