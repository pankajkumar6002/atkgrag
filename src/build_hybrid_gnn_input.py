#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
Build hybrid gnn_input dirs by merging ELQ and ELA seeds + subgraphs.

For each question: union(ELQ entities, ELA entities) as seeds;
union(ELQ subgraph tuples, ELA subgraph tuples) as the combined subgraph.

Usage:
  python3 src/build_hybrid_gnn_input.py [--dataset cwq|webqsp] [--p-type p1..p6] [--all]
  python3 src/build_hybrid_gnn_input.py --all         # all datasets × p-types
"""

import argparse, json, os
from pathlib import Path

REPO_ROOT =  Path(__file__).resolve().parents[2]
GNN_INPUT = REPO_ROOT / "data" / "perturbations" / "gnn_input"

DATASETS  = ["cwq", "webqsp"]
P_TYPES   = ["p1", "p2", "p3", "p4", "p5", "p6"]


def merge_one(elq_path: Path, ela_path: Path, out_path: Path) -> dict:
    if not elq_path.exists():
        return {"skipped": f"ELQ missing: {elq_path}"}
    if not ela_path.exists():
        return {"skipped": f"ELA missing: {ela_path}"}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "elq_only": 0, "ela_only": 0, "both": 0,
             "elq_seeds": 0, "ela_seeds": 0, "hybrid_seeds": 0,
             "elq_tuples": 0, "ela_tuples": 0, "hybrid_tuples": 0}

    # Load both files indexed by id
    def load(p):
        d = {}
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    d[r["id"]] = r
        return d

    elq = load(elq_path)
    ela = load(ela_path)
    all_ids = sorted(set(elq) | set(ela))

    with open(out_path, "w") as fout:
        for qid in all_ids:
            re = elq.get(qid, {})
            ra = ela.get(qid, {})

            # Use ELA question as preferred (same as perturbation), fallback ELQ
            question      = ra.get("question") or re.get("question", "")
            question_orig = ra.get("question_orig") or re.get("question_orig", "")
            answers       = ra.get("answers") or re.get("answers", [])
            pert_type     = ra.get("perturbation_type") or re.get("perturbation_type", "")

            # Union seeds (integer entity IDs)
            elq_ents = set(re.get("entities", []))
            ela_ents = set(ra.get("entities", []))
            hybrid_ents = sorted(elq_ents | ela_ents)

            # Union subgraph tuples (lists of 3 ints), deduplicate
            elq_tuples = [tuple(t) for t in re.get("subgraph", {}).get("tuples", [])]
            ela_tuples = [tuple(t) for t in ra.get("subgraph", {}).get("tuples", [])]
            hybrid_tuples_set = set(elq_tuples) | set(ela_tuples)
            hybrid_tuples = [list(t) for t in hybrid_tuples_set]

            # Union subgraph entity sets
            elq_sg_ents = set(re.get("subgraph", {}).get("entities", []))
            ela_sg_ents = set(ra.get("subgraph", {}).get("entities", []))
            hybrid_sg_ents = sorted(elq_sg_ents | ela_sg_ents | elq_ents | ela_ents)

            rec = {
                "id": qid,
                "question": question,
                "question_orig": question_orig,
                "answers": answers,
                "entities": hybrid_ents,
                "subgraph": {"tuples": hybrid_tuples, "entities": hybrid_sg_ents},
                "perturbation_type": pert_type,
            }
            fout.write(json.dumps(rec) + "\n")

            stats["total"] += 1
            has_elq = bool(elq_ents)
            has_ela = bool(ela_ents)
            if has_elq and has_ela:   stats["both"] += 1
            elif has_elq:             stats["elq_only"] += 1
            elif has_ela:             stats["ela_only"] += 1
            stats["elq_seeds"]    += len(elq_ents)
            stats["ela_seeds"]    += len(ela_ents)
            stats["hybrid_seeds"] += len(hybrid_ents)
            stats["elq_tuples"]   += len(elq_tuples)
            stats["ela_tuples"]   += len(ela_tuples)
            stats["hybrid_tuples"]+= len(hybrid_tuples)

    return stats


def symlink_aux(src_dir: Path, out_dir: Path):
    """Symlink vocab/embedding files from ELQ dir."""
    for fname in ["entities.txt", "relations.txt", "vocab.txt", "word_emb.npy"]:
        src = src_dir / fname
        dst = out_dir / fname
        if src.exists() and not dst.exists():
            os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--p-type",  choices=P_TYPES)
    ap.add_argument("--all",     action="store_true")
    args = ap.parse_args()

    pairs = []
    if args.all:
        for ds in DATASETS:
            for pt in P_TYPES:
                pairs.append((ds, pt))
    else:
        if not args.dataset or not args.p_type:
            ap.error("Provide --dataset and --p-type, or --all")
        pairs = [(args.dataset, args.p_type)]

    for ds, pt in pairs:
        elq_dir = GNN_INPUT / f"{ds}_{pt}"
        ela_dir = GNN_INPUT / f"{ds}_{pt}_ela"
        out_dir = GNN_INPUT / f"{ds}_{pt}_hybrid"

        if (out_dir / "test_subgraph.json").exists():
            print(f"  [SKIP] {ds} {pt} hybrid already exists")
            continue

        stats = merge_one(
            elq_dir / "test_subgraph.json",
            ela_dir / "test_subgraph.json",
            out_dir / "test_subgraph.json",
        )
        if "skipped" in stats:
            print(f"  [SKIP] {ds} {pt}: {stats['skipped']}")
            continue

        symlink_aux(elq_dir, out_dir)
        avg_gain = (stats["hybrid_seeds"] - stats["elq_seeds"]) / max(stats["total"], 1)
        print(f"  {ds} {pt} hybrid: {stats['total']} q  "
              f"seeds ELQ={stats['elq_seeds']//stats['total']} "
              f"ELA={stats['ela_seeds']//stats['total']} "
              f"hybrid={stats['hybrid_seeds']//stats['total']} "
              f"| tuples {stats['elq_tuples']//stats['total']}→"
              f"{stats['hybrid_tuples']//stats['total']}")

    print("Done.")


if __name__ == "__main__":
    main()