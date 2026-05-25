#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_p3p6_gnn_subgraph.py  —  Build GNN-RAG test_subgraph.json for P3–P6.

P3–P6 only change the question text (paraphrase / relation synonym /
adversarial paraphrase / grammatical perturbation).  Entity seeds, subgraph
topology, and gold answers are identical to the original test split.

We take data/webqsp/test.json or data/CWQ/test.json (integer IDs) and
replace only the 'question' field with the perturbed question.

Outputs  →  data/perturbations/gnn-rag/
  {dataset}_orig_test.json       original test split (copy, for baseline run)
  {dataset}_{p_type}_test.json   perturbed question + original subgraph

Format (one JSON per line, same as GNN-RAG expects):
  id, question, question_orig, answers, entities, subgraph, perturbation_type

Usage
-----
  python3 src/build_p3p6_gnn_subgraph.py            # all
  python3 src/build_p3p6_gnn_subgraph.py --dataset webqsp --p-type p3
  python3 src/build_p3p6_gnn_subgraph.py --dry-run
  python3 src/build_p3p6_gnn_subgraph.py --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT =  Path(__file__).resolve().parents[2]
GNN_DATA = REPO_ROOT / "external" / "GNN-RAG" / "data"
PERT_DIR = REPO_ROOT / "data" / "perturbations"
OUT_DIR  = REPO_ROOT / "data/perturbations/gnn-rag"

ORIG_TEST = {
    "webqsp": GNN_DATA / "webqsp" / "test.json",
    "cwq":    GNN_DATA / "CWQ"    / "test.json",
}

P_TYPES = ["p3", "p4", "p5", "p6"]


def load_orig(dataset: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with open(ORIG_TEST[dataset]) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["id"]] = rec
    return records


def load_pert(dataset: str, p_type: str) -> list[dict]:
    with open(PERT_DIR / f"{dataset}_{p_type}.json") as f:
        return json.load(f)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_orig(dataset: str, orig: dict[str, dict], dry_run: bool) -> None:
    out_path = OUT_DIR / f"{dataset}_orig_test.json"
    if out_path.exists() or dry_run:
        return
    lines = [json.dumps(rec) for rec in orig.values()]
    write_lines(out_path, lines)
    print(f"  → {out_path}  ({len(orig)} records)")


def build_one(
    dataset: str,
    p_type: str,
    orig: dict[str, dict],
    dry_run: bool,
    overwrite: bool,
) -> dict:
    out_path = OUT_DIR / f"{dataset}_{p_type}_test.json"
    if out_path.exists() and not overwrite and not dry_run:
        print(f"  [SKIP] {out_path} — pass --overwrite to rebuild")
        return {"skipped": True}

    pert = load_pert(dataset, p_type)
    stats = {"total": 0, "written": 0, "missing_orig": 0,
             "empty_entities": 0, "empty_subgraph": 0}
    issues: list[str] = []
    lines: list[str] = []

    for rec in pert:
        stats["total"] += 1
        qid = rec["id"]
        orig_rec = orig.get(qid)
        if orig_rec is None:
            issues.append(f"MISSING in orig: {qid}")
            stats["missing_orig"] += 1
            continue

        ents = orig_rec.get("entities", [])
        if not ents:
            issues.append(f"EMPTY entities: {qid}")
            stats["empty_entities"] += 1

        sg = orig_rec.get("subgraph", {"tuples": [], "entities": []})
        if not sg.get("tuples"):
            issues.append(f"EMPTY subgraph tuples: {qid}")
            stats["empty_subgraph"] += 1

        out_rec = {
            "id":               qid,
            "question":         rec["perturbed_question"],
            "question_orig":    rec.get("original_question",
                                        orig_rec.get("question", "")),
            "answers":          orig_rec.get("answers", []),
            "entities":         ents,
            "subgraph":         sg,
            "perturbation_type": rec.get("perturbation_type", p_type),
        }
        lines.append(json.dumps(out_rec))
        stats["written"] += 1

    for msg in issues[:20]:
        print(f"    [WARN] {msg}")
    if len(issues) > 20:
        print(f"    ... {len(issues) - 20} more issues not shown")

    if not dry_run and lines:
        write_lines(out_path, lines)

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",   choices=["webqsp", "cwq"])
    ap.add_argument("--p-type",    choices=P_TYPES)
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else ["webqsp", "cwq"]
    p_types  = [args.p_type]  if args.p_type  else P_TYPES

    for dataset in datasets:
        print(f"\n── {dataset} ──")
        orig = load_orig(dataset)
        print(f"  Loaded {len(orig)} original records")
        write_orig(dataset, orig, dry_run=args.dry_run)

        for p_type in p_types:
            print(f"\n  {dataset} {p_type} ...")
            stats = build_one(dataset, p_type, orig,
                              dry_run=args.dry_run,
                              overwrite=args.overwrite)
            if stats.get("skipped"):
                continue
            tag = "[DRY-RUN] " if args.dry_run else ""
            print(f"  {tag}{dataset} {p_type}: "
                  f"written={stats['written']}/{stats['total']}  "
                  f"missing_orig={stats['missing_orig']}  "
                  f"empty_ents={stats['empty_entities']}  "
                  f"empty_sg={stats['empty_subgraph']}")
            if not args.dry_run and stats["written"]:
                print(f"  → {OUT_DIR / f'{dataset}_{p_type}_test.json'}")


if __name__ == "__main__":
    main()