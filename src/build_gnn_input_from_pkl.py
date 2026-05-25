#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
Build gnn_input test_subgraph.json dirs from agent_subgraph PKLs.

Strategy (fast path):
  - Start from the existing ELQ gnn_input as the base (tier-0 questions are identical).
  - Only REPLACE records for which the agent PKL has actual ELA links (tier 1 or 1.5).
  - Update: entity seeds changed to ELA-linked MIDs; subgraph updated to ELA subgraph.

This avoids re-reading 117 GB of nxhd files for tier-0 questions.

Output dir: gnn_input/{dataset}_{p_type}_agent_fresh/

Usage:
  python3 src/build_gnn_input_from_pkl.py [--dataset cwq|webqsp] [--p-type p1..p6] [--all]
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT =  Path(__file__).resolve().parents[2]

SUBGRAPH_DIR = REPO_ROOT / "data" / "perturbations" / "subgraphs"
DATASET_DIR  = REPO_ROOT / "data" / "perturbations"
GNN_DATA     = REPO_ROOT / "external" / "GNN-RAG" / "data"
GNN_INPUT    = REPO_ROOT / "data" / "perturbations" / "gnn_input"

DATASETS = ["cwq", "webqsp"]
P_TYPES  = ["p1", "p2", "p3", "p4", "p5", "p6"]

MAX_TRIPLES  = 10000
MAX_ENTITIES = 2000


def strip_fb(token: str) -> str:
    if token.startswith("<fb:") and token.endswith(">"):
        return token[4:-1]
    return token


def build_one(dataset: str, p_type: str, el_source: str = "agent") -> None:
    subdir    = "CWQ" if dataset == "cwq" else "webqsp"
    elq_dir   = GNN_INPUT / f"{dataset}_{p_type}"
    out_dir   = GNN_INPUT / f"{dataset}_{p_type}_{el_source}_fresh"

    if (out_dir / "test_subgraph.json").exists():
        print(f"  [SKIP] {dataset} {p_type} {el_source}_fresh already exists")
        return

    # Load vocab
    ent_path = GNN_DATA / subdir / "entities.txt"
    rel_path = GNN_DATA / subdir / "relations.txt"
    with open(ent_path) as f:
        entities_list = [l.strip() for l in f if l.strip()]
    with open(rel_path) as f:
        relations_list = [l.strip() for l in f if l.strip()]
    entity2id  = {e: i for i, e in enumerate(entities_list)}
    relation2id = {r: i for i, r in enumerate(relations_list)}

    # Load agent subgraph PKL
    pkl_path = SUBGRAPH_DIR / f"{dataset}_{p_type}_{el_source}_subgraph.pkl"
    if not pkl_path.exists():
        print(f"  [MISS] {pkl_path} not found — run dense_retrieval.py first")
        return

    with open(pkl_path, "rb") as f:
        sg_data: Dict[str, dict] = pickle.load(f)

    # Collect qids where ELA found actual links (tier > 0)
    ela_linked: Dict[str, dict] = {
        qid: v for qid, v in sg_data.items()
        if v.get("tier_used", 0) > 0
    }
    print(f"  {dataset} {p_type}: {len(ela_linked)} ELA-linked questions "
          f"out of {len(sg_data)} total")

    # Read ELQ base gnn_input (fast: just load all lines)
    elq_subgraph_path = elq_dir / "test_subgraph.json"
    if not elq_subgraph_path.exists():
        print(f"  [MISS] ELQ base {elq_subgraph_path} not found")
        return

    base_records: Dict[str, dict] = {}
    with open(elq_subgraph_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                base_records[r["id"]] = r

    # Load canonical dataset records for perturbed question text
    canon_path = DATASET_DIR / f"{dataset}_{p_type}.json"
    with open(canon_path) as f:
        canon_list = json.load(f)
    canon_by_id = {r["id"]: r for r in canon_list}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_subgraph.json"

    stats = {"total": 0, "ela_replaced": 0, "elq_base": 0}

    with open(out_path, "w") as fout:
        for qid, base_rec in base_records.items():
            if qid in ela_linked:
                # Build fresh subgraph from ELA-linked MIDs
                sg = ela_linked[qid]
                seed_mids = sg.get("mids", [])
                raw_triples: List[Tuple[str, str, str]] = [
                    (strip_fb(s), strip_fb(p), strip_fb(o))
                    for s, p, o in sg.get("triples", [])
                ]
                mid_set = set(seed_mids)
                valid = [
                    (s, p, o) for s, p, o in raw_triples
                    if s in entity2id and o in entity2id and p in relation2id
                ]
                priority = [t for t in valid if t[0] in mid_set or t[2] in mid_set]
                other    = [t for t in valid if t[0] not in mid_set and t[2] not in mid_set]
                selected = (priority + other)[:MAX_TRIPLES]

                entity_mid_set = set()
                for s, p, o in selected:
                    entity_mid_set.add(s); entity_mid_set.add(o)
                for m in seed_mids:
                    if m in entity2id:
                        entity_mid_set.add(m)

                entities_int = [entity2id[m] for m in entity_mid_set][:MAX_ENTITIES]
                kept = set(entities_int)
                tuples_int = [
                    [entity2id[s], relation2id[p], entity2id[o]]
                    for s, p, o in selected
                    if entity2id[s] in kept and entity2id[o] in kept
                ]
                seed_ids = [entity2id[m] for m in seed_mids if m in entity2id]

                # Get perturbed question from canonical JSON
                canon = canon_by_id.get(qid, {})
                question = canon.get("perturbed_question", base_rec.get("question", ""))

                rec = {
                    "id": qid,
                    "question": question,
                    "question_orig": base_rec.get("question_orig", ""),
                    "answers": base_rec.get("answers", []),
                    "entities": seed_ids,
                    "subgraph": {"tuples": tuples_int, "entities": entities_int},
                    "perturbation_type": p_type,
                }
                stats["ela_replaced"] += 1
            else:
                # Keep ELQ base record as-is (same question text, same subgraph)
                rec = base_rec
                stats["elq_base"] += 1

            fout.write(json.dumps(rec) + "\n")
            stats["total"] += 1

    # Symlink vocab files from ELQ dir
    for fname in ["entities.txt", "relations.txt", "vocab.txt", "word_emb.npy"]:
        src = elq_dir / fname
        dst = out_dir / fname
        if src.exists() and not dst.exists():
            os.symlink(src, dst)

    print(f"  → {stats['total']} records  "
          f"ELA_replaced={stats['ela_replaced']} ELQ_base={stats['elq_base']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",   choices=DATASETS)
    ap.add_argument("--p-type",    choices=P_TYPES)
    ap.add_argument("--el-source", default="agent", choices=["agent", "hybrid"])
    ap.add_argument("--all",       action="store_true")
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
        build_one(ds, pt, args.el_source)

    print("Done.")


if __name__ == "__main__":
    main()