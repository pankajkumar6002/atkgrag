#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_gnn_test_subgraph.py — Build GNN-RAG test_subgraph.json from scratch
for each perturbed P1-P6 dataset.

Strategy:
  1. Load GNN-RAG entities.txt / relations.txt → integer index maps
  2. For each perturbed question, get linked MIDs from ELQ output
  3. Look up subgraph triples from PKL cache (Tier 1) or nxhd files (Tier 2)
  4. Filter to triples where both entities AND relation are in GNN-RAG vocab
  5. Convert strings → integers and write GNN-RAG JSONL format

Output per dataset × p_type:
  data/perturbations/gnn_input/{dataset}_{p_type}/test_subgraph.json
  (symlinks to shared entities.txt / relations.txt / vocab.txt / word_emb.npy)

Usage:
  python build_gnn_test_subgraph.py --dataset cwq --p-type p1
  python build_gnn_test_subgraph.py --all
  python build_gnn_test_subgraph.py --all --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────────
GNN_DATA = REPO_ROOT / "external" / "GNN-RAG" / "data"
DATASET_DIR = REPO_ROOT / "data" / "perturbations"
MID_CACHE_DIR = REPO_ROOT / "data" / "mid_cache"
NXHD_CWQ_DIR = REPO_ROOT / "data" / "cwq_4hop_nxhd"
NXHD_WEBQSP_DIR = Path(
    str(REPO_ROOT / "data" / "freebase_2hops" / "stagg.neighborhoods")
)
EMB_DIR = DATASET_DIR / "embeddings"
OUT_BASE = DATASET_DIR / "gnn_input"

DATASET_SUBDIR = {"cwq": "CWQ", "webqsp": "webqsp"}
PERT_TYPES = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]

# Max triples per question — matches GNN-RAG p99 (~10K) but capped for memory
MAX_TRIPLES = 10000
MAX_ENTITIES = 2000

# ── Load GNN-RAG vocab once ────────────────────────────────────────────────────
_vocab_cache: Dict[str, Tuple[Dict, Dict, Dict, Dict]] = {}


def load_gnn_vocab(dataset: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Returns (entity2id, id2entity, relation2id, id2relation)."""
    if dataset in _vocab_cache:
        return _vocab_cache[dataset]
    subdir = DATASET_SUBDIR[dataset]
    ent_path = GNN_DATA / subdir / "entities.txt"
    rel_path = GNN_DATA / subdir / "relations.txt"
    with open(ent_path) as f:
        entities = [l.strip() for l in f if l.strip()]
    with open(rel_path) as f:
        relations = [l.strip() for l in f if l.strip()]
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    id2entity = {i: e for e, i in entity2id.items()}
    id2relation = {i: r for r, i in relation2id.items()}
    print(f"  [{dataset}] vocab: {len(entity2id):,} entities, {len(relation2id):,} relations")
    _vocab_cache[dataset] = (entity2id, id2entity, relation2id, id2relation)
    return _vocab_cache[dataset]


# ── PKL neighborhood cache ────────────────────────────────────────────────────
_pkl_cache: Dict[str, Any] = {}


def load_pkl(dataset: str) -> Dict:
    if dataset not in _pkl_cache:
        path = MID_CACHE_DIR / f"neighborhoods_{dataset}.pkl"
        print(f"  Loading {path} ...")
        with open(path, "rb") as f:
            _pkl_cache[dataset] = pickle.load(f)
        print(f"  Loaded {len(_pkl_cache[dataset]):,} MID entries")
    return _pkl_cache[dataset]


# ── Tier-2: .nxhd per-question files ─────────────────────────────────────────

def nxhd_triples(dataset: str, question_id: str) -> List[Tuple[str, str, str]]:
    if dataset == "cwq":
        pattern = str(NXHD_CWQ_DIR / f"{question_id}*.nxhd")
        matches = glob.glob(pattern)
        path = Path(matches[0]) if matches else None
    else:
        base_id = question_id.split("_")[0] if "_" in question_id else question_id
        p = NXHD_WEBQSP_DIR / f"{base_id}.nxhd"
        path = p if p.exists() else None

    if path is None:
        return []
    triples = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    triples.append((parts[0], parts[1], parts[2]))
    except Exception:
        pass
    return triples


# ── Core subgraph extraction ──────────────────────────────────────────────────

def extract_subgraph(
    dataset: str,
    question_id: str,
    seed_mids: List[str],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> Tuple[List[List[int]], List[int], int]:
    """
    Returns (tuples_int, entities_int, tier_used).
    tuples_int: [[eid, rid, eid], ...]
    entities_int: [eid, ...]  — union of all entity nodes in subgraph
    """
    nb = load_pkl(dataset)
    mid_set = set(seed_mids)

    raw_triples: List[Tuple[str, str, str]] = []
    tier = 0

    # Tier 1: PKL
    seen: Set[Tuple[str, str, str]] = set()
    for mid in seed_mids:
        entry = nb.get(mid)
        if entry:
            tier = 1
            for t in entry.get("tuples", []):
                key = (t[0], t[1], t[2])
                if key not in seen:
                    seen.add(key)
                    raw_triples.append(key)

    # Tier 2: nxhd if PKL missed
    if not raw_triples:
        raw_triples = nxhd_triples(dataset, question_id)
        if raw_triples:
            tier = 2

    if not raw_triples:
        # No subgraph found — return seed entities only
        ent_ids = [entity2id[m] for m in seed_mids if m in entity2id]
        return [], ent_ids[:MAX_ENTITIES], 0

    # Filter to triples where BOTH entities and relation are in GNN-RAG vocab
    valid: List[Tuple[str, str, str]] = []
    for s, p, o in raw_triples:
        if s in entity2id and o in entity2id and p in relation2id:
            valid.append((s, p, o))

    # Prioritise triples where subject or object is a seed MID
    priority = [t for t in valid if t[0] in mid_set or t[2] in mid_set]
    other = [t for t in valid if t[0] not in mid_set and t[2] not in mid_set]
    selected = priority[:MAX_TRIPLES]
    remaining = MAX_TRIPLES - len(selected)
    if remaining > 0:
        selected.extend(other[:remaining])

    # Entity set: all nodes appearing in selected triples + seed MIDs
    entity_mid_set: Set[str] = set()
    for s, p, o in selected:
        entity_mid_set.add(s)
        entity_mid_set.add(o)
    for m in seed_mids:
        if m in entity2id:
            entity_mid_set.add(m)

    entities_int = [entity2id[m] for m in entity_mid_set][:MAX_ENTITIES]

    # Convert to integer IDs — filter tuples to only use entities in the
    # truncated entity set so g2l lookups never miss.
    kept_entity_ids: Set[int] = set(entities_int)
    tuples_int = [
        [entity2id[s], relation2id[p], entity2id[o]]
        for s, p, o in selected
        if entity2id[s] in kept_entity_ids and entity2id[o] in kept_entity_ids
    ]
    return tuples_int, entities_int, tier


# ── Module-level worker (must be picklable for ProcessPoolExecutor) ───────────

def _process_one_worker(item: Dict) -> Dict:
    """Extract subgraph for one record. Runs in a subprocess."""
    rec       = item["rec"]
    seed_mids = item["seed_mids"]
    answers   = item["answers"]
    dataset   = item["dataset"]
    p_type    = item["p_type"]
    qid       = rec["id"]
    entity2id, id2entity, relation2id, id2relation = load_gnn_vocab(dataset)
    tuples_int, entities_int, tier = extract_subgraph(
        dataset, qid, seed_mids, entity2id, relation2id
    )
    seed_entity_ids = [entity2id[m] for m in seed_mids if m in entity2id]
    return {
        "id":       qid,
        "question": rec["perturbed_question"],
        "question_orig": rec.get("original_question", ""),
        "answers":  answers,
        "entities": seed_entity_ids,
        "subgraph": {"tuples": tuples_int, "entities": entities_int},
        "perturbation_type": p_type,
        "_tier":    tier,
    }


# ── Build test_subgraph.json ──────────────────────────────────────────────────

def build_dataset(
    dataset: str,
    p_type: str,
    dry_run: bool,
    el_source: str = "elq",
) -> None:
    subdir = DATASET_SUBDIR[dataset]
    in_path = DATASET_DIR / f"{dataset}_{p_type}.json"
    # Resolve EL source path
    if el_source == "agent_v1":
        el_path = DATASET_DIR / "el_agent" / "ELA_v1" / f"{dataset}_{p_type}_el.json"
        out_dir = OUT_BASE / f"{dataset}_{p_type}_agent_v1"
    elif el_source == "agent":
        el_path = DATASET_DIR / "el_agent" / f"{dataset}_{p_type}_el.json"
        out_dir = OUT_BASE / f"{dataset}_{p_type}_ela"
    else:  # elq (default)
        el_path = DATASET_DIR / "el_elq" / f"{dataset}_{p_type}_el.json"
        out_dir = OUT_BASE / f"{dataset}_{p_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_subgraph.json"

    if not in_path.exists():
        print(f"  [SKIP] {in_path} not found")
        return

    with open(in_path) as f:
        records: List[Dict] = json.load(f)
    if not el_path.exists():
        print(f"  [WARN] ELQ file missing: {el_path} — using empty entity seeds")
        el_by_id: Dict[str, List] = {}
    else:
        with open(el_path) as f:
            el_records = json.load(f)
        el_by_id = {r["id"]: r.get("linked_entities", []) for r in el_records}

    if dry_run:
        records = records[:10]

    entity2id, id2entity, relation2id, id2relation = load_gnn_vocab(dataset)

    # Load gold answers from original GNN-RAG test.json for answer IDs
    orig_test_path = GNN_DATA / subdir / "test.json"
    orig_by_id: Dict[str, Dict] = {}
    if orig_test_path.exists():
        with open(orig_test_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    orig_by_id[r["id"]] = r

    tier_counts = {0: 0, 1: 0, 2: 0}
    linked_count = 0

    # ── Pre-resolve seed MIDs for every record (CPU-bound, instant) ──────────
    resolved: List[Dict] = []
    for rec in records:
        qid = rec["id"]
        el_ents = el_by_id.get(qid, [])
        seed_mids = [e["mid"] for e in el_ents if e.get("mid")]
        seeded_by_elq = bool(seed_mids)

        if not seed_mids and rec.get("gold_entity_map"):
            seed_mids = [m for m in rec["gold_entity_map"].keys()
                         if m in entity2id and not m.startswith("g.")]

        if not seed_mids and qid in orig_by_id:
            orig_ent_ids = orig_by_id[qid].get("entities", [])
            seed_mids = [id2entity[i] for i in orig_ent_ids if i in id2entity]

        answers = orig_by_id.get(qid, {}).get("answers", rec.get("answers", []))
        resolved.append({
            "rec":        rec,
            "seed_mids":  seed_mids,
            "seeded":     seeded_by_elq or bool(seed_mids),
            "answers":    answers,
        })
    linked_count = sum(1 for r in resolved if r["seeded"])

    # ── Parallel subgraph extraction — 1 worker per core (96 cores) ──────────
    # extract_subgraph is I/O-bound (nxhd reads) + CPU (filtering).
    # PKL is loaded per-process via load_pkl() with module-level cache.
    NUM_WORKERS = min(40, len(resolved))

    # Embed dataset/p_type into each item so module-level worker can use them
    for item in resolved:
        item["dataset"] = dataset
        item["p_type"] = p_type

    # Use processes for true parallelism (bypasses GIL for nxhd I/O + parsing)
    out_records_by_qid: Dict[str, Dict] = {}
    if NUM_WORKERS > 1 and len(resolved) > 50:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
            futs = {pool.submit(_process_one_worker, item): item["rec"]["id"]
                    for item in resolved}
            done = 0
            for fut in as_completed(futs):
                result = fut.result()
                out_records_by_qid[result["id"]] = result
                done += 1
                if done % 500 == 0:
                    print(f"    {done}/{len(resolved)} done …", flush=True)
    else:
        for item in resolved:
            result = _process_one_worker(item)
            out_records_by_qid[result["id"]] = result

    # Write in original record order and tally tier stats
    with open(out_path, "w") as fout:
        for item in resolved:
            qid = item["rec"]["id"]
            out_rec = out_records_by_qid[qid]
            tier = out_rec.pop("_tier", 0)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            fout.write(json.dumps(out_rec) + "\n")

    # Symlink shared vocab files
    for fname in ["entities.txt", "relations.txt"]:
        src = GNN_DATA / subdir / fname
        dst = out_dir / fname
        if not dst.exists() and src.exists():
            os.symlink(src, dst)

    # Use extended vocab/emb for perturbations
    vocab_src = EMB_DIR / f"{dataset}_pert_vocab.txt"
    emb_src = EMB_DIR / f"{dataset}_pert_word_emb.npy"
    for src, fname in [(vocab_src, "vocab.txt"), (emb_src, "word_emb.npy")]:
        dst = out_dir / fname
        if not dst.exists():
            if src.exists():
                os.symlink(src, dst)
            else:
                fallback = GNN_DATA / subdir / fname
                if fallback.exists():
                    os.symlink(fallback, dst)

    n = len(records)
    print(f"  {dataset} {p_type}: {n} records → {out_path}")
    print(f"  ELQ linked: {linked_count}/{n} ({linked_count/n*100:.1f}%)")
    print(f"  Tier coverage: T1={tier_counts.get(1,0)} T2={tier_counts.get(2,0)} T0={tier_counts.get(0,0)}")
    triple_counts = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            triple_counts.append(len(r["subgraph"]["tuples"]))
    if triple_counts:
        import statistics
        print(f"  Triples: min={min(triple_counts)} max={max(triple_counts)} "
              f"median={statistics.median(triple_counts):.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GNN-RAG test_subgraph.json from PKL/nxhd for P1-P6"
    )
    parser.add_argument("--dataset", choices=["cwq", "webqsp"], default=None)
    parser.add_argument("--p-type", choices=PERT_TYPES, default=None)
    parser.add_argument("--el-source", choices=["elq", "agent", "agent_v1"], default="elq",
                        help="EL source: elq (default), agent (old ELA), agent_v1 (ELA-v1/BLINK+ELQ)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all:
        datasets = ["cwq", "webqsp"]
        p_types = PERT_TYPES
    else:
        if not args.dataset or not args.p_type:
            parser.error("Specify --dataset + --p-type or --all")
        datasets = [args.dataset]
        p_types = [args.p_type]

    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        for p_type in p_types:
            print(f"\n--- {dataset} {p_type} ---")
            build_dataset(dataset, p_type, dry_run=args.dry_run, el_source=args.el_source)


if __name__ == "__main__":
    main()