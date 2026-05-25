#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
sem_agent.py — Semantic Entity Linker Evaluator

Instead of binary exact-match (predicted MID == GT MID), measures how
semantically close the predicted entity NAME is to the GT entity NAME.

Motivation: ELQ/ELA often find the "right neighbourhood" entity but miss
the exact GT MID (e.g., predicts "World Series 2013" instead of "World Series").
A hard exact-match evaluation unfairly penalises such near-misses.

Algorithm:
  1. For each question, get GT entity names and predicted entity names
     (via entities_sup_names.tsv — 16M Freebase MID→name entries)
  2. Compute character-trigram TF-IDF cosine similarity between each
     predicted name and each GT name (fast, no GPU needed)
  3. Report:
     - Exact MID match rate (hard)
     - Soft GT match rate: max(similarity) >= threshold (default 0.6)
     - Average peak similarity (predicted→GT)
     - Distribution: exact / close (≥0.6) / partial (0.3–0.6) / far (<0.3)

Usage:
  python sem_agent.py --dataset cwq --p-type orig --el-source ela_v2
  python sem_agent.py --all --el-source ela_v2
  python sem_agent.py --dataset cwq --p-type orig --el-source elq
  python sem_agent.py --compare   # ELQ vs ELA v2 side-by-side on orig
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

DATASET_DIR  = REPO_ROOT / "data" / "perturbations"
GNN_DATA     = REPO_ROOT / "external" / "GNN-RAG" / "data"
RESOURCES    = REPO_ROOT / "data" / "resources"
NAMES_TSV    = RESOURCES / "entities_sup_names.tsv"

# ── Load entity name index ─────────────────────────────────────────────────────

_mid2name: Optional[Dict[str, str]] = None
_name2mid: Optional[Dict[str, str]] = None


def load_name_index() -> Tuple[Dict[str, str], Dict[str, str]]:
    global _mid2name, _name2mid
    if _mid2name is not None:
        return _mid2name, _name2mid  # type: ignore
    print(f"Loading entity name index from {NAMES_TSV.name} (16M entries)…", flush=True)
    mid2name: Dict[str, str] = {}
    name2mid: Dict[str, str] = {}
    with open(NAMES_TSV, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            mid, name = parts
            mid2name[mid] = name
            key = name.lower()
            if key not in name2mid:
                name2mid[key] = mid
    _mid2name = mid2name
    _name2mid = name2mid
    print(f"  {len(mid2name):,} MID→name entries loaded.", flush=True)
    return _mid2name, _name2mid


# ── TF-IDF character trigram similarity ───────────────────────────────────────

def char_trigrams(text: str) -> Dict[str, int]:
    text = text.lower().strip()
    ngrams: Dict[str, int] = {}
    padded = f"  {text}  "
    for i in range(len(padded) - 2):
        ng = padded[i:i+3]
        ngrams[ng] = ngrams.get(ng, 0) + 1
    return ngrams


def cosine_sim(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0) for k in a)
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def best_sim(pred_name: str, gt_names: List[str]) -> Tuple[float, str]:
    """Max cosine trigram similarity from pred_name to any GT name."""
    if not pred_name or not gt_names:
        return 0.0, ""
    pred_tg = char_trigrams(pred_name)
    best, best_gt = 0.0, ""
    for gn in gt_names:
        s = cosine_sim(pred_tg, char_trigrams(gn))
        if s > best:
            best, best_gt = s, gn
    return best, best_gt


# ── Load EL output ─────────────────────────────────────────────────────────────

def load_el(dataset: str, p_type: str, el_source: str) -> Optional[List[Dict]]:
    if el_source == "elq":
        path = DATASET_DIR / "el_elq" / f"{dataset}_{p_type}_el.json"
    elif el_source == "ela_v2":
        path = DATASET_DIR / "el_agent" / "ELA_v2" / f"{dataset}_{p_type}_el.json"
    elif el_source == "ela_v1":
        path = DATASET_DIR / "el_agent" / "ELA_v1" / f"{dataset}_{p_type}_el.json"
    else:
        path = Path(el_source)  # direct path
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def load_id2mid(dataset: str) -> Dict[int, str]:
    subdir = "CWQ" if dataset == "cwq" else "webqsp"
    with open(GNN_DATA / subdir / "entities.txt") as f:
        ents = [l.strip() for l in f if l.strip()]
    return {i: e for i, e in enumerate(ents)}


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate(
    dataset: str,
    p_type: str,
    el_source: str,
    threshold: float = 0.6,
    verbose: bool = False,
) -> Optional[Dict]:
    mid2name, _ = load_name_index()
    id2mid = load_id2mid(dataset)

    # GT seeds from orig
    orig_path = DATASET_DIR / f"{dataset}_orig.json"
    gt_map: Dict[str, Set[str]] = {}
    for r in json.load(open(orig_path)):
        gt_map[r["id"]] = {id2mid[i] for i in r.get("gnn_entities", []) if i in id2mid}

    records = load_el(dataset, p_type, el_source)
    if records is None:
        return None

    stats = {
        "n": 0, "exact": 0, "soft": 0, "partial": 0, "far": 0,
        "no_link": 0, "sims": [],
    }

    for rec in records:
        qid = rec["id"]
        gt_mids = gt_map.get(qid, set())
        if not gt_mids:
            continue
        gt_names = [mid2name.get(m, m) for m in gt_mids]

        linked = rec.get("linked_entities", [])
        if not linked:
            stats["no_link"] += 1
            stats["n"] += 1
            continue

        # Best predicted entity across all linked spans
        pred_mids = [e["mid"] for e in linked if e.get("mid")]

        # Hard exact match
        exact = bool(set(pred_mids) & gt_mids)

        # Soft similarity: for each predicted MID, get name, find best GT sim
        peak_sim = 0.0
        peak_gt  = ""
        peak_pred = ""
        for pmid in pred_mids:
            pname = mid2name.get(pmid, "")
            if not pname:
                continue
            s, gn = best_sim(pname, gt_names)
            if s > peak_sim:
                peak_sim, peak_gt, peak_pred = s, gn, pname

        stats["n"] += 1
        stats["sims"].append(peak_sim)

        if exact:
            stats["exact"] += 1
        elif peak_sim >= threshold:
            stats["soft"] += 1
        elif peak_sim >= 0.3:
            stats["partial"] += 1
        else:
            stats["far"] += 1

        if verbose and not exact and peak_sim >= 0.3:
            print(f"  Near-miss: pred={peak_pred!r} → gt={peak_gt!r} (sim={peak_sim:.2f})")

    n = max(1, stats["n"])
    result = {
        "dataset": dataset, "p_type": p_type, "el_source": el_source,
        "n": stats["n"],
        "exact_rate":   stats["exact"]   / n,
        "soft_rate":    stats["soft"]    / n,   # close but not exact
        "partial_rate": stats["partial"] / n,
        "far_rate":     stats["far"]     / n,
        "no_link_rate": stats["no_link"] / n,
        "mean_peak_sim": float(np.mean(stats["sims"])) if stats["sims"] else 0.0,
        # Combined: how many questions have AT LEAST a close semantic match
        "recoverable_rate": (stats["exact"] + stats["soft"]) / n,
    }
    return result


def print_table(results: List[Dict], threshold: float = 0.6) -> None:
    print(f"\n{'='*80}")
    print(f"  Semantic Entity Linker Evaluation (trigram cosine, soft-threshold={threshold})")
    print(f"  exact=MID match | soft=sim≥{threshold} | partial=0.3–{threshold} | far=<0.3 | recoverable=exact+soft")
    print(f"{'='*80}")
    print(f"  {'DS':<7} {'PT':<6} {'Source':<9} {'n':>5} {'exact%':>7} {'soft%':>6} {'partial%':>9} "
          f"{'far%':>5} {'recover%':>9} {'mean-sim':>9}")
    print("  " + "-"*75)
    for r in results:
        print(f"  {r['dataset']:<7} {r['p_type']:<6} {r['el_source']:<9} "
              f"{r['n']:>5} "
              f"{r['exact_rate']*100:>6.1f}% "
              f"{r['soft_rate']*100:>5.1f}% "
              f"{r['partial_rate']*100:>8.1f}% "
              f"{r['far_rate']*100:>4.1f}% "
              f"{r['recoverable_rate']*100:>8.1f}% "
              f"{r['mean_peak_sim']:>9.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic entity linker evaluator")
    parser.add_argument("--dataset",   choices=["cwq","webqsp"], default=None)
    parser.add_argument("--p-type",    choices=["orig","p1","p2","p3","p4","p5","p6"], default=None)
    parser.add_argument("--el-source", default="ela_v2",
                        help="elq | ela_v2 | ela_v1 (default: ela_v2)")
    parser.add_argument("--all",       action="store_true",
                        help="Run all 14 combos for both datasets")
    parser.add_argument("--compare",   action="store_true",
                        help="ELQ vs ELA v2 side-by-side on orig for both datasets")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Soft-match cosine threshold (default 0.6)")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print near-miss examples")
    args = parser.parse_args()

    results = []

    if args.compare:
        for ds in ["cwq","webqsp"]:
            for src in ["elq","ela_v2"]:
                r = evaluate(ds, "orig", src, threshold=args.threshold, verbose=args.verbose)
                if r: results.append(r)

    elif args.all:
        for ds in ["cwq","webqsp"]:
            for pt in ["orig","p1","p2","p3","p4","p5","p6"]:
                r = evaluate(ds, pt, args.el_source, threshold=args.threshold, verbose=args.verbose)
                if r: results.append(r)

    else:
        if not args.dataset or not args.p_type:
            parser.error("Specify --dataset + --p-type, --all, or --compare")
        r = evaluate(args.dataset, args.p_type, args.el_source,
                     threshold=args.threshold, verbose=args.verbose)
        if r: results.append(r)

    if results:
        print_table(results, threshold=args.threshold)


if __name__ == "__main__":
    main()