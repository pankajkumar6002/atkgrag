#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_subgraph_nxhd.py  —  Build per-question subgraphs from .nxhd files with PPR pruning.

Sources
-------
CWQ  (4-hop):
  1. cwq_4hop_nxhd/{exact_qid}.nxhd        — 4-hop neighbourhood keyed by full question ID
  2. new_seed_nxhd/{top_elq_mid}.nxhd      — 2-hop of the top-scored ELQ seed entity

WebQSP (2-hop):
  1. webqsp_2hop_nxhd/.../stagg.neighborhoods/{base_qid}.nxhd  — base "WebQTest-N" ID
  2. webqsp_seed_nxhd/{top_elq_mid}.nxhd                       — top-scored ELQ seed

Pipeline per question
---------------------
  1. Parse & UNION triples from both sources (no cap during load)
  2. Extract entity set: subjects + objects that look like MIDs (m.* or g.*)
  3. Run approximate Personalized PageRank (PPR) from seed entity
  4. Prune to top-K entities by PPR score (K=2500 CWQ, K=2000 WebQSP)
  5. Filter triples to edges whose subject is in kept set

Optimisations
-------------
  - LRU file cache (_NXHD_CACHE): parsed qid nxhd files (41 MB each) are reused
    across pert and orig comparisons for the same question.  Cache size = 128 files.
  - Skip-trivial: when pert_seed == orig_seed the subgraph is identical; Jaccard=1.0.
  - --dataset cwq --all  runs only CWQ combos (7 p-types).

Output
------
  tmlr/results/subgraph/{dataset}_{p_type}.json   — per-p-type result file
  tmlr/results/subgraph/summary.json              — aggregate across all done combos

Usage
-----
  python build_subgraph_nxhd.py --dataset cwq --p-type orig --n-sample 10
  python build_subgraph_nxhd.py --dataset cwq --all             # cwq orig+p1-p6
  python build_subgraph_nxhd.py --dataset webqsp --all          # webqsp orig+p1-p6
  python build_subgraph_nxhd.py --all                           # all 14 combos
  python build_subgraph_nxhd.py --all --full                    # no sampling
  python build_subgraph_nxhd.py --dataset cwq --p-type p1 --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import time
from collections import defaultdict, deque, OrderedDict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = REPO_ROOT
TMLR      = REPO_ROOT / "tmlr"
PREPROC   = REPO_ROOT / "Pre-process"

DATASET_DIR = TMLR / "pert" / "dataset"
EL_ELQ_DIR  = DATASET_DIR / "el_elq"
GNN_DATA    = TMLR / "Repo" / "GNN-RAG" / "data"

# Output directory
OUT_DIR = TMLR / "results" / "subgraph"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CWQ nxhd sources
CWQ_4HOP_DIR  = PREPROC / "cwq_4hop_nxhd"
CWQ_SEED_DIR  = PREPROC / "new_seed_nxhd"

# WebQSP nxhd sources
WQ_2HOP_DIR   = PREPROC / "webqsp_2hop_nxhd" / "dataset" / "freebase_2hops" / "stagg.neighborhoods"
WQ_SEED_DIR   = PREPROC / "webqsp_seed_nxhd"

# Entity caps
CAP_CWQ    = 2500
CAP_WEBQSP = 2000

# PPR parameters
PPR_ALPHA = 0.15    # teleportation probability
PPR_ITERS = 30      # power-iteration steps
PPR_TOL   = 1e-6    # convergence tolerance

# nxhd parse cap — stop reading a file after this many lines to limit PPR graph size.
# CWQ 4-hop files have 660K lines; PPR over 100K+ nodes is too slow.
# At 50K lines we get ~30K-50K triples → manageable PPR graph, still good coverage.
# Set to None to disable (use full file).
NXHD_MAX_LINES: Optional[int] = 50_000

# nxhd file LRU cache (avoids re-parsing 41MB files for pert+orig of same qid)
_NXHD_CACHE: "OrderedDict[str, Tuple[Set[str], Set[Tuple[str,str,str]]]]" = OrderedDict()
_NXHD_CACHE_SIZE = 128


# ── GNN entity index ──────────────────────────────────────────────────────────

_id2mid_cache: Dict[str, Dict[int, str]] = {}

def load_id2mid(dataset: str) -> Dict[int, str]:
    if dataset not in _id2mid_cache:
        subdir = "CWQ" if dataset == "cwq" else "webqsp"
        with open(GNN_DATA / subdir / "entities.txt") as f:
            _id2mid_cache[dataset] = {i: l.strip() for i, l in enumerate(f) if l.strip()}
    return _id2mid_cache[dataset]


# ── nxhd parsing ──────────────────────────────────────────────────────────────

_FB_PREFIX = "<fb:"

def _strip(token: str) -> str:
    t = token.strip()
    if t.startswith(_FB_PREFIX) and t.endswith(">"):
        return t[4:-1]
    return t


def _is_entity(token: str) -> bool:
    return token.startswith("m.") or token.startswith("g.")


def parse_nxhd(path: Path) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """Parse one .nxhd file → (entity_set, triple_set), with LRU cache.

    Entities = subjects always + objects that are MIDs (m.* / g.*).
    Plain-text objects included in triples but NOT counted as entities.
    """
    key = str(path)
    if key in _NXHD_CACHE:
        _NXHD_CACHE.move_to_end(key)
        return _NXHD_CACHE[key]

    entities: Set[str] = set()
    triples:  Set[Tuple[str, str, str]] = set()

    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for n_lines, line in enumerate(fh):
                if NXHD_MAX_LINES and n_lines >= NXHD_MAX_LINES:
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                s = _strip(parts[0])
                r = _strip(parts[1])
                o = _strip(parts[2])
                if not s or not r:
                    continue
                triples.add((s, r, o))
                if _is_entity(s):
                    entities.add(s)
                if _is_entity(o):
                    entities.add(o)
    except FileNotFoundError:
        pass

    result = (entities, triples)
    _NXHD_CACHE[key] = result
    if len(_NXHD_CACHE) > _NXHD_CACHE_SIZE:
        _NXHD_CACHE.popitem(last=False)
    return result


# ── Approximate Personalized PageRank (PPR) ───────────────────────────────────

def approx_ppr(
    triples: Set[Tuple[str, str, str]],
    seed_mid: str,
    alpha: float = PPR_ALPHA,
    n_iters: int = PPR_ITERS,
    tol: float   = PPR_TOL,
) -> Dict[str, float]:
    """Power-iteration PPR (undirected).

    Personalization = uniform over {seed} ∪ 1-hop neighbours of seed.
    Returns {entity_mid: ppr_score}.
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    all_ents: Set[str] = set()

    for s, r, o in triples:
        s_ok = _is_entity(s)
        o_ok = _is_entity(o)
        if s_ok:
            all_ents.add(s)
        if o_ok:
            all_ents.add(o)
        if s_ok and o_ok:
            adj[s].append(o)
            adj[o].append(s)

    if not all_ents:
        return {}

    ent_list = sorted(all_ents)
    ent_idx  = {e: i for i, e in enumerate(ent_list)}
    N        = len(ent_list)

    # Personalization vector
    seed_nbrs: Set[str] = {seed_mid} | (set(adj.get(seed_mid, [])) & all_ents)
    p = np.zeros(N, dtype=np.float32)
    for e in seed_nbrs:
        if e in ent_idx:
            p[ent_idx[e]] = 1.0
    if p.sum() == 0:
        p[:] = 1.0
    p /= p.sum()

    # Degree for row-normalisation
    deg = np.array([max(1, len(adj[e])) for e in ent_list], dtype=np.float32)

    r = p.copy()
    for _ in range(n_iters):
        r_new = np.zeros(N, dtype=np.float32)
        for i, e in enumerate(ent_list):
            if r[i] == 0.0:
                continue
            contrib = r[i] / deg[i]
            for nb in adj[e]:
                j = ent_idx.get(nb)
                if j is not None:
                    r_new[j] += contrib
        r_new = (1.0 - alpha) * r_new + alpha * p
        s = r_new.sum()
        if s > 0:
            r_new /= s
        if np.abs(r_new - r).max() < tol:
            r = r_new
            break
        r = r_new

    return {ent_list[i]: float(r[i]) for i in range(N)}


# ── Subgraph pruning ──────────────────────────────────────────────────────────

def prune_by_ppr(
    raw_entities: Set[str],
    raw_triples:  Set[Tuple[str, str, str]],
    seed_mid:     str,
    cap:          int,
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """Keep top-cap entities by PPR score; filter triples accordingly."""
    if len(raw_entities) <= cap:
        return raw_entities, raw_triples

    ppr = approx_ppr(raw_triples, seed_mid)
    if not ppr:
        kept = set(list(raw_entities)[:cap])
    else:
        ranked = sorted(ppr.items(), key=lambda x: -x[1])
        kept   = {seed_mid} if seed_mid in raw_entities else set()
        for e, _ in ranked:
            if len(kept) >= cap:
                break
            kept.add(e)

    # Keep triple if its MID-entity subject is in kept set
    # (non-entity objects like plain strings are kept along with kept subjects)
    kept_trips = {(s, r, o) for s, r, o in raw_triples
                  if not _is_entity(s) or s in kept}

    return kept, kept_trips


# ── BFS golden-path check ─────────────────────────────────────────────────────

def bfs_path_exists(
    adj: Dict[str, Set[str]],
    start: str,
    targets: Set[str],
    max_hops: int = 4,
) -> bool:
    if start in targets:
        return True
    visited = {start}
    queue   = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for nb in adj.get(node, set()):
            if nb in targets:
                return True
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, depth + 1))
    return False


def build_adj(triples: Set[Tuple[str, str, str]]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for s, r, o in triples:
        if _is_entity(s) and _is_entity(o):
            adj[s].add(o)
            adj[o].add(s)
    return adj


# ── Load ELQ top-scored seed ──────────────────────────────────────────────────

_elq_cache: Dict[str, Dict[str, Optional[str]]] = {}

def load_elq_top_seed(dataset: str, p_type: str) -> Dict[str, Optional[str]]:
    """Return {qid: top_scored_mid} from ELQ output."""
    key = f"{dataset}_{p_type}"
    if key in _elq_cache:
        return _elq_cache[key]

    path = EL_ELQ_DIR / f"{dataset}_{p_type}_el.json"
    if not path.exists():
        _elq_cache[key] = {}
        return {}

    result: Dict[str, Optional[str]] = {}
    for rec in json.load(open(path)):
        qid    = rec["id"]
        linked = rec.get("linked_entities", [])
        if not linked:
            result[qid] = None
            continue
        top = max(linked, key=lambda e: e.get("score", 0.0))
        result[qid] = top.get("mid") or None

    _elq_cache[key] = result
    return result


# ── Per-question subgraph builder ─────────────────────────────────────────────

def build_subgraph_cwq(
    qid:      str,
    seed_mid: Optional[str],
    cap:      int = CAP_CWQ,
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """UNION cwq_4hop/{qid}.nxhd + new_seed_nxhd/{seed_mid}.nxhd, then PPR-prune."""
    e1, t1 = parse_nxhd(CWQ_4HOP_DIR / f"{qid}.nxhd")
    raw_ents  = set(e1)
    raw_trips = set(t1)

    if seed_mid:
        e2, t2 = parse_nxhd(CWQ_SEED_DIR / f"{seed_mid}.nxhd")
        raw_ents  |= e2
        raw_trips |= t2

    if not raw_ents:
        return raw_ents, raw_trips

    effective_seed = seed_mid if (seed_mid and seed_mid in raw_ents) else next(iter(raw_ents))
    return prune_by_ppr(raw_ents, raw_trips, effective_seed, cap)


def build_subgraph_webqsp(
    qid:      str,
    seed_mid: Optional[str],
    cap:      int = CAP_WEBQSP,
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """UNION webqsp_2hop/{base_qid}.nxhd + webqsp_seed_nxhd/{seed_mid}.nxhd, PPR-prune."""
    base_qid = qid.split("_")[0]
    e1, t1 = parse_nxhd(WQ_2HOP_DIR / f"{base_qid}.nxhd")
    raw_ents  = set(e1)
    raw_trips = set(t1)

    if seed_mid:
        e2, t2 = parse_nxhd(WQ_SEED_DIR / f"{seed_mid}.nxhd")
        raw_ents  |= e2
        raw_trips |= t2

    if not raw_ents:
        return raw_ents, raw_trips

    effective_seed = seed_mid if (seed_mid and seed_mid in raw_ents) else next(iter(raw_ents))
    return prune_by_ppr(raw_ents, raw_trips, effective_seed, cap)


# ── Per-question worker (for multiprocessing) ─────────────────────────────────

def _process_one_question(args_tuple) -> Optional[Dict]:
    """Worker function for multiprocessing.Pool.map — processes a single question.

    Each worker has its own LRU nxhd cache (not shared, but amortised per task).
    """
    (rec, orig_rec, pert_seed, orig_seed, id2mid_list, cap, dataset, max_lines) = args_tuple

    # Restore global NXHD_MAX_LINES in worker process
    global NXHD_MAX_LINES
    NXHD_MAX_LINES = max_lines

    qid = rec["id"]

    ans_mids: Set[str] = set()
    id2mid = {i: m for i, m in enumerate(id2mid_list)}
    for a in orig_rec.get("answers", []):
        kb = a.get("kb_id", "")
        if kb and _is_entity(kb):
            ans_mids.add(kb)

    if pert_seed == orig_seed and pert_seed is not None:
        if dataset == "cwq":
            pert_ents, pert_trips = build_subgraph_cwq(qid, pert_seed, cap)
        else:
            pert_ents, pert_trips = build_subgraph_webqsp(qid, pert_seed, cap)
        orig_ents, orig_trips = pert_ents, pert_trips
        ent_jaccard  = 1.0
        trip_jaccard = 1.0
        delta_sr     = 0.0
        ged          = 0
    else:
        try:
            if dataset == "cwq":
                pert_ents, pert_trips = build_subgraph_cwq(qid, pert_seed, cap)
                orig_ents, orig_trips = build_subgraph_cwq(qid, orig_seed, cap)
            else:
                pert_ents, pert_trips = build_subgraph_webqsp(qid, pert_seed, cap)
                orig_ents, orig_trips = build_subgraph_webqsp(qid, orig_seed, cap)
        except Exception:
            return None

        u_e = len(pert_ents | orig_ents)
        i_e = len(pert_ents & orig_ents)
        ent_jaccard = i_e / u_e if u_e else 1.0

        pert_trip_keys = {(s, r) for s, r, o in pert_trips}
        orig_trip_keys = {(s, r) for s, r, o in orig_trips}
        u_t = len(pert_trip_keys | orig_trip_keys)
        i_t = len(pert_trip_keys & orig_trip_keys)
        trip_jaccard = i_t / u_t if u_t else 1.0

        delta_sr = 1.0 - ent_jaccard
        ged      = len(pert_ents ^ orig_ents)

    ans_in_pert = bool(ans_mids & pert_ents)
    ans_in_orig = bool(ans_mids & orig_ents)

    pert_adj = build_adj(pert_trips)
    orig_adj = build_adj(orig_trips)

    pert_golden = bfs_path_exists(pert_adj, pert_seed, ans_mids) if pert_seed else False
    orig_golden = bfs_path_exists(orig_adj, orig_seed, ans_mids) if orig_seed else False
    golden_retained = pert_golden and orig_golden
    same_seed = (pert_seed == orig_seed) and (pert_seed is not None)

    return {
        "qid":             qid,
        "pert_seed":       pert_seed,
        "orig_seed":       orig_seed,
        "same_seed":       same_seed,
        "pert_n_ents":     len(pert_ents),
        "orig_n_ents":     len(orig_ents),
        "pert_n_trips":    len(pert_trips),
        "orig_n_trips":    len(orig_trips),
        "ent_jaccard":     round(ent_jaccard, 4),
        "trip_jaccard":    round(trip_jaccard, 4),
        "delta_SR":        round(delta_sr, 4),
        "GED":             ged,
        "ans_in_pert":     ans_in_pert,
        "ans_in_orig":     ans_in_orig,
        "pert_golden":     pert_golden,
        "orig_golden":     orig_golden,
        "golden_retained": golden_retained,
    }


# ── Core evaluation per (dataset, p_type) ────────────────────────────────────

def evaluate_one(
    dataset:  str,
    p_type:   str,
    n_sample: Optional[int],
    seed:     int = 42,
    dry_run:  bool = False,
    workers:  int = 1,
) -> Dict:
    rng = random.Random(seed)

    id2mid = load_id2mid(dataset)
    cap    = CAP_CWQ if dataset == "cwq" else CAP_WEBQSP

    ds_path = DATASET_DIR / f"{dataset}_{p_type}.json"
    if not ds_path.exists():
        print(f"  [SKIP] {ds_path} not found")
        return {}

    ds_records = json.load(open(ds_path))

    pert_seeds = load_elq_top_seed(dataset, p_type)
    orig_seeds = load_elq_top_seed(dataset, "orig")

    orig_records = json.load(open(DATASET_DIR / f"{dataset}_orig.json"))
    orig_by_id   = {r["id"]: r for r in orig_records}

    if n_sample and n_sample < len(ds_records):
        sampled = rng.sample(ds_records, n_sample)
    else:
        sampled = ds_records

    if dry_run:
        sampled = sampled[:5]

    # Prepare args for parallel processing
    id2mid_list = [id2mid[i] for i in range(max(id2mid.keys()) + 1) if i in id2mid]
    task_args = []
    for rec in sampled:
        qid      = rec["id"]
        orig_rec = orig_by_id.get(qid, {})
        pert_seed = pert_seeds.get(qid)
        orig_seed = orig_seeds.get(qid)
        task_args.append((rec, orig_rec, pert_seed, orig_seed, id2mid_list, cap,
                          dataset, NXHD_MAX_LINES))

    t0 = time.time()

    if dry_run or workers <= 1:
        # Sequential (dry-run always sequential for clean output)
        results_per_q = []
        for i, args_t in enumerate(task_args):
            res = _process_one_question(args_t)
            if res:
                results_per_q.append(res)
            if dry_run:
                rec, orig_rec, pert_seed, orig_seed = args_t[0], args_t[1], args_t[2], args_t[3]
                qid = rec["id"]
                n_ents = res["pert_n_ents"] if res else 0
                j = res["ent_jaccard"] if res else 0.0
                ans = res["ans_in_pert"] if res else False
                print(f"    [{i+1}] qid={qid[:40]}")
                print(f"         pert_seed={pert_seed}  orig_seed={orig_seed}")
                print(f"         ents={n_ents}  Jaccard={j:.3f}  ans_pert={ans}")
    else:
        # Parallel using multiprocessing.Pool
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            all_results = []
            for i, res in enumerate(pool.imap_unordered(_process_one_question, task_args,
                                                         chunksize=max(1, len(task_args) // (workers * 4)))):
                all_results.append(res)
                if (i + 1) % max(1, len(task_args) // 10) == 0:
                    elapsed = time.time() - t0
                    speed   = (i + 1) / elapsed
                    eta     = (len(task_args) - i - 1) / speed if speed > 0 else 0
                    print(f"    [{i+1}/{len(task_args)}] {elapsed:.0f}s  {speed:.1f}q/s  ETA {eta:.0f}s",
                          flush=True)
        results_per_q = [r for r in all_results if r is not None]

    if not results_per_q:
        return {}

    # Filter out None (failed) results
    results_per_q = [r for r in results_per_q if r is not None]
    if not results_per_q:
        return {}

    def mean(lst): return float(np.mean(lst)) if lst else 0.0

    agg = {
        "dataset":             dataset,
        "p_type":              p_type,
        "n_total":             len(ds_records),
        "n_sample":            len(results_per_q),
        "cap":                 cap,
        "avg_pert_ents":       round(mean([r["pert_n_ents"]  for r in results_per_q]), 1),
        "avg_orig_ents":       round(mean([r["orig_n_ents"]  for r in results_per_q]), 1),
        "avg_pert_trips":      round(mean([r["pert_n_trips"] for r in results_per_q]), 1),
        "avg_orig_trips":      round(mean([r["orig_n_trips"] for r in results_per_q]), 1),
        "ent_jaccard_mean":    round(mean([r["ent_jaccard"]  for r in results_per_q]), 4),
        "trip_jaccard_mean":   round(mean([r["trip_jaccard"] for r in results_per_q]), 4),
        "delta_SR_mean":       round(mean([r["delta_SR"]     for r in results_per_q]), 4),
        "GED_mean":            round(mean([r["GED"]          for r in results_per_q]), 1),
        "ans_cov_pert_pct":    round(100 * sum(r["ans_in_pert"]    for r in results_per_q) / len(results_per_q), 2),
        "ans_cov_orig_pct":    round(100 * sum(r["ans_in_orig"]    for r in results_per_q) / len(results_per_q), 2),
        "pert_golden_pct":     round(100 * sum(r["pert_golden"]    for r in results_per_q) / len(results_per_q), 2),
        "orig_golden_pct":     round(100 * sum(r["orig_golden"]    for r in results_per_q) / len(results_per_q), 2),
        "golden_retained_pct": round(100 * sum(r["golden_retained"] for r in results_per_q) / len(results_per_q), 2),
        "same_seed_pct":       round(100 * sum(r["same_seed"]      for r in results_per_q) / len(results_per_q), 2),
        "samples":             results_per_q,
    }

    elapsed = time.time() - t0
    print(f"  ✓ {dataset} {p_type}: n={len(results_per_q)} "
          f"ent_J={agg['ent_jaccard_mean']:.3f} "
          f"ans_pert={agg['ans_cov_pert_pct']:.1f}% "
          f"golden_ret={agg['golden_retained_pct']:.1f}% "
          f"same_seed={agg['same_seed_pct']:.1f}% "
          f"({elapsed:.0f}s)", flush=True)

    return agg


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",  choices=["cwq", "webqsp"],
                    help="Dataset to run (can combine with --all to run only that dataset's combos)")
    ap.add_argument("--p-type",   choices=["orig", "p1", "p2", "p3", "p4", "p5", "p6"])
    ap.add_argument("--all",      action="store_true",
                    help="Run all p-types (orig+p1-p6). If --dataset given, runs that dataset only.")
    ap.add_argument("--n-sample", type=int, default=None,
                    help="Questions to sample per combo (default: full dataset)")
    ap.add_argument("--full",     action="store_true",
                    help="Run on full dataset (overrides --n-sample)")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--dry-run",   action="store_true",
                    help="Print stats for 5 questions per combo, do not save")
    ap.add_argument("--max-lines", type=int, default=50_000,
                    help="Max nxhd lines to parse per file (default 50000, 0=unlimited)")
    ap.add_argument("--workers",   type=int, default=8,
                    help="Parallel workers (default 8; use 1 for sequential)")
    args = ap.parse_args()

    global NXHD_MAX_LINES
    NXHD_MAX_LINES = args.max_lines if args.max_lines > 0 else None

    n_sample = None if args.full else args.n_sample

    # Determine combos
    p_types  = ["orig", "p1", "p2", "p3", "p4", "p5", "p6"]
    datasets = ["cwq", "webqsp"]

    if args.all:
        if args.dataset:
            datasets = [args.dataset]
        combos = [(ds, pt) for ds in datasets for pt in p_types]
    elif args.dataset and args.p_type:
        combos = [(args.dataset, args.p_type)]
    elif args.dataset and not args.p_type:
        ap.error("Specify --p-type or --all when using --dataset without --all")
    else:
        ap.error("Specify --dataset + --p-type, --dataset --all, or --all")

    print(f"Combos to run: {combos}")
    print(f"Output directory: {OUT_DIR}")

    for dataset, p_type in combos:
        key = f"{dataset}_{p_type}"
        out_path = OUT_DIR / f"{key}.json"

        # Skip already-computed (unless dry-run)
        if out_path.exists() and not args.dry_run:
            print(f"\n── {key} — already done ({out_path.name}), skipping ──")
            continue

        print(f"\n── {key} ──", flush=True)
        result = evaluate_one(dataset, p_type, n_sample, seed=args.seed,
                              dry_run=args.dry_run, workers=args.workers)

        if result and not args.dry_run:
            out_path.write_text(json.dumps(result, indent=2))
            print(f"  Saved → {out_path}")

    if args.dry_run:
        print("\nDry run complete — no files written.")
        return

    # ── Rebuild summary.json ─────────────────────────────────────────────────
    summary: Dict[str, Dict] = {}
    for ds in ["cwq", "webqsp"]:
        for pt in ["orig", "p1", "p2", "p3", "p4", "p5", "p6"]:
            p = OUT_DIR / f"{ds}_{pt}.json"
            if p.exists():
                d = json.loads(p.read_text())
                # Store only aggregate (drop samples list for smaller summary)
                summary[f"{ds}_{pt}"] = {k: v for k, v in d.items() if k != "samples"}

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("  nxhd Subgraph Analysis — PPR-pruned  (output: tmlr/results/subgraph/)")
    print(f"{'='*110}")
    hdr = (f"  {'DS':<8} {'PT':<5} {'n':>5}  {'AvgE':>6}  {'EntJ':>6}  {'TripJ':>6}  "
           f"{'ΔSR':>6}  {'GED':>7}  {'AnsPert':>8}  {'AnsOrig':>8}  "
           f"{'GoldRet':>8}  {'SameSeed':>9}")
    print(hdr)
    print("  " + "-"*107)
    for ds in ["cwq", "webqsp"]:
        for pt in ["orig", "p1", "p2", "p3", "p4", "p5", "p6"]:
            key = f"{ds}_{pt}"
            if key not in summary:
                continue
            r = summary[key]
            print(f"  {r['dataset']:<8} {r['p_type']:<5} {r['n_sample']:>5}  "
                  f"{r['avg_pert_ents']:>6.0f}  "
                  f"{r['ent_jaccard_mean']:>6.3f}  {r['trip_jaccard_mean']:>6.3f}  "
                  f"{r['delta_SR_mean']:>6.3f}  {r['GED_mean']:>7.0f}  "
                  f"{r['ans_cov_pert_pct']:>7.1f}%  {r['ans_cov_orig_pct']:>7.1f}%  "
                  f"{r['golden_retained_pct']:>7.1f}%  {r['same_seed_pct']:>8.1f}%")

    print(f"\nSummary saved → {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()