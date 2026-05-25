#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_p7_graftnet_subgraph.py
==============================
Build GraftNet-style PPR-pruned subgraphs for the p7 (relation-synonym) perturbation.

Since p7 is relation-only perturbation, GEM entity seeds are identical to the
originals. We reuse existing neighborhood caches rather than re-running BFS.

Source retrieval tiers (per GEM seed MID):
  Tier 1 — per-MID .nxhd files (cwq_4hop_nxhd / freebase_2hops/stagg.neighborhoods)
  Tier 2 — PKL batch caches (Pre-process/mid_cache/neighborhoods_{dataset}.pkl)
  Tier 3 — BFS over fb_en.txt (edge cases not covered by T1+T2, ~2-12%)

Pruning:
  CWQ    — GraftNet PPR (exact port, relation-weighted by gold_relation_map, restart=0.8)
  WebQSP — PageRank-Nibble (multi-seed, topology-only, alpha=0.15)

Output (separate from the simple hard-cap build_gnn_test_subgraph.py output):
  data/perturbations/gnn_input/cwq_p7_graftnet/
  data/perturbations/gnn_input/webqsp_p7_graftnet/

Usage:
  python build_p7_graftnet_subgraph.py --dataset both
  python build_p7_graftnet_subgraph.py --dataset cwq --stage 2   # re-run PPR only
  python build_p7_graftnet_subgraph.py --dataset cwq --dry-run
  python build_p7_graftnet_subgraph.py --dataset both --workers 40
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = REPO_ROOT
PERT_DIR      = REPO_ROOT / "tmlr" / "pert" / "dataset"
GNN_DATA      = REPO_ROOT / "tmlr" / "Repo" / "GNN-RAG" / "data"

# Tier 1: nxhd
NXHD_CWQ     = REPO_ROOT / "Pre-process" / "cwq_4hop_nxhd"
NXHD_WEBQSP  = REPO_ROOT / "freebase" / "data" / "dataset" / "freebase_2hops" / "stagg.neighborhoods"

# Tier 2: PKL batch caches
PKL_CWQ      = REPO_ROOT / "Pre-process" / "mid_cache" / "neighborhoods_cwq.pkl"
PKL_WEBQSP   = REPO_ROOT / "Pre-process" / "mid_cache" / "neighborhoods_webqsp.pkl"

# Tier 3: fb_en.txt BFS fallback
DEFAULT_FB   = REPO_ROOT / "freebase" / "data" / "fb_en.txt"

# Per-MID cache dir (same structure as build_subgraphs.py for reuse)
MID_CACHE    = REPO_ROOT / "new_exp" / "mid_cache"

# Intermediate per-question subgraph JSONs
SUBGRAPH_DIR = PERT_DIR / "subgraphs_p7"

# Final GNN-RAG JSONL output
GNN_OUT_BASE = PERT_DIR / "gnn_input"

DATASET_SUBDIR = {"cwq": "CWQ", "webqsp": "webqsp"}

# ── Hyperparameters ────────────────────────────────────────────────────────────
# CWQ: 4-hop BFS caps (used for Tier 3 BFS only — Tier 1/2 already multi-hop)
CWQ_HOPS  = 4
CWQ_CAPS  = [500, 300, 100, 50]
CWQ_CAP   = 2500          # max entities after PPR

# WebQSP: 2-hop
WEBQSP_HOPS = 2
WEBQSP_CAPS = [1000, 500]
WEBQSP_CAP  = 2000

# GraftNet PPR (exact values from GraftNet step4_extract_subgraphs.py)
PPR_RESTART   = 0.8
PPR_MAX_ITER  = 20
PPR_EXPONENT  = 2.0
PPR_NOTFOUND  = 0.0
PPR_DECOMPOSE = True

# PageRank-Nibble (WebQSP)
PRN_ALPHA   = 0.15
PRN_EPSILON = 1e-6

NXHD_MAX_LINES = 5_000_000  # GraftNet uses MAX_FACTS=5M; nxhd files are ≤660K lines
SPARQL_BATCH   = 100

log = logging.getLogger("build_p7")


def setup_logging(log_file: Optional[Path] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_entity(tok: str) -> bool:
    return tok.startswith("m.") or tok.startswith("g.")


def _strip_nxhd(tok: str) -> str:
    t = tok.strip()
    if t.startswith("<fb:") and t.endswith(">"):
        return t[4:-1]
    return t


def _write_atomic(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1: nxhd parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_nxhd(path: Path) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    entities: Set[str] = set()
    triples:  Set[Tuple[str, str, str]] = set()
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh):
                if n >= NXHD_MAX_LINES:
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                s = _strip_nxhd(parts[0])
                r = _strip_nxhd(parts[1])
                o = _strip_nxhd(parts[2])
                if not s or not r:
                    continue
                triples.add((s, r, o))
                if _is_entity(s):
                    entities.add(s)
                if _is_entity(o):
                    entities.add(o)
    except FileNotFoundError:
        pass
    return entities, triples


def load_nxhd_for_qid(qid: str, dataset: str) -> Optional[Tuple[Set[str], Set[Tuple[str,str,str]]]]:
    """
    Try to load Tier 1 nxhd for a question ID.
    CWQ:    cwq_4hop_nxhd/{qid}.nxhd   (full QID including hash suffix)
    WebQSP: stagg.neighborhoods/{base_qid}.nxhd  (base part before '_', e.g. 'WebQTest-0')
    """
    if dataset == "cwq":
        path = NXHD_CWQ / f"{qid}.nxhd"
        if path.exists():
            ents, trips = parse_nxhd(path)
            if ents or trips:
                return ents, trips
        # Perturbed QIDs share the same base as the original nxhd files.
        # CWQ IDs are "WebQTest-832_<hash>" — base is "WebQTest-832", not "WebQTest".
        # Split on the last underscore to get the base: "WebQTest-832_<hash>" → "WebQTest-832"
        parts = qid.rsplit("_", 1)
        base = parts[0] if len(parts) == 2 else qid
        matches = sorted(glob.glob(str(NXHD_CWQ / f"{base}_*.nxhd")))
        for m in matches:
            ents, trips = parse_nxhd(Path(m))
            if ents or trips:
                return ents, trips
    else:
        # WebQSP: strip hash suffix — files named 'WebQTest-123.nxhd'
        base_qid = qid.split("_")[0] if "_" in qid else qid
        path = NXHD_WEBQSP / f"{base_qid}.nxhd"
        if path.exists():
            ents, trips = parse_nxhd(path)
            if ents or trips:
                return ents, trips
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Tier 2: PKL batch cache
# ══════════════════════════════════════════════════════════════════════════════

_pkl_cache: Dict[str, Dict] = {}  # dataset -> loaded pkl dict
_pkl_lock = __import__("threading").Lock()


def load_pkl(dataset: str) -> Dict:
    if dataset in _pkl_cache:
        return _pkl_cache[dataset]
    with _pkl_lock:
        if dataset not in _pkl_cache:
            path = PKL_CWQ if dataset == "cwq" else PKL_WEBQSP
            log.info(f"  Loading PKL {path.name} …")
            t0 = time.time()
            with open(path, "rb") as f:
                _pkl_cache[dataset] = pickle.load(f)
            log.info(f"  PKL loaded: {len(_pkl_cache[dataset]):,} entries in {time.time()-t0:.1f}s")
    return _pkl_cache[dataset]


def load_pkl_for_mid(mid: str, dataset: str) -> Optional[Tuple[Set[str], Set[Tuple[str,str,str]]]]:
    """Try Tier 2 PKL lookup for a single MID."""
    pkl = load_pkl(dataset)
    entry = pkl.get(mid)
    if not entry:
        return None
    ents   = set(entry.get("entities", []))
    tuples = entry.get("tuples", [])
    trips: Set[Tuple[str,str,str]] = set()
    for t in tuples:
        if len(t) >= 3:
            trips.add((str(t[0]), str(t[1]), str(t[2])))
    if ents or trips:
        return ents, trips
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Tier 3: fb_en.txt BFS (fallback for edge cases)
# ══════════════════════════════════════════════════════════════════════════════

_fb_fwd: Dict[str, List[Tuple[str,str]]] = {}
_fb_bwd: Dict[str, List[Tuple[str,str]]] = {}
_fb_loaded = False


def load_fb_index(fb_file: Path) -> None:
    global _fb_fwd, _fb_bwd, _fb_loaded
    if _fb_loaded:
        return
    log.info(f"Loading fb_en.txt index from {fb_file} (this takes ~3 min, ~100GB RAM) …")
    t0 = time.time()
    fwd: Dict[str, List] = defaultdict(list)
    bwd: Dict[str, List] = defaultdict(list)
    n = 0
    with open(fb_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            s, p, o = parts[0], parts[1], parts[2]
            if not s or not p or not o:
                continue
            fwd[s].append((p, o))
            if _is_entity(o):
                bwd[o].append((p, s))
            n += 1
            if n % 50_000_000 == 0:
                log.info(f"  {n/1e6:.0f}M lines, {time.time()-t0:.0f}s")
    _fb_fwd = dict(fwd)
    _fb_bwd = dict(bwd)
    _fb_loaded = True
    log.info(f"  fb index ready: {len(_fb_fwd):,} subjects in {time.time()-t0:.1f}s")


def bfs_fb(
    seed_mid: str,
    max_hops: int,
    caps: List[int],
) -> Tuple[Set[str], Set[Tuple[str,str,str]]]:
    """Degree-sorted BFS over in-memory fb_en.txt fwd/bwd dicts."""
    all_ents:    Set[str]                   = {seed_mid}
    all_triples: Set[Tuple[str,str,str]]    = set()
    frontier:    Set[str]                   = {seed_mid}

    for hop in range(max_hops):
        cap = caps[hop] if hop < len(caps) else 50
        new_deg:  Dict[str, int] = defaultdict(int)
        hop_trips: Set[Tuple[str,str,str]] = set()

        for entity in frontier:
            for pred, obj in _fb_fwd.get(entity, []):
                hop_trips.add((entity, pred, obj))
                if _is_entity(obj) and obj not in all_ents:
                    new_deg[obj] += 1
            for pred, subj in _fb_bwd.get(entity, []):
                hop_trips.add((subj, pred, entity))
                if _is_entity(subj) and subj not in all_ents:
                    new_deg[subj] += 1

        new_sorted = sorted(new_deg, key=lambda e: -new_deg[e])
        new_frontier = set(new_sorted[:cap])

        all_triples |= hop_trips
        all_ents    |= new_frontier
        frontier     = new_frontier

    return all_ents, all_triples


# ══════════════════════════════════════════════════════════════════════════════
# MID subgraph retrieval (Tier 2+3 only — per MID)
# ══════════════════════════════════════════════════════════════════════════════

def get_mid_subgraph_t2t3(
    mid: str,
    dataset: str,
    max_hops: int,
    caps: List[int],
    use_fb: bool = False,
) -> Tuple[Set[str], Set[Tuple[str,str,str]], int]:
    """
    Tier 2 (PKL) → Tier 3 (fb BFS) lookup for a single MID.
    Tier 1 (nxhd) is QID-based and handled by stage1_collect_records.
    Returns (entities, triples, tier_used).
    """
    result = load_pkl_for_mid(mid, dataset)
    if result:
        return result[0], result[1], 2

    if use_fb and _fb_loaded:
        ents, trips = bfs_fb(mid, max_hops, caps)
        return ents, trips, 3

    return set(), set(), 0


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: collect raw triples per question (QID-first, then per-MID fallback)
# ══════════════════════════════════════════════════════════════════════════════

def stage1_collect_records(
    dataset: str,
    records: List[dict],
    max_hops: int,
    caps: List[int],
    use_fb: bool,
    n_workers: int,
) -> Dict[str, Tuple[Set[str], Set[Tuple[str,str,str]], int]]:
    """
    For each record, retrieve raw triples matching GraftNet's source strategy:
      Tier 1: QID-based nxhd file — the primary source GraftNet uses.
              Reads up to NXHD_MAX_LINES (5M) facts per question.
      Tier 2: Per-MID PKL cache — fallback when nxhd file is missing (~0.2% WebQSP).
      Tier 3: fb_en.txt BFS — only if --use-fb and MID not in PKL.

    Returns qid -> (entities, triples, tier_used).
    tier_used: 1=nxhd, 2=pkl, 3=fb, 0=empty
    """
    log.info(f"[{dataset}] Stage 1: {len(records)} records to retrieve")

    # Pass 1: Tier 1 — nxhd QID lookup (I/O, sequential is fine for ~3531 files)
    qid_data: Dict[str, Tuple[Set, Set, int]] = {}
    t1_hits = 0
    for rec in records:
        qid = rec["id"]
        result = load_nxhd_for_qid(qid, dataset)
        if result:
            qid_data[qid] = (result[0], result[1], 1)
            t1_hits += 1
        else:
            qid_data[qid] = (set(), set(), 0)  # placeholder for Tier 2

    log.info(f"  [{dataset}] Tier 1 (nxhd QID): {t1_hits}/{len(records)} hits")

    # Pass 2: Tier 2/3 — PKL/fb for records without nxhd
    missing = [rec for rec in records if qid_data[rec["id"]][2] == 0]
    log.info(f"  [{dataset}] Tier 2/3 fallback: {len(missing)} records without nxhd")

    if missing:
        all_mids: Set[str] = set()
        for rec in missing:
            gem = rec.get("gold_entity_map") or {}
            for m in gem.keys():
                if m and not m.startswith("g."):
                    all_mids.add(m.strip())

        mid_cache: Dict[str, Tuple[Set, Set, int]] = {}
        done_mid = [0]
        t2_counts: Dict[int, int] = {}

        def _work_mid(mid: str):
            return mid, *get_mid_subgraph_t2t3(mid, dataset, max_hops, caps, use_fb)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_work_mid, m): m for m in sorted(all_mids)}
            for fut in as_completed(futs):
                mid, ents, trips, tier = fut.result()
                mid_cache[mid] = (ents, trips, tier)
                t2_counts[tier] = t2_counts.get(tier, 0) + 1
                done_mid[0] += 1
                if done_mid[0] % 200 == 0 or done_mid[0] == len(all_mids):
                    log.info(f"  [{dataset}] PKL: {done_mid[0]}/{len(all_mids)} MIDs | "
                             f"T2={t2_counts.get(2,0)} T3={t2_counts.get(3,0)} "
                             f"T0={t2_counts.get(0,0)}")

        for rec in missing:
            qid = rec["id"]
            gem = rec.get("gold_entity_map") or {}
            seeds = [m.strip() for m in gem.keys() if m and not m.startswith("g.")]
            all_ents: Set[str] = set()
            all_trips: Set[Tuple] = set()
            best_tier = 0
            for mid in seeds:
                if mid in mid_cache:
                    e, t, tier = mid_cache[mid]
                    all_ents |= e
                    all_trips |= t
                    if tier > 0 and (best_tier == 0 or tier < best_tier):
                        best_tier = tier
            qid_data[qid] = (all_ents, all_trips, best_tier)

    tier_summary: Dict[int, int] = {}
    for _, (_, _, t) in qid_data.items():
        tier_summary[t] = tier_summary.get(t, 0) + 1
    log.info(f"[{dataset}] Stage 1 done — tier distribution: {tier_summary}")
    return qid_data


# ══════════════════════════════════════════════════════════════════════════════
# GraftNet PPR (CWQ) — exact port
# ══════════════════════════════════════════════════════════════════════════════

def _graftnet_read_facts(
    triples: Set[Tuple[str,str,str]],
    gold_rels: Set[str],
) -> Tuple[Dict[str,int], Dict, Optional[csr_matrix]]:
    entity_map: Dict[str,int] = {}
    relation_map: Dict[str, List[List[int]]] = {}
    num_entities = 0

    for s, r, o in triples:
        if not _is_entity(s) or not _is_entity(o):
            continue
        if s not in entity_map:
            entity_map[s] = num_entities; num_entities += 1
        if o not in entity_map:
            entity_map[o] = num_entities; num_entities += 1
        if r not in relation_map:
            relation_map[r] = [[], []]
        relation_map[r][0].append(entity_map[s])
        relation_map[r][1].append(entity_map[o])

    if not relation_map:
        return {}, {}, None

    n = num_entities
    # Build symmetrised (undirected) per-relation matrices so PPR can propagate
    # in both directions. Matching GraftNet's DECOMPOSE_PPV=True but adding
    # reverse edges so seed nodes with only incoming gold-rel triples still spread.
    per_rel: Dict[str, csr_matrix] = {}
    for rel, (rows, cols) in relation_map.items():
        # Forward + reverse edges for bidirectional spread
        all_rows = np.array(rows + cols, dtype=np.int32)
        all_cols = np.array(cols + rows, dtype=np.int32)
        m = csr_matrix(
            (np.ones(len(all_rows)), (all_rows, all_cols)),
            shape=(n, n),
        )
        m = normalize(m, norm="l1", axis=1)
        score = 1.0 if rel in gold_rels else PPR_NOTFOUND
        per_rel[rel] = m * (score ** PPR_EXPONENT)

    adj_mat = sum(per_rel.values()) / len(per_rel)
    adj_mat = normalize(adj_mat, norm="l1", axis=1)

    # Fallback: if gold-relation weighting collapses the graph (nnz too small
    # relative to entity count), retry with topology-only (all rels score=1.0).
    if adj_mat.nnz < max(2, n // 100):
        per_rel_topo: Dict[str, csr_matrix] = {}
        for rel, (rows, cols) in relation_map.items():
            all_rows = np.array(rows + cols, dtype=np.int32)
            all_cols = np.array(cols + rows, dtype=np.int32)
            m = csr_matrix(
                (np.ones(len(all_rows)), (all_rows, all_cols)),
                shape=(n, n),
            )
            m = normalize(m, norm="l1", axis=1)
            per_rel_topo[rel] = m
        adj_mat = sum(per_rel_topo.values()) / len(per_rel_topo)
        adj_mat = normalize(adj_mat, norm="l1", axis=1)

    return entity_map, per_rel, adj_mat


def _graftnet_ppr(seed: np.ndarray, W: csr_matrix) -> np.ndarray:
    restart_prob = PPR_RESTART
    r = restart_prob * seed
    s_ovr = np.copy(r)
    for _ in range(PPR_MAX_ITER):
        r_new = (1. - restart_prob) * W.T.dot(r)
        s_ovr += r_new
        if abs(r_new.sum()) < 1e-5:
            break
        r = r_new
    return np.squeeze(s_ovr)


def graftnet_ppr(
    triples: Set[Tuple[str,str,str]],
    seed_mids: List[str],
    gold_rels: Set[str],
    top_m: int,
) -> List[str]:
    entity_map, _, adj_mat = _graftnet_read_facts(triples, gold_rels)
    if adj_mat is None or not entity_map:
        return list(seed_mids)

    n = adj_mat.shape[0]
    seed_indices = [entity_map[m] for m in seed_mids if m in entity_map]
    seed = np.zeros((n, 1))
    if seed_indices:
        seed[seed_indices] = np.expand_dims(
            np.arange(len(seed_indices), 0, -1, dtype=float), axis=1
        )
        seed /= seed.sum()

    ppr = _graftnet_ppr(seed, adj_mat)
    sorted_idx = np.argsort(ppr)[::-1]
    extracted = sorted_idx[:top_m]

    # drop near-zero PPR entries
    zero_idx = np.where(ppr[extracted] < 1e-6)[0]
    if zero_idx.shape[0] > 0:
        extracted = extracted[:zero_idx[0]]

    inv_map = {i: e for e, i in entity_map.items()}
    return [inv_map[i] for i in extracted if i in inv_map]


# ══════════════════════════════════════════════════════════════════════════════
# PageRank-Nibble (WebQSP)
# ══════════════════════════════════════════════════════════════════════════════

def pagerank_nibble(
    triples: Set[Tuple[str,str,str]],
    seed_nodes: List[str],
    top_m: int,
    alpha: float = PRN_ALPHA,
    epsilon: float = PRN_EPSILON,
) -> List[str]:
    # Build undirected adjacency from entity-entity triples only
    adj: Dict[str, List[str]] = defaultdict(list)
    for s, _, o in triples:
        if _is_entity(s) and _is_entity(o):
            adj[s].append(o)
            adj[o].append(s)
    for s in seed_nodes:
        if s not in adj:
            adj[s] = []

    all_nodes = list(adj.keys())
    if not all_nodes:
        return seed_nodes[:]

    pr: Dict[str, float] = defaultdict(float)
    seed_weight = 1.0 / len(seed_nodes) if seed_nodes else 1.0
    for s in seed_nodes:
        pr[s] = seed_weight

    for _ in range(100):
        new_pr: Dict[str, float] = defaultdict(float)
        for node in all_nodes:
            deg = len(adj[node])
            if deg == 0:
                continue
            share = pr[node] / deg
            for nb in adj[node]:
                new_pr[nb] += (1 - alpha) * share
        for s in seed_nodes:
            new_pr[s] += alpha * seed_weight
        delta = sum(abs(new_pr[n] - pr[n]) for n in all_nodes)
        pr = new_pr
        if delta < epsilon:
            break

    sorted_nodes = sorted(pr.items(), key=lambda x: -x[1])
    return [node for node, _ in sorted_nodes[:top_m]]


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: per-question PPR pruning
# ══════════════════════════════════════════════════════════════════════════════

# Worker-process globals
_g_mid_data: Dict = {}
_g_dataset: str = ""
_g_cap: int = 0
_g_out_dir: Path = Path(".")
_g_resume: bool = True


def _init_stage2(mid_data, dataset, cap, out_dir, resume):
    global _g_mid_data, _g_dataset, _g_cap, _g_out_dir, _g_resume
    _g_mid_data = mid_data
    _g_dataset  = dataset
    _g_cap      = cap
    _g_out_dir  = Path(out_dir)
    _g_resume   = resume


def _build_one(record: dict) -> Tuple[str, str]:
    rid      = record["id"]
    out_path = _g_out_dir / f"{rid}.json"

    if _g_resume and out_path.exists():
        return rid, "skipped"

    gem      = record.get("gold_entity_map") or {}
    seeds    = [m.strip() for m in gem.keys() if m and not m.startswith("g.")]
    gold_rels = set((record.get("gold_relation_map") or {}).keys())

    # Retrieve raw triples: qid_data is keyed by question ID
    entry = _g_mid_data.get(rid)
    if entry:
        all_ents, all_triples, _ = entry
    else:
        all_ents    = set()
        all_triples = set()

    if not all_ents and not all_triples:
        out = {
            "id": rid, "question": record.get("perturbed_question", ""),
            "original_question": record.get("original_question", ""),
            "answers": record.get("answers", []),
            "seed_mids": seeds, "entities": seeds, "triples": [],
        }
        _write_atomic(out, out_path)
        return rid, "empty"

    # PPR pruning
    if _g_dataset == "cwq":
        kept_ents = graftnet_ppr(all_triples, seeds, gold_rels, top_m=_g_cap)
    else:
        kept_ents = pagerank_nibble(all_triples, seeds, top_m=_g_cap)

    kept_set = set(kept_ents) | {s for s in seeds if s}

    kept_trips = [
        [s, r, o] for s, r, o in all_triples
        if s in kept_set and o in kept_set
    ]

    out = {
        "id": rid,
        "question":          record.get("perturbed_question", ""),
        "original_question": record.get("original_question", ""),
        "answers":           record.get("answers", []),
        "seed_mids":         seeds,
        "entities":          sorted(kept_set),
        "triples":           kept_trips,
    }
    _write_atomic(out, out_path)
    return rid, "done"


def stage2_ppr(
    dataset: str,
    records: List[dict],
    qid_data: Dict,
    n_workers: int,
    resume: bool,
    out_root: Optional[Path] = None,
) -> None:
    out_dir = (out_root if out_root else SUBGRAPH_DIR) / f"{dataset}_p7"
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = CWQ_CAP if dataset == "cwq" else WEBQSP_CAP

    pending = records
    if resume:
        pending = [r for r in records if not (out_dir / f"{r['id']}.json").exists()]
    log.info(f"[{dataset}] Stage 2: {len(pending)}/{len(records)} questions to process (PPR)")

    if not pending:
        return

    done_n = [0]
    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_stage2,
        initargs=(qid_data, dataset, cap, str(out_dir), resume),
    ) as pool:
        futs = {pool.submit(_build_one, r): r["id"] for r in pending}
        for fut in as_completed(futs):
            rid = futs[fut]
            try:
                _, status = fut.result()
            except Exception as e:
                log.warning(f"  {rid}: ERROR {e}")
                status = "error"
            done_n[0] += 1
            if done_n[0] % 100 == 0 or done_n[0] == len(pending):
                elapsed = time.time() - t0
                rate = done_n[0] / max(elapsed, 1e-9)
                eta  = (len(pending) - done_n[0]) / max(rate, 1e-9)
                log.info(f"  [{dataset}] {done_n[0]}/{len(pending)} | {rate:.1f}/s | ETA {eta:.0f}s")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: GNN-RAG integer encoding
# ══════════════════════════════════════════════════════════════════════════════

def stage3_gnn_encode(dataset: str, records: List[dict], out_root: Optional[Path] = None) -> None:
    subgraph_dir = (out_root if out_root else SUBGRAPH_DIR) / f"{dataset}_p7"
    gnn_out_dir  = (out_root / "gnn_input" if out_root else GNN_OUT_BASE) / f"{dataset}_p7_graftnet"
    gnn_out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(subgraph_dir.glob("*.json"))
    log.info(f"[{dataset}] Stage 3: encoding {len(json_files)} subgraph files")

    # Load GNN-RAG baseline vocab (entities.txt / relations.txt).
    # Using the existing vocab as the integer-ID baseline ensures compatibility
    # with the trained ReaRev model weights. Any new MIDs/rels from the
    # GraftNet subgraph that are NOT in the baseline are appended at the end.
    subdir = DATASET_SUBDIR[dataset]
    base_ent_path = GNN_DATA / subdir / "entities.txt"
    base_rel_path = GNN_DATA / subdir / "relations.txt"

    log.info(f"  [{dataset}] loading baseline vocab from {base_ent_path.parent}")
    ent2id: Dict[str, int] = {}
    with open(base_ent_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            mid = line.strip()
            if mid:
                ent2id[mid] = i
    rel2id: Dict[str, int] = {}
    with open(base_rel_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rel = line.strip()
            if rel:
                rel2id[rel] = i
    log.info(f"  [{dataset}] baseline vocab: {len(ent2id):,} entities, {len(rel2id):,} relations")

    # Single pass: collect new MIDs/rels not in baseline vocab
    new_ents: Set[str] = set()
    new_rels: Set[str] = set()
    for jp in json_files:
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
            for e in d.get("entities", []):
                if e and e not in ent2id:
                    new_ents.add(e)
            for t in d.get("triples", []):
                if len(t) < 3:
                    continue
                s, r, o = t[0], t[1], t[2]
                if s and s not in ent2id: new_ents.add(s)
                if o and o not in ent2id: new_ents.add(o)
                if r and r not in rel2id: new_rels.add(r)
        except Exception:
            pass

    # Append new entries to vocab (sorted for determinism)
    base_ent_count = len(ent2id)
    base_rel_count = len(rel2id)
    for i, e in enumerate(sorted(new_ents)):
        ent2id[e] = base_ent_count + i
    for i, r in enumerate(sorted(new_rels)):
        rel2id[r] = base_rel_count + i
    log.info(f"  [{dataset}] final vocab: {len(ent2id):,} entities "
             f"(+{len(new_ents)} new), {len(rel2id):,} relations (+{len(new_rels)} new)")

    # Write extended vocab files
    ent_list = [""] * len(ent2id)
    for m, i in ent2id.items():
        ent_list[i] = m
    rel_list = [""] * len(rel2id)
    for r, i in rel2id.items():
        rel_list[i] = r
    (gnn_out_dir / "entities.txt").write_text("\n".join(ent_list) + "\n", encoding="utf-8")
    (gnn_out_dir / "relations.txt").write_text("\n".join(rel_list) + "\n", encoding="utf-8")

    # Build record index for ordering output by original record order
    rec_by_id = {r["id"]: r for r in records}

    # Pass 2: encode and write JSONL in original record order
    out_path = gnn_out_dir / "test_subgraph.json"
    n_written = 0
    n_empty   = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in records:
            rid      = rec["id"]
            jp       = subgraph_dir / f"{rid}.json"
            if not jp.exists():
                log.debug(f"  missing subgraph for {rid}, skipping")
                continue
            try:
                d = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue

            seed_ids    = [ent2id[m] for m in d.get("seed_mids", []) if m in ent2id]
            all_ent_ids = sorted(set(ent2id[e] for e in d.get("entities", []) if e in ent2id))

            tuples = []
            for t in d.get("triples", []):
                if len(t) < 3:
                    continue
                s, r, o = str(t[0]), str(t[1]), str(t[2])
                if s in ent2id and r in rel2id and o in ent2id:
                    tuples.append([ent2id[s], rel2id[r], ent2id[o]])

            answers = rec.get("answers", [])

            out_rec = {
                "id":               rid,
                "question":         rec.get("perturbed_question", ""),
                "question_orig":    rec.get("original_question", ""),
                "answers":          answers,
                "entities":         seed_ids,
                "subgraph":         {"tuples": tuples, "entities": all_ent_ids},
                "perturbation_type": "p7",
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_written += 1
            if not tuples:
                n_empty += 1

    log.info(f"  [{dataset}] wrote {n_written} records ({n_empty} with empty subgraph) → {out_path}")

    # Symlinks for vocab/embedding files (reuse pert embeddings)
    subdir = DATASET_SUBDIR[dataset]
    emb_dir = PERT_DIR / "embeddings"
    for src, fname in [
        (GNN_DATA / subdir / "entities.txt",    None),   # overwritten above
        (GNN_DATA / subdir / "relations.txt",   None),   # overwritten above
        (emb_dir / f"{dataset}_pert_vocab.txt", "vocab.txt"),
        (emb_dir / f"{dataset}_pert_word_emb.npy", "word_emb.npy"),
    ]:
        if fname is None:
            continue
        dst = gnn_out_dir / fname
        if not dst.exists() and src.exists():
            os.symlink(src.resolve(), dst)

    log.info(f"  [{dataset}] output dir: {gnn_out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Quality report
# ══════════════════════════════════════════════════════════════════════════════

def quality_report(dataset: str, n: int = 5, out_root: Optional[Path] = None) -> None:
    subgraph_dir = (out_root if out_root else SUBGRAPH_DIR) / f"{dataset}_p7"
    files = sorted(subgraph_dir.glob("*.json"))[:n]
    if not files:
        log.info(f"[{dataset}] No subgraph files for quality report")
        return

    log.info(f"\n{'='*60}")
    log.info(f"Quality report: {dataset} p7 ({len(files)} records)")
    log.info(f"{'='*60}")

    for jp in files:
        d     = json.loads(jp.read_text())
        ents  = set(d.get("entities", []))
        trips = [(t[0],t[1],t[2]) for t in d.get("triples",[]) if len(t)>=3]
        seeds = d.get("seed_mids", [])
        ans   = d.get("answers", [])
        ans_mids = [a.get("kb_id","") if isinstance(a,dict) else a for a in ans]
        ans_in = sum(1 for a in ans_mids if a in ents)
        log.info(
            f"  {d['id'][:55]:55s} | ents={len(ents):5d} | trips={len(trips):6d} | "
            f"ans={ans_in}/{len(ans_mids)} | seeds={len(seeds)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description="Build GraftNet-pruned p7 subgraphs")
    ap.add_argument("--dataset",  choices=["cwq","webqsp","both"], default="both")
    ap.add_argument("--stage",    choices=["1","2","3","all"],     default="all")
    ap.add_argument("--workers",  type=int, default=40,  help="Workers for Stage 2 PPR")
    ap.add_argument("--workers-s1", type=int, default=20, help="Threads for Stage 1 MID retrieval")
    ap.add_argument("--fb-file",  type=Path, default=DEFAULT_FB,
                    help="Path to fb_en.txt (Tier 3, only loaded if needed)")
    ap.add_argument("--use-fb",   action="store_true",
                    help="Load fb_en.txt for Tier 3 fallback (slow, ~3 min, ~100GB RAM)")
    ap.add_argument("--dry-run",  action="store_true", help="Process 5 records per dataset only")
    ap.add_argument("--no-resume",action="store_true", help="Re-process even if output exists")
    ap.add_argument("--out-dir",  type=str, default="",
                    help="Root output directory. Subdirs: <out-dir>/<dataset>_p7/ (stage 2 JSONs) "
                         "and <out-dir>/gnn_input/<dataset>_p7_graftnet/ (stage 3 JSONL). "
                         "Defaults to data/perturbations/subgraphs_p7/ and gnn_input/.")
    return ap.parse_args()


def main():
    args     = parse_args()
    resume   = not args.no_resume
    stages   = ["1","2","3"] if args.stage == "all" else [args.stage]
    out_root = Path(args.out_dir) if args.out_dir else None

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_dir = REPO_ROOT / "tmlr" / "logs"
    setup_logging(log_dir / f"build_p7_graftnet_{ts}.log")
    log.info(f"Starting: dataset={args.dataset} stage={args.stage} dry_run={args.dry_run}")

    datasets = ["cwq","webqsp"] if args.dataset == "both" else [args.dataset]

    # Tier 3: only load fb_en.txt if explicitly requested
    if "1" in stages and args.use_fb:
        if args.fb_file.exists():
            load_fb_index(args.fb_file)
        else:
            log.warning(f"fb_file not found: {args.fb_file} — Tier 3 unavailable")

    for dataset in datasets:
        src_path = PERT_DIR / f"{dataset}_p7.json"
        if not src_path.exists():
            log.warning(f"Dataset not found: {src_path}")
            continue

        records = json.loads(src_path.read_text(encoding="utf-8"))
        if args.dry_run:
            records = records[:5]
            log.info(f"[{dataset}] Dry-run: {len(records)} records")
        else:
            log.info(f"[{dataset}] Loaded {len(records)} records")

        hops = CWQ_HOPS   if dataset == "cwq" else WEBQSP_HOPS
        caps = CWQ_CAPS   if dataset == "cwq" else WEBQSP_CAPS

        qid_data: Dict = {}

        if "1" in stages:
            log.info(f"\n{'─'*50}")
            log.info(f"[{dataset}] === Stage 1: subgraph retrieval (nxhd QID → PKL MID → fb_en) ===")
            t0 = time.time()
            qid_data = stage1_collect_records(
                dataset, records, hops, caps,
                use_fb=args.use_fb,
                n_workers=args.workers_s1,
            )
            log.info(f"[{dataset}] Stage 1 done in {time.time()-t0:.1f}s")

        if "2" in stages:
            log.info(f"\n{'─'*50}")
            log.info(f"[{dataset}] === Stage 2: PPR pruning ===")
            t0 = time.time()
            if not qid_data and "1" not in stages:
                log.warning(f"  Stage 1 was skipped — qid_data empty; Stage 2 will produce empty subgraphs")
            stage2_ppr(dataset, records, qid_data, args.workers, resume, out_root)
            log.info(f"[{dataset}] Stage 2 done in {time.time()-t0:.1f}s")

        if "3" in stages:
            log.info(f"\n{'─'*50}")
            log.info(f"[{dataset}] === Stage 3: GNN-RAG encoding ===")
            t0 = time.time()
            stage3_gnn_encode(dataset, records, out_root)
            log.info(f"[{dataset}] Stage 3 done in {time.time()-t0:.1f}s")

        if args.dry_run:
            quality_report(dataset, out_root=out_root)

    log.info("\nAll done.")
    for dataset in datasets:
        out = GNN_OUT_BASE / f"{dataset}_p7_graftnet" / "test_subgraph.json"
        if out.exists():
            n = sum(1 for _ in open(out))
            log.info(f"  {out}: {n} lines")


if __name__ == "__main__":
    main()
