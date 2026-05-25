"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_subgraphs.py
==================
3-stage pipeline to build KG subgraphs for new_exp/ perturbed datasets.

Stage 1: Extract per-MID subgraphs → new_exp/mid_cache/{dataset}/{mid}.pkl
  Tier 1 – nxhd file cache (already on disk, instant)
  Tier 2 – in-memory BFS over fb_en.txt fwd/bwd index (O(1) lookups)
           + Virtuoso SPARQL for incoming triples (iri_id_ filtered)

Stage 2: Per-question PPR pruning → new_exp/subgraphs/{dataset}_p1/{id}.json
  CWQ:    GraftNet-style relation-weighted PPR (uses gold_relation_map)
  WebQSP: Standard PageRank-Nibble (multi-seed, topology-only)

Stage 3: GNN-RAG JSONL conversion → new_exp/gnn_input/{dataset}_p1/
  Build entities.txt + relations.txt vocab files
  Integer-encode and write test_subgraph.json (one JSON line per record)

Usage:
  python build_subgraphs.py --dataset both --stage all --dry-run
  python build_subgraphs.py --dataset cwq  --stage 1 --workers-s1 20
  bash run_build_subgraphs.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = REPO_ROOT
NEW_EXP       = REPO_ROOT / "new_exp"
DATASET_DIR   = NEW_EXP / "dataset"
MID_CACHE_DIR = NEW_EXP / "mid_cache"
SUBGRAPH_DIR  = NEW_EXP / "subgraphs"
GNN_DIR       = NEW_EXP / "gnn_input"
LOG_DIR       = NEW_EXP / "logs"

NXHD_CWQ     = REPO_ROOT / "Pre-process" / "new_seed_nxhd"
NXHD_WEBQSP  = REPO_ROOT / "Pre-process" / "webqsp_seed_nxhd"

DEFAULT_FB_FILE  = REPO_ROOT / "freebase" / "data" / "fb_en.txt"
DEFAULT_TSV_FILE = REPO_ROOT / "tmlr" / "resources" / "entities_sup_names.tsv"
SPARQL_EP        = "http://127.0.0.1:3001/sparql"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
CWQ_HOPS      = 4
CWQ_CAPS      = [500, 300, 100, 50]
CWQ_CAP       = 2500

WEBQSP_HOPS   = 2
WEBQSP_CAPS   = [1000, 500]
WEBQSP_CAP    = 2000

# GraftNet PPR (CWQ) — exact values from step4_extract_subgraphs_2000.py
PPR_RESTART    = 0.8
PPR_MAX_ITER   = 20
PPR_EXPONENT   = 2.0   # power applied to relation score before weighting
PPR_NOTFOUND   = 0.0   # score for relations not in gold_relation_map
PPR_DECOMPOSE  = True  # average per-relation matrices (DECOMPOSE_PPV=True)

# PageRank-Nibble (WebQSP)
PRN_ALPHA      = 0.15
PRN_EPSILON    = 1e-6

SPARQL_BATCH   = 100    # entities per Virtuoso VALUES clause
NXHD_MAX_LINES = 100_000

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("build_subgraphs")


def setup_logging(log_file: Optional[Path] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _is_entity(token: str) -> bool:
    return token.startswith("m.") or token.startswith("g.")


def _strip_nxhd(token: str) -> str:
    t = token.strip()
    if t.startswith("<fb:") and t.endswith(">"):
        return t[4:-1]
    return t


def run_sparql(query: str, retries: int = 3, timeout: int = 30) -> List[dict]:
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    url    = f"{SPARQL_EP}?{params}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            rows = []
            for b in data.get("results", {}).get("bindings", []):
                row = {}
                for var, info in b.items():
                    val = info.get("value", "")
                    val = val.replace("http://rdf.freebase.com/ns/", "")
                    row[var] = val
                rows.append(row)
            return rows
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.3)
    return []


def load_tsv(path: Path) -> Dict[str, str]:
    log.info(f"Loading TSV {path} …")
    t0  = time.time()
    out = {}
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    mid  = parts[0].strip().lstrip("/").replace("/", ".")
                    name = parts[1].strip()
                    if mid and name:
                        out[mid] = name
    except FileNotFoundError:
        log.warning(f"TSV not found: {path}")
    log.info(f"  Loaded {len(out):,} TSV entries in {time.time()-t0:.1f}s")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 – MID subgraph extraction
# ═══════════════════════════════════════════════════════════════════════════════

# ── fb_en.txt in-memory index ─────────────────────────────────────────────────

def load_fb_index(fb_file: Path) -> Tuple[Dict, Dict]:
    """
    Stream fb_file and build:
      fwd[subject]  = [(pred, obj), ...]
      bwd[object]   = [(pred, subj), ...]   (only when object is an entity MID)
    Returns (fwd_index, bwd_index).
    ~3 min for 920M lines, ~100-120GB RAM.
    """
    log.info(f"Loading fb index from {fb_file} …")
    t0  = time.time()
    fwd: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    bwd: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    n   = 0
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
                log.info(f"  {n/1e6:.0f}M lines, elapsed {time.time()-t0:.0f}s")
    log.info(f"  fb index: {len(fwd):,} subjects, {len(bwd):,} objects in {time.time()-t0:.1f}s")
    return dict(fwd), dict(bwd)


# ── nxhd parser ──────────────────────────────────────────────────────────────

def parse_nxhd(path: Path) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    entities: Set[str] = set()
    triples:  Set[Tuple[str, str, str]] = set()
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for n_lines, line in enumerate(fh):
                if n_lines >= NXHD_MAX_LINES:
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


# ── BFS over in-memory fb index ───────────────────────────────────────────────

def hop_bfs_fb(
    seed_mid: str,
    max_hops: int,
    caps: List[int],
    fwd: Dict,
    bwd: Dict,
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """Iterative BFS using in-memory fwd/bwd dicts."""
    all_ents:   Set[str]                      = {seed_mid}
    all_triples: Set[Tuple[str, str, str]]    = set()
    frontier:   Set[str]                      = {seed_mid}

    for hop in range(max_hops):
        cap       = caps[hop] if hop < len(caps) else 50
        new_deg:  Dict[str, int] = defaultdict(int)
        hop_trips: Set[Tuple[str, str, str]]  = set()

        for entity in frontier:
            # outgoing (entity is subject)
            for pred, obj in fwd.get(entity, []):
                hop_trips.add((entity, pred, obj))
                if _is_entity(obj) and obj not in all_ents:
                    new_deg[obj] += 1

            # incoming (entity is object) — fb_en.txt bwd
            for pred, subj in bwd.get(entity, []):
                hop_trips.add((subj, pred, entity))
                if _is_entity(subj) and subj not in all_ents:
                    new_deg[subj] += 1

        # degree-sort → keep top-cap new entities
        new_sorted = sorted(new_deg, key=lambda e: -new_deg[e])
        new_frontier = set(new_sorted[:cap])

        all_triples |= hop_trips
        all_ents    |= new_frontier
        frontier     = new_frontier

    return all_ents, all_triples


# ── Virtuoso enrichment (incoming triples not in fb_en.txt) ──────────────────

def enrich_virtuoso_incoming(
    all_entities: Set[str],
    batch_size: int = SPARQL_BATCH,
) -> Set[Tuple[str, str, str]]:
    """
    Query Virtuoso for incoming triples ?s ?p ?e where ?e is in all_entities.
    Filters out iri_id_ corrupted predicates.
    Returns new triples not already in the entity set (subjects may be new).
    """
    extra: Set[Tuple[str, str, str]] = set()
    ents  = list(all_entities)

    for i in range(0, len(ents), batch_size):
        batch = ents[i:i + batch_size]
        vals  = " ".join(f"ns:{e}" for e in batch)
        query = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?s ?p ?o WHERE {{
  VALUES ?o {{ {vals} }}
  ?s ?p ?o .
  FILTER(STRSTARTS(STR(?p), 'http://rdf.freebase.com/ns/'))
  FILTER(!CONTAINS(STR(?p), 'iri_id_'))
  FILTER(STRSTARTS(STR(?s), 'http://rdf.freebase.com/ns/'))
}} LIMIT 5000"""
        for row in run_sparql(query):
            s = row.get("s", "")
            p = row.get("p", "")
            o = row.get("o", "")
            if s and p and o:
                extra.add((s, p, o))

    return extra


# ── MID cache helpers ─────────────────────────────────────────────────────────

def _cache_path(mid: str, dataset: str) -> Path:
    return MID_CACHE_DIR / dataset / f"{mid}.pkl"


def load_mid_cache(mid: str, dataset: str) -> Optional[Tuple[Set, Set]]:
    p = _cache_path(mid, dataset)
    if p.exists():
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def save_mid_cache(mid: str, dataset: str, entities: Set, triples: Set) -> None:
    p   = _cache_path(mid, dataset)
    tmp = p.with_suffix(".tmp")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        pickle.dump((entities, triples), f, protocol=4)
    os.replace(tmp, p)


# ── Per-MID extractor ─────────────────────────────────────────────────────────

def extract_mid_subgraph(
    mid: str,
    dataset: str,
    max_hops: int,
    caps: List[int],
    fwd: Dict,
    bwd: Dict,
    use_nxhd: bool = True,
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """
    Tier 1: nxhd file cache
    Tier 2: BFS over fb_en.txt in-memory index + Virtuoso incoming enrichment
    """
    # Tier 1
    if use_nxhd:
        nxhd_dir = NXHD_CWQ if dataset == "cwq" else NXHD_WEBQSP
        nxhd_path = nxhd_dir / f"{mid}.nxhd"
        if nxhd_path.exists() and nxhd_path.stat().st_size > 0:
            ents, trips = parse_nxhd(nxhd_path)
            if ents:
                return ents, trips

    # Tier 2: BFS over fb_en.txt
    ents, trips = hop_bfs_fb(mid, max_hops, caps, fwd, bwd)

    # Virtuoso enrichment for incoming edges
    extra = enrich_virtuoso_incoming(ents)
    for s, p, o in extra:
        trips.add((s, p, o))
        if _is_entity(s):
            ents.add(s)
        if _is_entity(o):
            ents.add(o)

    return ents, trips


# ── Stage 1 orchestrator ──────────────────────────────────────────────────────

def stage1_extract_all_mids(
    dataset: str,
    records: List[dict],
    max_hops: int,
    caps: List[int],
    fwd: Dict,
    bwd: Dict,
    n_workers: int = 20,
    use_nxhd: bool = True,
    failed_log: Optional[Path] = None,
) -> None:
    # Collect unique MIDs
    all_mids: Set[str] = set()
    for rec in records:
        all_mids.update((rec.get("gold_entity_map") or {}).keys())

    # Skip already cached
    todo = [m for m in sorted(all_mids) if load_mid_cache(m, dataset) is None]
    log.info(f"[{dataset}] Stage 1: {len(all_mids)} unique MIDs, {len(todo)} to extract")

    if not todo:
        return

    failed: List[str] = []
    lock = threading.Lock()
    done = [0]

    def _work(mid: str):
        try:
            ents, trips = extract_mid_subgraph(mid, dataset, max_hops, caps, fwd, bwd, use_nxhd)
            save_mid_cache(mid, dataset, ents, trips)
        except Exception as e:
            with lock:
                failed.append(mid)
                log.warning(f"  MID {mid} failed: {e}")
            return
        with lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(todo):
                log.info(f"  [{dataset}] {done[0]}/{len(todo)} MIDs extracted")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_work, m) for m in todo]
        for f in as_completed(futs):
            pass  # results handled inside _work

    if failed:
        log.warning(f"[{dataset}] {len(failed)} MIDs failed")
        if failed_log:
            with open(failed_log, "a") as f:
                f.write("\n".join(failed) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 – PPR pruning
# ═══════════════════════════════════════════════════════════════════════════════

# ── GraftNet PPR — exact port of step4_extract_subgraphs.py ──────────────────
# Source: freebase/GraftNet/preprocessing/step4_extract_subgraphs.py
# Flags used: DECOMPOSE_PPV=True, SEED_WEIGHTING=True, RELATION_WEIGHTING=True,
#             USEANSWER=False, RESTART=0.8, MAX_ITER=20, EXPONENT=2, NOTFOUNDSCORE=0
#
# Adaptation: relation score = 1.0 if rel in gold_relation_map else 0.0
#   (replaces cosine-sim of question vs relation embeddings — we have explicit
#    gold_relation_map so the binary signal is strictly stronger than embeddings)

def _graftnet_read_facts(
    triples: Set[Tuple[str, str, str]],
    gold_rels: Set[str],
) -> Tuple[Dict[str, int], Dict[str, csr_matrix], Optional[csr_matrix]]:
    """
    Mirrors _read_facts() from GraftNet step4.

    Builds:
      entity_map : MID → integer index
      per_rel    : relation → (E×E) csr_matrix, row-normalised, weighted by
                   score^PPR_EXPONENT  where score = 1.0 (gold) or 0.0 (other)
      adj_mat    : average of per_rel matrices, row-normalised  (DECOMPOSE_PPV=True)

    NOTE: relation_map stores only the forward direction (e1→e2), matching
    GraftNet exactly. The global all_row/all_col uses both directions for the
    undirected fallback (not used here since DECOMPOSE_PPV=True).
    """
    entity_map: Dict[str, int] = {}
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
        si, oi = entity_map[s], entity_map[o]
        # GraftNet _read_facts: relation_map stores forward edge only (e1→e2)
        relation_map[r][0].append(si)
        relation_map[r][1].append(oi)

    if not relation_map:
        return {}, {}, None

    n = num_entities
    per_rel: Dict[str, csr_matrix] = {}
    for rel, (rows, cols) in relation_map.items():
        m = csr_matrix(
            (np.ones(len(rows)), (np.array(rows), np.array(cols))),
            shape=(n, n),
        )
        m = normalize(m, norm="l1", axis=1)            # row-normalise
        score = 1.0 if rel in gold_rels else PPR_NOTFOUND   # RELATION_WEIGHTING
        per_rel[rel] = m * (score ** PPR_EXPONENT)     # scale by score^2

    # DECOMPOSE_PPV=True: average per-relation matrices, then normalise
    adj_mat = sum(per_rel.values()) / len(per_rel)
    adj_mat = normalize(adj_mat, norm="l1", axis=1)

    return entity_map, per_rel, adj_mat


def _graftnet_ppr(seed: np.ndarray, W: csr_matrix) -> np.ndarray:
    """
    Exact copy of _personalized_pagerank() from GraftNet step4.
    seed : (E, 1) column vector.   W : (E, E) row-stochastic matrix.
    Returns accumulated PPR vector shape (E,).
    """
    restart_prob = PPR_RESTART          # 0.8
    r     = restart_prob * seed
    s_ovr = np.copy(r)
    for _ in range(PPR_MAX_ITER):       # 20
        r_new = (1. - restart_prob) * W.T.dot(r)
        s_ovr = s_ovr + r_new
        delta = abs(r_new.sum())
        if delta < 1e-5:
            break
        r = r_new
    return np.squeeze(s_ovr)


def graftnet_ppr(
    triples: Set[Tuple[str, str, str]],
    seed_mids: List[str],
    gold_rels: Set[str],
    top_m: int,
) -> List[str]:
    """
    Full GraftNet PPR pipeline.
    Mirrors the __main__ loop of step4_extract_subgraphs.py:
      _read_facts → _get_subgraph (SEED_WEIGHTING=True, USEANSWER=False)
    Returns list of up to top_m entity MIDs ranked by PPR score.
    """
    entity_map, per_rel, adj_mat = _graftnet_read_facts(triples, gold_rels)
    if adj_mat is None or not entity_map:
        return list(seed_mids)

    n = adj_mat.shape[0]

    # _get_subgraph: SEED_WEIGHTING=True — inverse-rank weights, normalised
    seed_indices = [entity_map[m] for m in seed_mids if m in entity_map]
    seed = np.zeros((n, 1))
    if seed_indices:
        seed[seed_indices] = np.expand_dims(
            np.arange(len(seed_indices), 0, -1, dtype=float), axis=1
        )
        seed = seed / seed.sum()

    ppr       = _graftnet_ppr(seed, adj_mat)
    sorted_idx = np.argsort(ppr)[::-1]
    extracted  = sorted_idx[:top_m]

    # _get_subgraph: drop near-zero PPR entries (< 1e-6)
    zero_idx = np.where(ppr[extracted] < 1e-6)[0]
    if zero_idx.shape[0] > 0:
        extracted = extracted[:zero_idx[0]]

    inv_map = {i: e for e, i in entity_map.items()}
    return [inv_map[i] for i in extracted if i in inv_map]


# ── PageRank-Nibble (WebQSP) — reused from new_exp/script/subgraph_methods.py ─

def pagerank_nibble(
    adj: Dict[str, List[str]],
    seed_nodes: List[str],
    alpha: float = PRN_ALPHA,
    epsilon: float = PRN_EPSILON,
    top_m: int = WEBQSP_CAP,
) -> List[str]:
    """
    Standard PageRank-Nibble with multi-seed uniform personalization.
    (Identical to new_exp/script/subgraph_methods.py)
    """
    degree   = {node: len(neighbors) for node, neighbors in adj.items()}
    all_nodes = list(adj.keys())

    if not all_nodes or not seed_nodes:
        return seed_nodes[:]

    pr: Dict[str, float] = defaultdict(float)
    seed_weight = 1.0 / len(seed_nodes)
    for s in seed_nodes:
        pr[s] = seed_weight

    for _ in range(100):
        new_pr: Dict[str, float] = defaultdict(float)
        for node in all_nodes:
            if degree.get(node, 0) == 0:
                continue
            share = pr[node] / degree[node]
            for nb in adj[node]:
                new_pr[nb] += (1 - alpha) * share
        for s in seed_nodes:
            new_pr[s] += alpha * seed_weight

        delta = sum(abs(new_pr[n] - pr[n]) for n in all_nodes)
        pr    = new_pr
        if delta < epsilon:
            break

    sorted_nodes = sorted(pr.items(), key=lambda x: -x[1])
    return [node for node, _ in sorted_nodes[:top_m]]


def _build_adj_list(triples: Set[Tuple[str, str, str]]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = defaultdict(list)
    for s, r, o in triples:
        if _is_entity(s) and _is_entity(o):
            adj[s].append(o)
            adj[o].append(s)
    return adj


# ── Per-question subgraph builder ─────────────────────────────────────────────

def _build_one_question(
    record: dict,
    dataset: str,
    cap: int,
    out_dir: Path,
    resume: bool,
) -> Tuple[str, str]:
    """
    Build and save subgraph JSON for one question.
    Returns (record_id, status).
    """
    rid      = record["id"]
    out_path = out_dir / f"{rid}.json"

    if resume and out_path.exists():
        return rid, "skipped"

    gem      = record.get("gold_entity_map") or {}
    seeds    = list(gem.keys())
    gold_rels = set((record.get("gold_relation_map") or {}).keys())

    # Load and union seed MID caches
    all_ents:   Set[str]                    = set()
    all_triples: Set[Tuple[str, str, str]]  = set()
    missing = []
    for mid in seeds:
        cached = load_mid_cache(mid, dataset)
        if cached:
            e, t = cached
            all_ents    |= e
            all_triples |= t
        else:
            missing.append(mid)
    if missing:
        log.debug(f"  {rid}: {len(missing)} seed MIDs not in cache — skipping those")

    if not all_ents:
        # No subgraph data — write minimal record
        out = {
            "id": rid,
            "question": record.get("perturbed_question", ""),
            "original_question": record.get("original_question", ""),
            "answers": record.get("answers", []),
            "seed_mids": seeds,
            "entities": seeds,
            "triples": [],
        }
        _write_json_atomic(out, out_path)
        return rid, "empty"

    # PPR pruning
    if dataset == "cwq":
        kept_ents = graftnet_ppr(all_triples, seeds, gold_rels, top_m=cap)
    else:
        adj = _build_adj_list(all_triples)
        # Ensure seeds are in adj
        for s in seeds:
            if s not in adj:
                adj[s] = []
        kept_ents = pagerank_nibble(adj, seeds, top_m=cap)

    kept_set  = set(kept_ents)
    # Always include seeds in entity set
    kept_set.update(s for s in seeds if s)

    kept_trips = [
        [s, r, o] for s, r, o in all_triples
        if s in kept_set and o in kept_set
    ]

    out = {
        "id": rid,
        "question": record.get("perturbed_question", ""),
        "original_question": record.get("original_question", ""),
        "answers": record.get("answers", []),
        "seed_mids": seeds,
        "entities": sorted(kept_set),
        "triples": kept_trips,
    }
    _write_json_atomic(out, out_path)
    return rid, "done"


def _write_json_atomic(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


# ── Stage 2 orchestrator (multiprocessing) ────────────────────────────────────

# Global state for worker processes (set via initializer)
_g_dataset  = None
_g_cap      = None
_g_out_dir  = None
_g_resume   = None


def _init_worker(dataset, cap, out_dir, resume):
    global _g_dataset, _g_cap, _g_out_dir, _g_resume
    _g_dataset = dataset
    _g_cap     = cap
    _g_out_dir = Path(out_dir)
    _g_resume  = resume


def _worker_one(record):
    return _build_one_question(record, _g_dataset, _g_cap, _g_out_dir, _g_resume)


def stage2_build_all_subgraphs(
    dataset: str,
    records: List[dict],
    n_workers: int = 40,
    resume: bool   = True,
) -> None:
    out_dir  = SUBGRAPH_DIR / f"{dataset}_p1"
    out_dir.mkdir(parents=True, exist_ok=True)
    cap      = CWQ_CAP if dataset == "cwq" else WEBQSP_CAP

    pending = records
    if resume:
        pending = [r for r in records if not (out_dir / f"{r['id']}.json").exists()]
    log.info(f"[{dataset}] Stage 2: {len(pending)}/{len(records)} questions to process")

    if not pending:
        return

    done_n = [0]
    t0     = time.time()

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(dataset, cap, str(out_dir), resume),
    ) as pool:
        futs = {pool.submit(_worker_one, r): r["id"] for r in pending}
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
                rate    = done_n[0] / elapsed
                eta     = (len(pending) - done_n[0]) / max(rate, 1e-9)
                log.info(f"  [{dataset}] {done_n[0]}/{len(pending)} | {rate:.1f}/s | ETA {eta:.0f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 – GNN-RAG integer encoding
# ═══════════════════════════════════════════════════════════════════════════════

def stage3_build_gnn_input(
    dataset: str,
    tsv_names: Dict[str, str],
) -> None:
    subgraph_dir = SUBGRAPH_DIR / f"{dataset}_p1"
    gnn_out_dir  = GNN_DIR / f"{dataset}_p1"
    gnn_out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(subgraph_dir.glob("*.json"))
    log.info(f"[{dataset}] Stage 3: building GNN vocab from {len(json_files)} files")

    # Pass 1: build vocab
    all_mids: Set[str] = set()
    all_rels: Set[str] = set()
    for jp in json_files:
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
            all_mids.update(d.get("entities", []))
            for t in d.get("triples", []):
                if len(t) >= 3:
                    all_mids.add(t[0]); all_rels.add(t[1]); all_mids.add(t[2])
        except Exception:
            pass

    ent_list = sorted(all_mids)
    rel_list = sorted(all_rels)
    ent2id   = {m: i for i, m in enumerate(ent_list)}
    rel2id   = {r: i for i, r in enumerate(rel_list)}

    (gnn_out_dir / "entities.txt").write_text("\n".join(ent_list) + "\n", encoding="utf-8")
    (gnn_out_dir / "relations.txt").write_text("\n".join(rel_list) + "\n", encoding="utf-8")
    log.info(f"  vocab: {len(ent_list):,} entities, {len(rel_list):,} relations")

    # Pass 2: encode records and write JSONL
    out_path = gnn_out_dir / "test_subgraph.json"
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for jp in json_files:
            try:
                d = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue

            seed_ids = [ent2id[m] for m in d.get("seed_mids", []) if m in ent2id]
            all_ent_ids = sorted(set(ent2id[e] for e in d.get("entities", []) if e in ent2id))

            tuples = []
            for t in d.get("triples", []):
                if len(t) < 3:
                    continue
                s, r, o = t[0], t[1], t[2]
                if s in ent2id and r in rel2id and o in ent2id:
                    tuples.append([ent2id[s], rel2id[r], ent2id[o]])

            answers = []
            for ans_mid in d.get("answers", []):
                name = tsv_names.get(ans_mid, "")
                answers.append({"kb_id": ans_mid, "text": name})

            rec = {
                "id":               d["id"],
                "question":         d.get("question", ""),
                "question_orig":    d.get("original_question", ""),
                "answers":          answers,
                "entities":         seed_ids,
                "subgraph":         {"tuples": tuples, "entities": all_ent_ids},
                "perturbation_type": "p1",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    log.info(f"  [{dataset}] wrote {n_written} records to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Dry-run quality report
# ═══════════════════════════════════════════════════════════════════════════════

def dry_run_quality_report(dataset: str, n: int = 5) -> None:
    subgraph_dir = SUBGRAPH_DIR / f"{dataset}_p1"
    files = sorted(subgraph_dir.glob("*.json"))[:n]
    if not files:
        log.info(f"[{dataset}] No subgraph files found for quality report")
        return

    log.info(f"\n{'='*60}")
    log.info(f"Quality report: {dataset} ({len(files)} records)")
    log.info(f"{'='*60}")

    total_ans = total_in = total_golden = 0
    for jp in files:
        d = json.loads(jp.read_text())
        ent_set  = set(d.get("entities", []))
        answers  = d.get("answers", [])
        seeds    = d.get("seed_mids", [])
        trips    = [(t[0], t[1], t[2]) for t in d.get("triples", []) if len(t) >= 3]

        ans_in = sum(1 for a in answers if a in ent_set)
        total_ans += len(answers); total_in += ans_in

        # BFS golden path check
        adj: Dict[str, Set[str]] = defaultdict(set)
        for s, r, o in trips:
            adj[s].add(o); adj[o].add(s)
        golden = False
        for seed in seeds:
            visited, queue = {seed}, [seed]
            while queue:
                curr = queue.pop(0)
                if curr in set(answers):
                    golden = True; break
                for nb in adj.get(curr, set()):
                    if nb not in visited:
                        visited.add(nb); queue.append(nb)
            if golden:
                break
        if golden:
            total_golden += 1

        log.info(
            f"  {d['id'][:50]:50s} | ents={len(ent_set):5d} | trips={len(trips):6d} | "
            f"ans={ans_in}/{len(answers)} | golden={'Y' if golden else 'N'}"
        )

    if total_ans:
        log.info(f"  Answer coverage: {total_in}/{total_ans} ({100*total_in/total_ans:.0f}%)")
    log.info(f"  Golden path:     {total_golden}/{len(files)} ({100*total_golden/len(files):.0f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description="Build KG subgraphs for new_exp datasets")
    ap.add_argument("--dataset",     choices=["cwq", "webqsp", "both"], default="both")
    ap.add_argument("--stage",       choices=["1", "2", "3", "all"],    default="all")
    ap.add_argument("--workers-s1",  type=int, default=20,  help="Threads for Virtuoso BFS")
    ap.add_argument("--workers-s2",  type=int, default=40,  help="Processes for PPR")
    ap.add_argument("--fb-file",     type=Path, default=DEFAULT_FB_FILE)
    ap.add_argument("--tsv-path",    type=Path, default=DEFAULT_TSV_FILE)
    ap.add_argument("--dry-run",     action="store_true", help="Process 5 records only")
    ap.add_argument("--no-resume",   action="store_true", help="Re-process even if output exists")
    ap.add_argument("--force-sparql",action="store_true", help="Skip nxhd Tier 1 cache")
    return ap.parse_args()


def main():
    args   = parse_args()
    resume = not args.no_resume
    stages = ["1", "2", "3"] if args.stage == "all" else [args.stage]

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(LOG_DIR / f"build_subgraphs_{ts}.log")
    log.info(f"Starting: dataset={args.dataset} stage={args.stage} dry_run={args.dry_run}")

    datasets = ["cwq", "webqsp"] if args.dataset == "both" else [args.dataset]

    # ── Load fb_en.txt index once (shared across both datasets) ───────────────
    fwd: Dict = {}
    bwd: Dict = {}
    if "1" in stages:
        if args.fb_file.exists():
            fwd, bwd = load_fb_index(args.fb_file)
        else:
            log.warning(f"fb_file not found: {args.fb_file} — Stage 1 will use Virtuoso only")

    # ── Load TSV once ─────────────────────────────────────────────────────────
    tsv_names: Dict[str, str] = {}
    if "3" in stages or "2" in stages:
        tsv_names = load_tsv(args.tsv_path)

    for dataset in datasets:
        dataset_file = DATASET_DIR / f"{dataset}_p1.json"
        if not dataset_file.exists():
            log.warning(f"Dataset not found: {dataset_file}")
            continue

        records = json.loads(dataset_file.read_text(encoding="utf-8"))
        if args.dry_run:
            records = records[:5]
            log.info(f"[{dataset}] Dry-run: limiting to {len(records)} records")

        hops = CWQ_HOPS   if dataset == "cwq" else WEBQSP_HOPS
        caps = CWQ_CAPS   if dataset == "cwq" else WEBQSP_CAPS

        failed_log = LOG_DIR / f"failed_mids_{dataset}.txt"

        if "1" in stages:
            log.info(f"\n{'─'*50}")
            log.info(f"[{dataset}] === Stage 1: MID Extraction ===")
            t0 = time.time()
            stage1_extract_all_mids(
                dataset=dataset,
                records=records,
                max_hops=hops,
                caps=caps,
                fwd=fwd,
                bwd=bwd,
                n_workers=args.workers_s1,
                use_nxhd=not args.force_sparql,
                failed_log=failed_log,
            )
            log.info(f"[{dataset}] Stage 1 done in {time.time()-t0:.1f}s")

        if "2" in stages:
            log.info(f"\n{'─'*50}")
            log.info(f"[{dataset}] === Stage 2: PPR Pruning ===")
            t0 = time.time()
            stage2_build_all_subgraphs(
                dataset=dataset,
                records=records,
                n_workers=args.workers_s2,
                resume=resume,
            )
            log.info(f"[{dataset}] Stage 2 done in {time.time()-t0:.1f}s")

        if "3" in stages:
            log.info(f"\n{'─'*50}")
            log.info(f"[{dataset}] === Stage 3: GNN-RAG Encoding ===")
            t0 = time.time()
            stage3_build_gnn_input(dataset=dataset, tsv_names=tsv_names)
            log.info(f"[{dataset}] Stage 3 done in {time.time()-t0:.1f}s")

        if args.dry_run:
            dry_run_quality_report(dataset, n=5)

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
