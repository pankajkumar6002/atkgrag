#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_subgraph_variants.py
==========================
Build three new subgraph variants for GNN-RAG inference:

  p7_graftnetv2  — GraftNet: fb_en.txt BFS (bidirectional, degree-sorted caps)
                   + Virtuoso incoming enrichment (port 8890). Forward-only
                   per-relation PPR weighted by gold_relation_map. No T1/T2.
  p7_nsm         — NSM: manual_fb_filter.txt streaming 2-hop expansion with
                   CVT bonus scan (get_2hop_subgraph.py logic), then load_kb
                   CVT-merge on collected triples + flat undirected PPR.
  p4_nsm         — Same NSM strategy on p4 dataset.

Usage:
  python build_subgraph_variants.py --variant p7_graftnetv2 --dataset both
  python build_subgraph_variants.py --variant p7_nsm --dataset both
  python build_subgraph_variants.py --variant p4_nsm --dataset both
  python build_subgraph_variants.py --variant all --dataset both
  python build_subgraph_variants.py --variant p7_nsm --dataset cwq --dry-run
  python build_subgraph_variants.py --variant p7_graftnetv2 --dataset cwq --stage 2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import resource
import struct
import sys
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

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = REPO_ROOT
PERT_DIR     = REPO_ROOT / "tmlr" / "pert" / "dataset"
GNN_DATA     = REPO_ROOT / "tmlr" / "Repo" / "GNN-RAG" / "data"

# GraftNetV2: fb_en.txt (full Freebase, bidirectional BFS)
FB_EN        = REPO_ROOT / "freebase" / "data" / "fb_en.txt"

# Virtuoso SPARQL endpoint for incoming edge enrichment (GraftNetV2)
SPARQL_EP    = "http://127.0.0.1:8890/sparql"
SPARQL_BATCH = 100

# NSM: manual_fb_filter.txt (subject-out, 2-hop streaming)
MANUAL_FB    = REPO_ROOT / "freebase" / "WSDM2021_NSM" / "preprocessing" / "Freebase" / "manual_fb_filter.txt"

# CVT node binary (NSM)
CVT_BIN      = REPO_ROOT / "freebase" / "data" / "cvtnodes.bin"

# Output roots
SUBGRAPH_ROOT = PERT_DIR / "subgraphs_p7"
SUBGRAPH_P4   = PERT_DIR / "subgraphs_p4"
GNN_OUT_BASE  = PERT_DIR / "gnn_input"

DATASET_SUBDIR = {"cwq": "CWQ", "webqsp": "webqsp"}

# ── Hyperparameters ────────────────────────────────────────────────────────────
# GraftNetV2 BFS caps per hop
CWQ_HOPS    = 4
CWQ_CAPS    = [500, 300, 100, 50]
CWQ_CAP_PPR = 2500

WEBQSP_HOPS    = 2
WEBQSP_CAPS    = [1000, 500]
WEBQSP_CAP_PPR = 2000

NSM_CAP_PPR   = 2000
# Hop-1 entity cap for NSM: after collecting hop1 triples, keep only the
# top-N entities by out-degree before expanding hop2. Prevents combinatorial
# explosion when seeds have very high fanout (e.g. popular entities).
NSM_HOP1_CAP  = 500

# Per-question triple cap written to Stage 1 temp pickle (prevents huge pickles
# that bloat the parent process and each worker's RSS on load).
# NSM 2-hop with hop1_cap=500 can produce O(500×avg_degree) triples.
# We sample randomly if over cap to keep coverage roughly uniform.
S1_TRIPLE_CAP = 100_000   # max triples stored per question in .pkl
                          # 100K triples × ~250 bytes/triple in RAM = ~25 MB per worker
                          # 20 workers × 25 MB = 500 MB overhead (safe)

# Per-worker virtual address space limit: kills runaway worker instead of the machine.
# CVT (5 GB) + pkl load (~25 MB) + PPR matrix (~50 MB) = ~6 GB realistic per worker.
# Cap at 20 GB to give headroom without risking system OOM.
WORKER_VIRT_LIMIT_GB = 30

NXHD_MAX_LINES = 5_000_000

# GraftNet PPR constants (from GraftNet step4_extract_subgraphs.py)
PPR_RESTART  = 0.8
PPR_MAX_ITER = 20
PPR_EXPONENT = 2.0
PPR_NOTFOUND = 0.0

log = logging.getLogger("build_variants")


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
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_entity(tok: str) -> bool:
    return tok.startswith("m.") or tok.startswith("g.")


def _write_atomic(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


# ══════════════════════════════════════════════════════════════════════════════
# GraftNetV2 — fb_en.txt in-memory index (bidirectional)
# ══════════════════════════════════════════════════════════════════════════════

_fb_fwd: Dict[str, List[Tuple[str, str]]] = {}
_fb_bwd: Dict[str, List[Tuple[str, str]]] = {}
_fb_loaded = False


def load_fb_index(fb_file: Path = FB_EN) -> None:
    global _fb_fwd, _fb_bwd, _fb_loaded
    if _fb_loaded:
        return
    log.info(f"Loading fb_en.txt from {fb_file} (~5 min, ~100 GB RAM) …")
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
    log.info(f"  fb_en.txt ready: {len(_fb_fwd):,} subjects in {time.time()-t0:.1f}s")


def bfs_fb(
    seed_mid: str,
    max_hops: int,
    caps: List[int],
    max_triples_per_mid: int = S1_TRIPLE_CAP,
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """Bidirectional degree-sorted BFS over fb_en.txt in-memory index.

    max_triples_per_mid caps the total triple set accumulated for this MID
    to prevent frontier entities with huge degree from exploding memory.
    """
    all_ents: Set[str] = {seed_mid}
    all_triples: Set[Tuple[str, str, str]] = set()
    frontier: Set[str] = {seed_mid}

    for hop in range(max_hops):
        if len(all_triples) >= max_triples_per_mid:
            break  # already at cap — no point expanding further
        cap = caps[hop] if hop < len(caps) else 50
        new_deg: Dict[str, int] = defaultdict(int)
        hop_trips: Set[Tuple[str, str, str]] = set()

        for entity in frontier:
            for pred, obj in _fb_fwd.get(entity, []):
                hop_trips.add((entity, pred, obj))
                if _is_entity(obj) and obj not in all_ents:
                    new_deg[obj] += 1
            for pred, subj in _fb_bwd.get(entity, []):
                hop_trips.add((subj, pred, entity))
                if _is_entity(subj) and subj not in all_ents:
                    new_deg[subj] += 1

        new_frontier = set(sorted(new_deg, key=lambda e: -new_deg[e])[:cap])
        all_triples |= hop_trips
        if len(all_triples) > max_triples_per_mid:
            import random as _random
            all_triples = set(_random.sample(list(all_triples), max_triples_per_mid))
        all_ents |= new_frontier
        frontier = new_frontier

    return all_ents, all_triples


def run_sparql(query: str, retries: int = 3, timeout: int = 30) -> List[dict]:
    params = urllib.parse.urlencode({"query": query, "format": "application/sparql-results+json"})
    url = f"{SPARQL_EP}?{params}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                bindings = data.get("results", {}).get("bindings", [])
                return [{k: v["value"] for k, v in row.items()} for row in bindings]
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(1)
    return []


def enrich_virtuoso_incoming(
    all_entities: Set[str],
    batch_size: int = SPARQL_BATCH,
) -> Set[Tuple[str, str, str]]:
    """
    Query Virtuoso for incoming triples ?s ?p ?e where ?e in all_entities.
    Mirrors build_subgraphs.py enrich_virtuoso_incoming exactly.
    """
    extra: Set[Tuple[str, str, str]] = set()
    ents = list(all_entities)
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
            s = row.get("s", "").replace("http://rdf.freebase.com/ns/", "")
            p = row.get("p", "").replace("http://rdf.freebase.com/ns/", "")
            o = row.get("o", "").replace("http://rdf.freebase.com/ns/", "")
            if s and p and o:
                extra.add((s, p, o))
    return extra


# ══════════════════════════════════════════════════════════════════════════════
# GraftNetV2 PPR — forward-only per-relation matrices (no symmetrisation)
# ══════════════════════════════════════════════════════════════════════════════

def _graftnetv2_read_facts(
    triples: Set[Tuple[str, str, str]],
    gold_rels: Set[str],
) -> Tuple[Dict[str, int], Optional[csr_matrix]]:
    """
    Symmetric per-relation sparse matrices, gold-relation binary weighting.
    Each triple (s,r,o) contributes both s→o and o→s edges so that PPR
    (which uses W.T) can propagate from seeds that are objects of gold_rel
    triples, not just subjects.
    score(r) = 1.0 if r in gold_rels else 0.0; M_r scaled by score^PPR_EXPONENT.
    W = mean(all M_r), row-normalised.
    """
    entity_map: Dict[str, int] = {}
    relation_map: Dict[str, Tuple[List[int], List[int]]] = {}
    num_entities = 0

    for s, r, o in triples:
        if not _is_entity(s) or not _is_entity(o):
            continue
        if s not in entity_map:
            entity_map[s] = num_entities; num_entities += 1
        if o not in entity_map:
            entity_map[o] = num_entities; num_entities += 1
        if r not in relation_map:
            relation_map[r] = ([], [])
        # Forward s→o and reverse o→s so seeds can be objects of gold_rel edges
        relation_map[r][0].append(entity_map[s])
        relation_map[r][1].append(entity_map[o])
        relation_map[r][0].append(entity_map[o])
        relation_map[r][1].append(entity_map[s])

    if not relation_map or not entity_map:
        return entity_map, None

    n = num_entities
    per_rel: Dict[str, csr_matrix] = {}
    for rel, (rows, cols) in relation_map.items():
        m = csr_matrix(
            (np.ones(len(rows)), (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
            shape=(n, n),
        )
        m = normalize(m, norm="l1", axis=1)
        score = 1.0 if rel in gold_rels else PPR_NOTFOUND
        per_rel[rel] = m * (score ** PPR_EXPONENT)

    adj_mat = sum(per_rel.values()) / len(per_rel)
    adj_mat = normalize(adj_mat, norm="l1", axis=1)
    return entity_map, adj_mat


def _graftnet_ppr(seed: np.ndarray, W: csr_matrix) -> np.ndarray:
    """Accumulated PPR — identical formula in both GraftNet and NSM."""
    restart_prob = PPR_RESTART
    r = restart_prob * seed
    s_ovr = np.copy(r)
    for _ in range(PPR_MAX_ITER):
        r_new = (1.0 - restart_prob) * W.T.dot(r)
        s_ovr += r_new
        if abs(r_new.sum()) < 1e-5:
            break
        r = r_new
    return np.squeeze(s_ovr)


def graftnetv2_ppr(
    triples: Set[Tuple[str, str, str]],
    seed_mids: List[str],
    gold_rels: Set[str],
    top_m: int,
) -> List[str]:
    """GraftNet PPR: forward-only per-relation matrices, inverse-rank seed."""
    entity_map, adj_mat = _graftnetv2_read_facts(triples, gold_rels)
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
    zero_idx = np.where(ppr[extracted] < 1e-6)[0]
    if zero_idx.shape[0] > 0:
        extracted = extracted[:zero_idx[0]]

    inv_map = {i: e for e, i in entity_map.items()}
    result = [inv_map[i] for i in extracted if i in inv_map]
    result_set = set(result)
    for s in seed_mids:
        if s not in result_set and s in entity_map:
            result.append(s)
    return result


def _pagerank_nibble(
    triples: Set[Tuple[str, str, str]],
    seed_nodes: List[str],
    top_m: int,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
) -> List[str]:
    """PageRank-Nibble for WebQSP GraftNetV2 (topology-only, multi-seed)."""
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

    return [node for node, _ in sorted(pr.items(), key=lambda x: -x[1])[:top_m]]


# ══════════════════════════════════════════════════════════════════════════════
# NSM — CVT node loading
# ══════════════════════════════════════════════════════════════════════════════

_cvt_nodes: Optional[Dict] = None
_cvt_lock = __import__("threading").Lock()


def load_cvt_nodes(cvt_bin: Path = CVT_BIN) -> Dict:
    """
    Load cvtnodes.bin via the exact BinaryStream format from deal_cvt.py:
      readInt32 → signed 32-bit LE
      readString → 1-byte length (<B) + that many bytes
      readBool → 1 byte
    Returns {prefix_key -> {mid -> bool}}.
    """
    global _cvt_nodes
    if _cvt_nodes is not None:
        return _cvt_nodes
    with _cvt_lock:
        if _cvt_nodes is not None:
            return _cvt_nodes
        log.info(f"  Loading CVT nodes from {cvt_bin} …")
        t0 = time.time()
        cvt: Dict = {}
        try:
            with open(cvt_bin, "rb") as f:
                def read_int32() -> int:
                    return struct.unpack("i", f.read(4))[0]
                def read_string() -> str:
                    length = struct.unpack("<B", f.read(1))[0]
                    return struct.unpack(f"{length}s", f.read(length))[0].decode("utf-8", errors="replace")
                def read_bool() -> bool:
                    return struct.unpack("?", f.read(1))[0]

                n_buckets = read_int32()
                for _ in range(n_buckets):
                    key = read_string()
                    count = read_int32()
                    bucket: Dict[str, bool] = {}
                    for _ in range(count):
                        mid = read_string()
                        bucket[mid] = read_bool()
                    cvt[key] = bucket
        except Exception as e:
            log.warning(f"  CVT load failed ({e}); CVT handling disabled")
            cvt = {}
        log.info(f"  CVT: {sum(len(v) for v in cvt.values()):,} MIDs in "
                 f"{len(cvt)} buckets ({time.time()-t0:.1f}s)")
        _cvt_nodes = cvt
    return _cvt_nodes


def is_cvt(mid: str, cvt_nodes: Dict) -> bool:
    """
    Mirrors deal_cvt.py get_key() + is_cvt() exactly.
    Key = subject[0:4] for m./g. MIDs with len>3.
    """
    if not cvt_nodes:
        return False
    if mid.startswith("m.") or mid.startswith("g."):
        if len(mid) > 3:
            key = mid[0:4]
        elif len(mid) > 2:
            key = mid[0:3]
        else:
            key = mid[0:2]
    else:
        key = mid[0:2] if len(mid) > 1 else mid[0:1]
    bucket = cvt_nodes.get(key)
    return bool(bucket.get(mid, False)) if bucket is not None else False


# ══════════════════════════════════════════════════════════════════════════════
# NSM — manual_fb_filter.txt in-memory index (raw forward only, no CVT merge)
# ══════════════════════════════════════════════════════════════════════════════
# We load into memory for efficiency (scanning 37GB per question is too slow),
# but store as raw (subject -> [(pred, obj)]) — NO CVT pre-merge at load time.
# CVT merging happens later in nsm_ppr() on the collected per-question triples,
# matching preprocess_step1.py load_kb() which is only called at PPR time.

_nsm_raw_fwd: Dict[str, List[Tuple[str, str]]] = {}  # subject -> [(pred, obj)]
_nsm_loaded = False
_nsm_lock = __import__("threading").Lock()


def load_nsm_kb(kb_file: Path = MANUAL_FB) -> None:
    """
    Load manual_fb_filter.txt into raw forward dict: subject -> [(pred, obj)].
    No CVT merging — that happens per-question in nsm_ppr().
    """
    global _nsm_raw_fwd, _nsm_loaded
    if _nsm_loaded:
        return
    with _nsm_lock:
        if _nsm_loaded:
            return
        log.info(f"Loading NSM KB from {kb_file} …")
        t0 = time.time()
        fwd: Dict[str, List] = defaultdict(list)
        n = 0
        with open(kb_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                s, p, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if not s or not p or not o:
                    continue
                fwd[s].append((p, o))
                n += 1
                if n % 50_000_000 == 0:
                    log.info(f"  {n/1e6:.0f}M lines, {time.time()-t0:.0f}s")
        _nsm_raw_fwd = dict(fwd)
        _nsm_loaded = True
        log.info(f"  NSM KB: {len(_nsm_raw_fwd):,} subjects, {n:,} triples in {time.time()-t0:.1f}s")


def _fetch_1hop(seed_set: Set[str], cvt_nodes: Dict) -> Tuple[Set[Tuple[str,str,str]], Set[str]]:
    """
    Mirrors get_2hop_subgraph.py fetch_triple_1hop():
      Pass 1: collect all triples where subject in seed_set.
              If object is a CVT node (and not in seed_set), add to cvt_set.
      Pass 2: collect all triples where subject in cvt_set.
    Returns (all collected triples, cvt_set found).
    No per-hop cap — collect everything reachable.
    """
    trip_out: Set[Tuple[str, str, str]] = set()
    cvt_set: Set[str] = set()

    # Pass 1: seed subjects
    for seed in seed_set:
        for pred, obj in _nsm_raw_fwd.get(seed, []):
            trip_out.add((seed, pred, obj))
            if obj not in seed_set and is_cvt(obj, cvt_nodes):
                cvt_set.add(obj)

    # Pass 2: CVT subjects (bonus scan, same as fetch_triple_1hop second loop)
    for cvt in cvt_set:
        for pred, obj in _nsm_raw_fwd.get(cvt, []):
            trip_out.add((cvt, pred, obj))

    return trip_out, cvt_set


def nsm_2hop_triples(
    seed_mids: List[str],
    cvt_nodes: Dict,
    hop1_cap: int = NSM_HOP1_CAP,
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """
    2-hop expansion mirroring get_2hop_subgraph.py:
      Hop 1: fetch_1hop from seeds (+ CVT bonus scan).
      Cap hop-1 entity set to top-hop1_cap by out-degree (prevents hop2 explosion).
      Hop 2: fetch_1hop from capped hop-1 entities (+ CVT bonus scan).
    """
    seed_set = set(seed_mids)

    # Hop 1
    hop1_trips, _ = _fetch_1hop(seed_set, cvt_nodes)

    # Collect hop-1 entity endpoints, measure out-degree, apply cap
    hop1_deg: Dict[str, int] = defaultdict(int)
    for s, r, o in hop1_trips:
        if _is_entity(s) and s not in seed_set:
            hop1_deg[s] += 1
        if _is_entity(o) and o not in seed_set:
            hop1_deg[o] += 1

    # Keep top hop1_cap entities by degree; always keep seeds
    if len(hop1_deg) > hop1_cap:
        top_ents = set(sorted(hop1_deg, key=lambda e: -hop1_deg[e])[:hop1_cap])
    else:
        top_ents = set(hop1_deg.keys())
    hop1_ents = seed_set | top_ents

    # Hop 2: expand from capped hop-1 entities
    hop2_trips, _ = _fetch_1hop(hop1_ents, cvt_nodes)

    all_trips = hop1_trips | hop2_trips
    all_ents: Set[str] = set(seed_mids)
    for s, r, o in all_trips:
        if _is_entity(s):
            all_ents.add(s)
        if _is_entity(o):
            all_ents.add(o)

    return all_ents, all_trips


# ══════════════════════════════════════════════════════════════════════════════
# NSM — PPR with load_kb-style CVT merge + flat undirected adjacency
# ══════════════════════════════════════════════════════════════════════════════

def _nsm_load_kb_local(
    raw_trips: Set[Tuple[str, str, str]],
    cvt_nodes: Dict,
) -> Tuple[Dict[str, Set], Dict[str, Set]]:
    """
    Apply preprocess_step1.py load_kb() logic to the per-question collected triples:
      - Build triples[head] and back_triples[cvt_tail].
      - Merge CVT chains: for each CVT node c in back_triples,
          for each (head, rel, c): if c has outgoing triples, merge; else discard.
    Returns (triples_dict, back_triples_dict).
    """
    triples: Dict[str, Set] = {}
    back_triples: Dict[str, Set] = {}

    for s, r, o in raw_trips:
        if s not in triples:
            triples[s] = set()
        triples[s].add((s, r, o))
        if is_cvt(o, cvt_nodes):
            if o not in back_triples:
                back_triples[o] = set()
            back_triples[o].add((s, r, o))

    # CVT chain merge — exactly as in load_kb()
    for cvt_node, pointing in back_triples.items():
        for triple in pointing:
            head, rel, tail = triple  # tail == cvt_node
            if tail not in triples:
                triples[head].discard(triple)
            else:
                triples[head] |= triples[tail]

    return triples, back_triples


def nsm_ppr(
    raw_trips: Set[Tuple[str, str, str]],
    seed_mids: List[str],
    cvt_nodes: Dict,
    top_m: int,
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    NSM PPR pipeline mirroring preprocess_step1.py retrieve_subgraph():
      1. load_kb() on collected triples → CVT-merged triples dict.
      2. get_2hop_triples() from each seed entity.
      3. build_sp_mat() — flat undirected adjacency, all rels equal weight.
      4. rank_ppr_ents() — uniform seed 1/K, accumulated PPR, top max_ent.
      5. get_subgraph(cvt_add_flag=False) — filter triples, drop dangling CVT.
    Returns (final_ents_sorted, reserve_triples_list).
    """
    # Step 1: CVT-merge collected triples (load_kb equivalent)
    triples_dict, back_triples = _nsm_load_kb_local(raw_trips, cvt_nodes)

    # Step 2: get_2hop_triples from each seed (after CVT merge, seeds may have
    # more triples merged in). Mirrors retrieve_subgraph's triple_set building.
    triple_set: Set[Tuple[str, str, str]] = set()
    entity_set: Set[str] = set()
    for seed in seed_mids:
        if not seed:
            continue
        entity_set.add(seed)
        seed_trips = triples_dict.get(seed, set())
        triple_set |= seed_trips
        for s2, r2, o2 in seed_trips:
            if o2 in triples_dict:
                triple_set |= triples_dict[o2]

    if not triple_set or not entity_set:
        return list(seed_mids), []

    # Step 3: build_sp_mat — flat undirected adjacency (rows/cols both directions)
    ent2id: Dict[str, int] = {}
    rows: List[int] = []
    cols: List[int] = []
    for s, r, o in triple_set:
        if not _is_entity(s) or not _is_entity(o):
            continue
        if s not in ent2id:
            ent2id[s] = len(ent2id)
        if o not in ent2id:
            ent2id[o] = len(ent2id)
        i, j = ent2id[s], ent2id[o]
        rows.extend([i, j])
        cols.extend([j, i])

    if not ent2id:
        return list(seed_mids), []

    for m in seed_mids:
        if m and m not in ent2id:
            ent2id[m] = len(ent2id)

    n = len(ent2id)
    if not rows:
        return list(seed_mids), []

    sp_mat = csr_matrix(
        (np.ones(len(rows)), (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n, n),
    )
    sp_mat = normalize(sp_mat, norm="l1", axis=1)

    # Step 4: rank_ppr_ents — uniform seed 1/K
    seed_list = [ent2id[m] for m in seed_mids if m and m in ent2id]
    if not seed_list:
        return list(seed_mids), []

    seed_vec = np.zeros((n, 1))
    seed_vec[seed_list] = 1.0 / len(set(seed_list))

    ppr = _graftnet_ppr(seed_vec, sp_mat)
    sorted_idx = np.argsort(ppr)[::-1]
    extracted = sorted_idx[:top_m]
    zero_idx = np.where(ppr[extracted] < 1e-6)[0]
    if zero_idx.shape[0] > 0:
        extracted = extracted[:zero_idx[0]]

    id2ent = {v: k for k, v in ent2id.items()}
    cand_ents: Set[str] = set(id2ent[i] for i in extracted if i in id2ent)
    for m in seed_mids:
        if m:
            cand_ents.add(m)

    # Step 5: get_subgraph(cvt_add_flag=False) — mirrors preprocess_step1.py exactly.
    # Collect 1-hop triples from each candidate entity (from CVT-merged triples_dict).
    triple_set2: Set[Tuple[str, str, str]] = set()
    for ent in cand_ents:
        triple_set2 |= triples_dict.get(ent, set())

    # Build reserve_set and tp_edges: both head and tail must be in cand_ents
    reserve_set: Set[Tuple[str, str, str]] = set()
    tp_edges: Dict[str, Set] = {}
    for triple in triple_set2:
        s2, r2, o2 = triple
        if s2 not in cand_ents or o2 not in cand_ents:
            continue
        reserve_set.add(triple)
        if s2 not in tp_edges:
            tp_edges[s2] = set()
        tp_edges[s2].add(triple)

    # Drop dangling CVT edges (cvt_add_flag=False):
    # Mirrors get_subgraph() exactly with both branches.
    to_remove: Set[Tuple[str, str, str]] = set()
    for head, tset in tp_edges.items():
        for triple in tset:
            _, __, tail = triple
            if is_cvt(tail, cvt_nodes):
                if tail not in triples_dict:
                    # CVT has no outgoing edges in per-Q KB → discard
                    to_remove.add(triple)
                elif tail not in tp_edges:
                    # CVT tail not a head in candidate subgraph → discard (cvt_add_flag=False)
                    to_remove.add(triple)
    reserve_set -= to_remove

    # Collect surviving entity set
    final_ents: Set[str] = set()
    for s2, r2, o2 in reserve_set:
        final_ents.add(s2)
        final_ents.add(o2)
    for m in seed_mids:
        if m:
            final_ents.add(m)

    return sorted(final_ents), list(reserve_set)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: raw triple retrieval
# ══════════════════════════════════════════════════════════════════════════════

def stage1_graftnetv2(
    dataset: str,
    records: List[dict],
    max_hops: int,
    caps: List[int],
    n_workers: int,
    s1_tmp_dir: Path = Path("."),
) -> None:
    """
    GraftNetV2 Stage 1: fb_en.txt BFS (bidirectional) + Virtuoso incoming enrichment.
    Writes per-question pkl files to s1_tmp_dir; returns None.
    """
    log.info(f"[{dataset}] Stage 1 (GraftNetV2): {len(records)} records")

    # Collect unique seed MIDs across all records
    all_mids: Set[str] = set()
    rec_seeds: Dict[str, List[str]] = {}
    for rec in records:
        seeds = [m.strip() for m in (rec.get("gold_entity_map") or {}).keys()
                 if m and not m.startswith("g.")]
        rec_seeds[rec["id"]] = seeds
        all_mids.update(seeds)

    log.info(f"  [{dataset}] {len(all_mids)} unique seed MIDs to expand")

    # Per-MID BFS + Virtuoso
    mid_cache: Dict[str, Tuple[Set, Set]] = {}
    done_mid = [0]
    t0 = time.time()

    def _work_mid(mid: str) -> Tuple[str, Set[str], Set]:
        ents, trips = bfs_fb(mid, max_hops, caps)
        # Virtuoso incoming enrichment
        extra = enrich_virtuoso_incoming(ents)
        for s, p, o in extra:
            trips.add((s, p, o))
            if _is_entity(s):
                ents.add(s)
            if _is_entity(o):
                ents.add(o)
        return mid, ents, trips

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_work_mid, m): m for m in sorted(all_mids)}
        for fut in as_completed(futs):
            mid, ents, trips = fut.result()
            mid_cache[mid] = (ents, trips)
            done_mid[0] += 1
            if done_mid[0] % 100 == 0 or done_mid[0] == len(all_mids):
                rate = done_mid[0] / max(time.time() - t0, 1e-9)
                log.info(f"  [{dataset}] BFS: {done_mid[0]}/{len(all_mids)} MIDs | {rate:.1f}/s")

    # Union per-question; cap triple set; write directly to per-question pkl files.
    # Writing to disk here (not returning qid_data) means the process can exit
    # and free all KB RAM before Stage 2 starts — the key RAM isolation benefit.
    s1_tmp_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  [{dataset}] Writing Stage 1 pkl files → {s1_tmp_dir}")
    empty = 0
    import random as _random
    for rec in records:
        qid = rec["id"]
        seeds = rec_seeds[qid]
        all_ents: Set[str] = set()
        all_trips: Set = set()
        for mid in seeds:
            if mid in mid_cache:
                e, t = mid_cache[mid]
                all_ents |= e
                all_trips |= t
        if len(all_trips) > S1_TRIPLE_CAP:
            all_trips = set(_random.sample(list(all_trips), S1_TRIPLE_CAP))
        if not all_ents and not all_trips:
            empty += 1
        p = s1_tmp_dir / f"{qid}.pkl"
        if not p.exists():
            with open(p, "wb") as fh:
                pickle.dump((all_ents, all_trips), fh, protocol=4)

    # Free mid_cache — no longer needed
    mid_cache.clear()
    del mid_cache
    gc.collect()

    log.info(f"[{dataset}] Stage 1 GraftNetV2 done: {empty} empty questions")


def stage1_nsm(
    dataset: str,
    records: List[dict],
    s1_tmp_dir: Path,
    n_workers: int = 1,  # kept for API compat; must be sequential — forking 120GB dict kills RAM
) -> None:
    """
    NSM Stage 1: per-MID 2-hop expansion, sequential with per-MID caching.
    Writes per-question pkl files to s1_tmp_dir and returns None.
    Caller exits after this; Stage 2 runs as a separate process reading from disk.

    Must be single-process: the _nsm_raw_fwd dict is ~120 GB. Forking N workers
    causes CoW page materialisation → N×120 GB RAM usage. ThreadPoolExecutor
    gives no speedup (GIL serialises all dict access). Sequential single-process
    with per-MID caching is the only viable approach.

    Optimisations:
      1. Per-MID cache: expand each unique seed MID once, union per question.
         (p7 CWQ: 2360 unique MIDs for 3531 Qs — many Qs share seeds.)
      2. Hop-1 cap (NSM_HOP1_CAP=500): prevents hop2 fanout explosion.
      3. S1_TRIPLE_CAP: caps per-question triple set before pickling.
    """
    log.info(f"[{dataset}] Stage 1 (NSM): {len(records)} records")
    load_nsm_kb()
    cvt = load_cvt_nodes()

    # Collect all unique seed MIDs across all records
    all_mids: Set[str] = set()
    rec_seeds: Dict[str, List[str]] = {}
    for rec in records:
        seeds = [m.strip() for m in (rec.get("gold_entity_map") or {}).keys()
                 if m and not m.startswith("g.")]
        rec_seeds[rec["id"]] = seeds
        all_mids.update(seeds)

    mid_list = sorted(all_mids)
    log.info(f"  [{dataset}] {len(mid_list)} unique seed MIDs → "
             f"sequential expansion (hop1_cap={NSM_HOP1_CAP}) …")

    # Per-MID cache: sequential, single-process
    mid_cache: Dict[str, Tuple[Set[str], Set]] = {}
    t0 = time.time()
    for i, mid in enumerate(mid_list):
        ents, trips = nsm_2hop_triples([mid], cvt)
        mid_cache[mid] = (ents, trips)
        if (i + 1) % 100 == 0 or (i + 1) == len(mid_list):
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            eta  = (len(mid_list) - i - 1) / max(rate, 1e-9)
            log.info(f"  [{dataset}] MID cache: {i+1}/{len(mid_list)} | "
                     f"{rate:.2f}/s | ETA {eta:.0f}s")

    log.info(f"  [{dataset}] MID cache built in {time.time()-t0:.1f}s")

    # Union per-question from cache; cap; write directly to per-question pkl files.
    # Writing to disk means Stage 2 can run as a separate process with no KB in RAM.
    s1_tmp_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  [{dataset}] Writing Stage 1 pkl files → {s1_tmp_dir}")
    empty = 0
    import random as _random
    for rec in records:
        qid   = rec["id"]
        seeds = rec_seeds[qid]
        q_ents:  Set[str] = set()
        q_trips: Set      = set()
        for mid in seeds:
            if mid in mid_cache:
                e, t = mid_cache[mid]
                q_ents  |= e
                q_trips |= t
        if len(q_trips) > S1_TRIPLE_CAP:
            q_trips = set(_random.sample(list(q_trips), S1_TRIPLE_CAP))
        if not q_ents and not q_trips:
            empty += 1
        p = s1_tmp_dir / f"{qid}.pkl"
        if not p.exists():
            with open(p, "wb") as fh:
                pickle.dump((q_ents, q_trips), fh, protocol=4)

    # Free mid_cache — no longer needed and large.
    mid_cache.clear()
    del mid_cache
    gc.collect()

    log.info(f"[{dataset}] Stage 1 NSM done: {empty} empty questions")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: per-question PPR pruning (multiprocess)
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 2 worker globals (small scalars only — no large dicts) ───────────────
_g_s1_tmp_dir: str = ""
_g_dataset: str = ""
_g_cap: int = 0
_g_out_dir: Path = Path(".")
_g_resume: bool = True
_g_variant: str = ""


def _init_stage2(s1_tmp_dir, dataset, cap, out_dir, resume, variant):
    global _g_s1_tmp_dir, _g_dataset, _g_cap, _g_out_dir, _g_resume, _g_variant
    _g_s1_tmp_dir = s1_tmp_dir
    _g_dataset    = dataset
    _g_cap        = cap
    _g_out_dir    = Path(out_dir)
    _g_resume     = resume
    _g_variant    = variant
    # Hard virtual-memory cap per worker: kills this process (not the machine)
    # if it grows beyond WORKER_VIRT_LIMIT_GB, avoiding system-wide OOM.
    try:
        limit = WORKER_VIRT_LIMIT_GB * (1 << 30)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass


def _build_one(record: dict) -> Tuple[str, str]:
    rid      = record["id"]
    out_path = _g_out_dir / f"{rid}.json"

    if _g_resume and out_path.exists():
        return rid, "skipped"

    seeds     = [m.strip() for m in (record.get("gold_entity_map") or {}).keys()
                 if m and not m.startswith("g.")]
    gold_rels = set((record.get("gold_relation_map") or {}).keys())
    ptype     = record.get("perturbation_type", "p7")

    # Load per-question raw triples from temp file written by Stage 1
    s1_file = Path(_g_s1_tmp_dir) / f"{rid}.pkl"
    all_ents:    Set[str] = set()
    all_triples: Set      = set()
    if s1_file.exists():
        try:
            with open(s1_file, "rb") as f:
                all_ents, all_triples = pickle.load(f)
        except Exception:
            pass

    if not all_ents and not all_triples:
        _write_atomic({
            "id": rid,
            "question": record.get("perturbed_question", ""),
            "original_question": record.get("original_question", ""),
            "answers": record.get("answers", []),
            "seed_mids": seeds, "entities": seeds, "triples": [],
            "perturbation_type": ptype,
        }, out_path)
        return rid, "empty"

    if _g_variant == "graftnetv2":
        if _g_dataset == "cwq":
            kept_ents = graftnetv2_ppr(all_triples, seeds, gold_rels, top_m=_g_cap)
        else:
            kept_ents = _pagerank_nibble(all_triples, seeds, top_m=_g_cap)
        kept_set  = set(kept_ents) | {s for s in seeds if s}
        kept_trips = [[s, r, o] for s, r, o in all_triples if s in kept_set and o in kept_set]
    else:
        cvt = load_cvt_nodes()
        kept_ents, nsm_trips = nsm_ppr(all_triples, seeds, cvt, top_m=_g_cap)
        kept_set  = set(kept_ents) | {s for s in seeds if s}
        kept_trips = [[s, r, o] for s, r, o in nsm_trips if s in kept_set and o in kept_set]

    _write_atomic({
        "id": rid,
        "question": record.get("perturbed_question", ""),
        "original_question": record.get("original_question", ""),
        "answers": record.get("answers", []),
        "seed_mids": seeds,
        "entities": sorted(kept_set),
        "triples": kept_trips,
        "perturbation_type": ptype,
    }, out_path)
    return rid, "done"


def stage2_ppr(
    dataset: str,
    variant: str,
    records: List[dict],
    s1_tmp_dir: Path,
    n_workers: int,
    resume: bool,
    out_dir: Path,
    cap: int,
) -> None:
    """Read per-question pkl files written by Stage 1 (already on disk).
    No KB in RAM — Stage 1 exited and freed all KB memory before this runs.
    Each worker reads only its own pkl file; parent process stays lean.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # For NSM: pre-load CVT nodes in parent before forking workers.
    # CoW means workers share the read-only CVT pages (5 GB) rather than
    # each loading it independently (20 workers × 5 GB = 100 GB extra).
    if variant == "nsm":
        load_cvt_nodes()

    n_pkls = sum(1 for _ in s1_tmp_dir.glob("*.pkl")) if s1_tmp_dir.exists() else 0
    log.info(f"[{dataset}] Stage 2 ({variant}): reading {n_pkls} pkl files from {s1_tmp_dir}")

    pending = [r for r in records if not (out_dir / f"{r['id']}.json").exists()] if resume else records
    log.info(f"[{dataset}] Stage 2 ({variant}): {len(pending)}/{len(records)} to process")
    if not pending:
        return

    done_n = [0]
    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_stage2,
        initargs=(str(s1_tmp_dir), dataset, cap, str(out_dir), resume, variant),
    ) as pool:
        futs = {pool.submit(_build_one, r): r["id"] for r in pending}
        for fut in as_completed(futs):
            rid = futs[fut]
            try:
                _, status = fut.result()
            except Exception as e:
                log.warning(f"  {rid}: ERROR {e}")
            done_n[0] += 1
            if done_n[0] % 100 == 0 or done_n[0] == len(pending):
                rate = done_n[0] / max(time.time() - t0, 1e-9)
                eta  = (len(pending) - done_n[0]) / max(rate, 1e-9)
                log.info(f"  [{dataset}] {done_n[0]}/{len(pending)} | {rate:.1f}/s | ETA {eta:.0f}s")

    # Clean up temp files after Stage 2 completes
    import shutil
    shutil.rmtree(s1_tmp_dir, ignore_errors=True)
    log.info(f"[{dataset}] Stage 1 temp dir cleaned: {s1_tmp_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: GNN-RAG integer encoding
# ══════════════════════════════════════════════════════════════════════════════

def stage3_gnn_encode(
    dataset: str,
    variant: str,
    records: List[dict],
    subgraph_dir: Path,
    gnn_out_dir: Path,
    ptype: str,
) -> None:
    gnn_out_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(subgraph_dir.glob("*.json"))
    log.info(f"[{dataset}] Stage 3 ({variant}): encoding {len(json_files)} subgraph files")

    subdir = DATASET_SUBDIR[dataset]
    ent2id: Dict[str, int] = {}
    with open(GNN_DATA / subdir / "entities.txt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            mid = line.strip()
            if mid:
                ent2id[mid] = i
    rel2id: Dict[str, int] = {}
    with open(GNN_DATA / subdir / "relations.txt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rel = line.strip()
            if rel:
                rel2id[rel] = i
    log.info(f"  [{dataset}] baseline vocab: {len(ent2id):,} ents, {len(rel2id):,} rels")

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
                s, r, o = str(t[0]), str(t[1]), str(t[2])
                if s and s not in ent2id: new_ents.add(s)
                if o and o not in ent2id: new_ents.add(o)
                if r and r not in rel2id: new_rels.add(r)
        except Exception:
            pass

    base_ent_count = len(ent2id)
    base_rel_count = len(rel2id)
    for i, e in enumerate(sorted(new_ents)):
        ent2id[e] = base_ent_count + i
    for i, r in enumerate(sorted(new_rels)):
        rel2id[r] = base_rel_count + i
    log.info(f"  [{dataset}] final: {len(ent2id):,} ents (+{len(new_ents)}), "
             f"{len(rel2id):,} rels (+{len(new_rels)})")

    ent_list = [""] * len(ent2id)
    for m, i in ent2id.items():
        ent_list[i] = m
    rel_list = [""] * len(rel2id)
    for r, i in rel2id.items():
        rel_list[i] = r
    (gnn_out_dir / "entities.txt").write_text("\n".join(ent_list) + "\n", encoding="utf-8")
    (gnn_out_dir / "relations.txt").write_text("\n".join(rel_list) + "\n", encoding="utf-8")

    out_path = gnn_out_dir / "test_subgraph.json"
    n_written = n_empty = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in records:
            rid = rec["id"]
            jp = subgraph_dir / f"{rid}.json"
            if not jp.exists():
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
            fout.write(json.dumps({
                "id": rid,
                "question": rec.get("perturbed_question", ""),
                "question_orig": rec.get("original_question", ""),
                "answers": rec.get("answers", []),
                "entities": seed_ids,
                "subgraph": {"tuples": tuples, "entities": all_ent_ids},
                "perturbation_type": ptype,
            }, ensure_ascii=False) + "\n")
            n_written += 1
            if not tuples:
                n_empty += 1

    log.info(f"  [{dataset}] wrote {n_written} records ({n_empty} empty) → {out_path}")

    emb_dir = PERT_DIR / "embeddings"
    for src, fname in [
        (emb_dir / f"{dataset}_pert_vocab.txt", "vocab.txt"),
        (emb_dir / f"{dataset}_pert_word_emb.npy", "word_emb.npy"),
    ]:
        dst = gnn_out_dir / fname
        if not dst.exists() and src.exists():
            os.symlink(src.resolve(), dst)

    log.info(f"  [{dataset}] output dir: {gnn_out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Variant configuration and orchestration
# ══════════════════════════════════════════════════════════════════════════════

VARIANT_CONFIG = {
    "p7_graftnetv2": {
        "ptype": "p7", "src_key": "p7",
        "subgraph_root": SUBGRAPH_ROOT,
        "subdir_name": "{dataset}_p7_v2",
        "gnn_name": "{dataset}_p7_graftnetv2",
        "mode": "graftnetv2",
    },
    "p7_nsm": {
        "ptype": "p7", "src_key": "p7",
        "subgraph_root": SUBGRAPH_ROOT,
        "subdir_name": "{dataset}_p7_nsm",
        "gnn_name": "{dataset}_p7_nsm",
        "mode": "nsm",
    },
    "p4_nsm": {
        "ptype": "p4", "src_key": "p4",
        "subgraph_root": SUBGRAPH_P4,
        "subdir_name": "{dataset}_p4_nsm",
        "gnn_name": "{dataset}_p4_nsm",
        "mode": "nsm",
    },
}


def run_variant(variant_name: str, dataset: str, records: List[dict],
                stages: List[str], args, resume: bool) -> None:
    cfg = VARIANT_CONFIG[variant_name]
    mode = cfg["mode"]
    ptype = cfg["ptype"]
    subgraph_dir = cfg["subgraph_root"] / cfg["subdir_name"].format(dataset=dataset)
    gnn_out_dir  = GNN_OUT_BASE / cfg["gnn_name"].format(dataset=dataset)

    # s1_tmp_dir is the on-disk handoff between Stage 1 and Stage 2.
    # Stage 1 writes per-question pkl files here; Stage 2 reads them.
    # Having this as a persistent directory means Stage 1 and Stage 2
    # can run as separate processes — the KB is freed between them.
    s1_tmp_dir = cfg["subgraph_root"] / f".s1_tmp_{dataset}_{mode}"

    log.info(f"\n{'═'*60}")
    log.info(f"[{dataset}] {variant_name} | mode={mode} | ptype={ptype}")
    log.info(f"  subgraph_dir : {subgraph_dir}")
    log.info(f"  gnn_out_dir  : {gnn_out_dir}")
    log.info(f"  s1_tmp_dir   : {s1_tmp_dir}")
    log.info(f"{'═'*60}")

    if "1" in stages:
        t0 = time.time()
        log.info(f"[{dataset}] === Stage 1 ===")
        if mode == "graftnetv2":
            hops = CWQ_HOPS if dataset == "cwq" else WEBQSP_HOPS
            caps = CWQ_CAPS if dataset == "cwq" else WEBQSP_CAPS
            stage1_graftnetv2(dataset, records, hops, caps, args.workers_s1, s1_tmp_dir)
        else:
            stage1_nsm(dataset, records, s1_tmp_dir, args.workers_s1)
        log.info(f"[{dataset}] Stage 1 done in {time.time()-t0:.1f}s")

    if "2" in stages:
        t0 = time.time()
        log.info(f"[{dataset}] === Stage 2 ===")
        cap = (CWQ_CAP_PPR if dataset == "cwq" else WEBQSP_CAP_PPR) if mode == "graftnetv2" else NSM_CAP_PPR
        w_s2 = args.workers_s2 if args.workers_s2 > 0 else args.workers
        stage2_ppr(dataset, mode, records, s1_tmp_dir, w_s2, resume,
                   subgraph_dir, cap)
        log.info(f"[{dataset}] Stage 2 done in {time.time()-t0:.1f}s")

    if "3" in stages:
        t0 = time.time()
        log.info(f"[{dataset}] === Stage 3 ===")
        stage3_gnn_encode(dataset, variant_name, records, subgraph_dir, gnn_out_dir, ptype)
        log.info(f"[{dataset}] Stage 3 done in {time.time()-t0:.1f}s")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["p7_graftnetv2","p7_nsm","p4_nsm","all"], default="all")
    ap.add_argument("--dataset", choices=["cwq","webqsp","both"], default="both")
    ap.add_argument("--stage",   choices=["1","2","3","all"], default="all")
    ap.add_argument("--workers",    type=int, default=200)
    ap.add_argument("--workers-s1", type=int, default=20)
    ap.add_argument("--workers-s2", type=int, default=0,
                    help="Stage 2 PPR workers (default: same as --workers)")
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--no-resume",  action="store_true")
    return ap.parse_args()


def main():
    args   = parse_args()
    resume = not args.no_resume
    stages = ["1","2","3"] if args.stage == "all" else [args.stage]

    ts = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(REPO_ROOT / "tmlr" / "logs" / f"build_variants_{ts}.log")
    log.info(f"variant={args.variant} dataset={args.dataset} stage={args.stage} "
             f"dry_run={args.dry_run}")

    variants = list(VARIANT_CONFIG.keys()) if args.variant == "all" else [args.variant]
    datasets = ["cwq","webqsp"] if args.dataset == "both" else [args.dataset]

    # Pre-load KBs once before iterating (NSM variants share the same KB)
    needs_fb  = any(VARIANT_CONFIG[v]["mode"] == "graftnetv2" for v in variants)
    needs_nsm = any(VARIANT_CONFIG[v]["mode"] == "nsm"         for v in variants)
    if "1" in stages:
        if needs_fb:
            load_fb_index(FB_EN)
        if needs_nsm:
            load_nsm_kb(MANUAL_FB)
            load_cvt_nodes(CVT_BIN)

    for variant_name in variants:
        cfg     = VARIANT_CONFIG[variant_name]
        src_key = cfg["src_key"]
        for dataset in datasets:
            src_path = PERT_DIR / f"{dataset}_{src_key}.json"
            if not src_path.exists():
                log.warning(f"Dataset not found: {src_path}")
                continue
            records = json.loads(src_path.read_text(encoding="utf-8"))
            if args.dry_run:
                records = records[:5]
            log.info(f"[{dataset}] {len(records)} records from {src_path.name}")
            run_variant(variant_name, dataset, records, stages, args, resume)

    log.info("\nAll done.")
    for variant_name in variants:
        cfg = VARIANT_CONFIG[variant_name]
        for dataset in datasets:
            out = GNN_OUT_BASE / cfg["gnn_name"].format(dataset=dataset) / "test_subgraph.json"
            if out.exists():
                log.info(f"  {out}: {sum(1 for _ in open(out))} lines")


if __name__ == "__main__":
    main()
