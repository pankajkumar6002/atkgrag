#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_graftnetv3.py
===================
GraftNetV3: question-embedding cosine similarity to weight per-relation
PPR matrices (Option A — averaged word_emb.npy vectors, same vocab as GNN-RAG).

Relation weighting: cos(q_emb, r_emb)^EXPONENT, where:
  - q_emb = mean of word vectors for perturbed question tokens (from cwq/webqsp_pert_word_emb.npy)
  - r_emb = mean of word vectors for relation surface text from gold_relation_map
  - EXPONENT = 2.0 (same as original GraftNet)

Seed strategy:
  - CWQ  p1:         SPARQL MIDs extracted from gold SPARQL (ns:m.xxx subjects) — different entity
  - CWQ  p2-p7:      gold_entity_map keys (each pert type has its own GEM from its SPARQL)
  - WebQSP p1:       SPARQL MIDs extracted from gold SPARQL (ns:m.xxx subjects in WHERE)
  - WebQSP p2-p7:    gold_entity_map keys (p2-p6 have own GEM; p1 has none)

Subgraph source: same fb_en.txt BFS as GraftNetV2 (reuses Stage 1 pkl files if present).
PPR: symmetric edges, question-cosine-weighted per-relation matrices.

Usage:
  python build_graftnetv3.py --ptype p7 --dataset both
  python build_graftnetv3.py --ptype p1 --dataset cwq
  python build_graftnetv3.py --ptype p4 --dataset webqsp
  python build_graftnetv3.py --ptype p7 --dataset both --stage 2  # reuse Stage 1 pkl
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = REPO_ROOT
PERT_DIR  = REPO_ROOT / "tmlr" / "pert" / "dataset"
GNN_DATA  = REPO_ROOT / "tmlr" / "Repo" / "GNN-RAG" / "data"
EMB_DIR   = PERT_DIR / "embeddings"
NEW_EXP   = REPO_ROOT / "new_exp"           # all outputs go here
FB_EN     = REPO_ROOT / "freebase" / "data" / "fb_en.txt"
TSV_NAMES = REPO_ROOT / "tmlr" / "resources" / "entities_sup_names.tsv"

# Existing graftnetv2 Stage2 dirs: MID-form triples already pruned, usable as S1 input
V2_SUBGRAPH_DIRS = {
    ("cwq",    "p7"): PERT_DIR / "subgraphs_p7" / "cwq_p7_v2",
    ("webqsp", "p7"): PERT_DIR / "subgraphs_p7" / "webqsp_p7_v2",
    ("cwq",    "p4"): PERT_DIR / "subgraphs_p4" / "cwq_p4_nsm",   # NSM s1 pkl reuse
    ("webqsp", "p4"): PERT_DIR / "subgraphs_p4" / "webqsp_p4_nsm",
}

DATASET_SUBDIR = {"cwq": "CWQ", "webqsp": "webqsp"}

# ── Hyperparameters ─────────────────────────────────────────────────────────────
PPR_RESTART  = 0.8
PPR_MAX_ITER = 20
PPR_EXPONENT = 2.0       # same as original GraftNet
PPR_NOTFOUND = 0.0       # score for relations not in gold_relation_map AND low cosine
COSINE_FLOOR = 0.0       # zero out negative cosine (treat as 0)

CWQ_CAP_PPR    = 2500
WEBQSP_CAP_PPR = 2000

BFS_CAPS_CWQ    = [500, 300, 100, 50]
BFS_HOPS_CWQ    = 4
BFS_CAPS_WEBQSP = [1000, 500]
BFS_HOPS_WEBQSP = 2

S1_TRIPLE_CAP   = 100_000
N_WORKERS       = 8

log = logging.getLogger("graftnetv3")


# ══════════════════════════════════════════════════════════════════════════════
# Word embedding helpers (Option A: average word_emb.npy)
# ══════════════════════════════════════════════════════════════════════════════

def load_word_emb(dataset: str):
    """Load vocab and word embedding matrix for dataset."""
    vocab_path = EMB_DIR / f"{dataset}_pert_vocab.txt"
    emb_path   = EMB_DIR / f"{dataset}_pert_word_emb.npy"
    vocab = {}
    with open(vocab_path) as f:
        for i, line in enumerate(f):
            w = line.strip()
            if w:
                vocab[w] = i
    emb = np.load(str(emb_path))   # shape (V, 300)
    log.info(f"  [{dataset}] word_emb: vocab={len(vocab)} dim={emb.shape[1]}")
    return vocab, emb


def text_to_vec(text: str, vocab: Dict[str, int], emb: np.ndarray) -> Optional[np.ndarray]:
    """Mean-pool word vectors for tokenized text. Returns None if no tokens found."""
    tokens = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    vecs = [emb[vocab[t]] for t in tokens if t in vocab]
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v


def compute_relation_scores(
    question: str,
    relation_map: Dict[str, str],  # rel_id -> rel_surface_text
    vocab: Dict[str, int],
    emb: np.ndarray,
) -> Dict[str, float]:
    """
    Returns score = max(cos(q_emb, r_emb), 0) for each relation.
    Relations in gold_relation_map get at least their cosine score.
    Relations NOT in relation_map get PPR_NOTFOUND (0.0).
    """
    q_vec = text_to_vec(question, vocab, emb)
    scores: Dict[str, float] = {}
    for rel_id, rel_text in relation_map.items():
        if q_vec is None:
            scores[rel_id] = 1.0   # fallback: uniform
            continue
        r_vec = text_to_vec(rel_text, vocab, emb)
        if r_vec is None:
            scores[rel_id] = PPR_NOTFOUND
            continue
        cos = float(np.dot(q_vec, r_vec))
        scores[rel_id] = max(cos, COSINE_FLOOR)
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# SPARQL entity extraction for p1 (no gold_entity_map)
# ══════════════════════════════════════════════════════════════════════════════

# Regex: matches ns:m.xxxxx or <http://rdf.freebase.com/ns/m.xxxxx>
_SPARQL_MID_RE = re.compile(r"ns:m\.([A-Za-z0-9_]+)")

def extract_mids_from_sparql(sparql: str) -> List[str]:
    """Extract topic-entity MIDs from SPARQL WHERE clause (ns:m.xxx subjects)."""
    if not sparql:
        return []
    # Only look in WHERE clause
    where = re.search(r"WHERE\s*\{(.*)\}", sparql, re.DOTALL)
    text = where.group(1) if where else sparql
    mids = list(dict.fromkeys(f"m.{m}" for m in _SPARQL_MID_RE.findall(text)))
    return mids


def build_mid_name_index(tsv_path: Path) -> Dict[str, str]:
    """Load entities_sup_names.tsv (MID → name) into memory."""
    log.info(f"Loading entity name index from {tsv_path} ...")
    idx: Dict[str, str] = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                idx[parts[0]] = parts[1]
    log.info(f"  Loaded {len(idx):,} MID→name entries")
    return idx


# ══════════════════════════════════════════════════════════════════════════════
# GraftNetV3 PPR — question-cosine-weighted per-relation matrices
# ══════════════════════════════════════════════════════════════════════════════

def _graftnetv3_read_facts(
    triples: Set[Tuple[str, str, str]],
    rel_scores: Dict[str, float],   # rel_id -> cosine score (0..1)
) -> Tuple[Dict[str, int], Optional[csr_matrix]]:
    """
    Symmetric per-relation sparse matrices weighted by question-cosine score.
    score(r) = cos(q_emb, r_emb)^EXPONENT if r in gold_relation_map else 0.0
    W = mean(all M_r), row-normalised.
    Symmetric: each (s,r,o) adds both s→o and o→s edges.
    """
    entity_map: Dict[str, int] = {}
    relation_map: Dict[str, Tuple[List[int], List[int]]] = {}
    num_entities = 0

    for s, r, o in triples:
        if not (s.startswith("m.") or s.startswith("g.")):
            continue
        if not (o.startswith("m.") or o.startswith("g.")):
            continue
        if s not in entity_map:
            entity_map[s] = num_entities; num_entities += 1
        if o not in entity_map:
            entity_map[o] = num_entities; num_entities += 1
        if r not in relation_map:
            relation_map[r] = ([], [])
        # Symmetric: s→o and o→s
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
        score = rel_scores.get(rel, PPR_NOTFOUND)
        per_rel[rel] = m * (score ** PPR_EXPONENT)

    adj_mat = sum(per_rel.values()) / len(per_rel)
    # If all scores were 0 (no gold relations found), fall back to uniform
    if adj_mat.nnz == 0:
        uniform_per_rel: Dict[str, csr_matrix] = {}
        for rel, (rows, cols) in relation_map.items():
            uniform_per_rel[rel] = normalize(
                csr_matrix(
                    (np.ones(len(rows)),
                     (np.array(rows, dtype=np.int32),
                      np.array(cols, dtype=np.int32))),
                    shape=(n, n),
                ), norm="l1", axis=1)
        adj_mat = sum(uniform_per_rel.values()) / len(uniform_per_rel)

    adj_mat = normalize(adj_mat, norm="l1", axis=1)
    return entity_map, adj_mat


def _graftnet_ppr(seed: np.ndarray, W: csr_matrix) -> np.ndarray:
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


def graftnetv3_ppr(
    triples: Set[Tuple[str, str, str]],
    seed_mids: List[str],
    rel_scores: Dict[str, float],
    top_m: int,
) -> List[str]:
    """Run GraftNetV3 PPR. Returns top_m entity MIDs by PPR score."""
    entity_map, adj_mat = _graftnetv3_read_facts(triples, rel_scores)
    if adj_mat is None or not entity_map or len(entity_map) < 2:
        return list(seed_mids)

    n = adj_mat.shape[0]
    seed_indices = [entity_map[m] for m in seed_mids if m in entity_map]
    seed = np.zeros((n, 1))
    if seed_indices:
        for i, idx in enumerate(seed_indices):
            seed[idx] = len(seed_indices) - i
        seed /= seed.sum()

    ppr = _graftnet_ppr(seed, adj_mat)
    sorted_idx = np.argsort(ppr)[::-1]

    # Keep top_m by PPR; trim near-zero
    extracted = sorted_idx[:top_m]
    zero_idx = np.where(ppr[extracted] < 1e-6)[0]
    if zero_idx.shape[0] > 0:
        extracted = extracted[:zero_idx[0]]

    inv_map = {v: k for k, v in entity_map.items()}
    return [inv_map[i] for i in extracted if i in inv_map]


# ══════════════════════════════════════════════════════════════════════════════
# fb_en.txt BFS (reused from graftnetv2 logic — same code, no duplication)
# ══════════════════════════════════════════════════════════════════════════════

_fb_fwd: Optional[Dict] = None
_fb_bwd: Optional[Dict] = None


def _load_fb_en():
    global _fb_fwd, _fb_bwd
    if _fb_fwd is not None:
        return
    log.info(f"Loading fb_en.txt (~5 min, large RAM) ...")
    t0 = time.time()
    fwd: Dict[str, List] = defaultdict(list)
    bwd: Dict[str, List] = defaultdict(list)
    with open(FB_EN, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            s, r, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if s.startswith("<") and s.endswith(">"):
                s = s[1:-1]
            if o.startswith("<") and o.endswith(">"):
                o = o[1:-1]
            if s.startswith("http://rdf.freebase.com/ns/"):
                s = s[len("http://rdf.freebase.com/ns/"):]
            if o.startswith("http://rdf.freebase.com/ns/"):
                o = o[len("http://rdf.freebase.com/ns/"):]
            if r.startswith("<") and r.endswith(">"):
                r = r[1:-1]
            if r.startswith("http://rdf.freebase.com/ns/"):
                r = r[len("http://rdf.freebase.com/ns/"):]
            if not (s.startswith("m.") or s.startswith("g.")):
                continue
            if not (o.startswith("m.") or o.startswith("g.")):
                continue
            fwd[s].append((r, o))
            bwd[o].append((r, s))
    _fb_fwd = dict(fwd)
    _fb_bwd = dict(bwd)
    log.info(f"  fb_en.txt ready: {len(_fb_fwd):,} subjects in {time.time()-t0:.1f}s")


def _free_fb_en():
    global _fb_fwd, _fb_bwd
    _fb_fwd = None
    _fb_bwd = None
    gc.collect()


def _bfs_one_mid(mid: str, hops: int, caps: List[int]) -> Set[Tuple[str, str, str]]:
    """BFS from mid using fb_en.txt (both fwd and bwd). Returns triple set."""
    triples: Set[Tuple[str, str, str]] = set()
    frontier = {mid}
    visited  = {mid}
    for hop in range(hops):
        cap = caps[hop] if hop < len(caps) else caps[-1]
        next_frontier: Set[str] = set()
        for node in frontier:
            nbrs = []
            for r, o in (_fb_fwd.get(node) or []):
                nbrs.append((node, r, o, o))
            for r, s in (_fb_bwd.get(node) or []):
                nbrs.append((s, r, node, s))
            # degree-sort: prefer low-degree neighbors (original GraftNet heuristic)
            nbrs.sort(key=lambda x: len(_fb_fwd.get(x[3], [])) + len(_fb_bwd.get(x[3], [])))
            for s, r, o, nxt in nbrs[:cap]:
                triples.add((s, r, o))
                if nxt not in visited:
                    visited.add(nxt)
                    next_frontier.add(nxt)
        frontier = next_frontier
        if not frontier:
            break
    return triples


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: BFS expansion (per-question pkl)
# ══════════════════════════════════════════════════════════════════════════════

def stage1_from_v2(
    records: List[dict],
    dataset: str,
    ptype: str,
    s1_dir: Path,
    resume: bool,
) -> List[dict]:
    """
    Fast path: build Stage1 pkls from existing graftnetv2/NSM Stage2 JSON files
    (which store MID-form triples). Saves ~5 hours of fb_en.txt BFS.
    Returns list of records that had empty triples and need BFS fallback.
    Returns all records (signals full BFS needed) if v2 dir is unavailable.
    """
    v2_dir = V2_SUBGRAPH_DIRS.get((dataset, ptype))
    if not v2_dir or not v2_dir.exists():
        return records  # all need BFS
    v2_files = list(v2_dir.glob("*.json"))
    if not v2_files:
        return records  # all need BFS

    log.info(f"  [{dataset}] Stage1 fast-path: loading triples from {v2_dir} ({len(v2_files)} files)")
    s1_dir.mkdir(parents=True, exist_ok=True)

    # Build MID → triples index from v2 JSON files
    mid_to_triples: Dict[str, Set[Tuple[str, str, str]]] = {}
    for jf in v2_files:
        try:
            d = json.loads(jf.read_text())
            seed_mids = d.get("seed_mids") or d.get("seeds") or []
            trips = {(str(t[0]), str(t[1]), str(t[2])) for t in d.get("triples", []) if len(t) >= 3}
            for mid in seed_mids:
                if mid not in mid_to_triples:
                    mid_to_triples[mid] = set()
                mid_to_triples[mid] |= trips
        except Exception:
            continue
    log.info(f"  [{dataset}] v2 index: {len(mid_to_triples)} unique MIDs with triples")

    done = skipped = empty = 0
    needs_bfs: List[dict] = []
    for rec in records:
        rid = rec["id"]
        pkl_path = s1_dir / f"{rid}.pkl"
        if resume and pkl_path.exists():
            # Verify pkl is non-empty; if empty, still needs BFS
            try:
                with open(pkl_path, "rb") as f:
                    _, existing_triples = pickle.load(f)
                if existing_triples:
                    skipped += 1; done += 1; continue
                # Empty pkl — fall through to rewrite or queue BFS
            except Exception:
                pass

        seeds = _get_seeds(rec, ptype, None)
        all_triples: Set[Tuple[str, str, str]] = set()
        all_ents: Set[str] = set(seeds)
        for mid in seeds:
            t = mid_to_triples.get(mid, set())
            all_triples |= t
            for s, r, o in t:
                all_ents.add(s); all_ents.add(o)

        if not all_triples:
            empty += 1
            needs_bfs.append(rec)
            # Write empty pkl as placeholder so stage2 can detect it
            with open(pkl_path, "wb") as f:
                pickle.dump((all_ents, all_triples), f, protocol=4)
        else:
            if len(all_triples) > S1_TRIPLE_CAP:
                all_triples = set(list(all_triples)[:S1_TRIPLE_CAP])
            with open(pkl_path, "wb") as f:
                pickle.dump((all_ents, all_triples), f, protocol=4)
        done += 1
        if done % 200 == 0:
            log.info(f"  [{dataset}] Stage1 fast-path: {done}/{len(records)}")

    log.info(f"  [{dataset}] Stage1 fast-path done: {done} records, {empty} empty "
             f"(queued for BFS fallback), {skipped} skipped")
    return needs_bfs


def stage1_bfs(
    records: List[dict],
    ptype: str,
    dataset: str,
    s1_dir: Path,
    resume: bool,
    mid_name_idx: Optional[Dict[str, str]] = None,
) -> None:
    """
    BFS from seed MIDs using fb_en.txt. Deduplicates by MID so each unique
    seed MID is only expanded once. Writes per-question pkl to s1_dir.
    """
    _load_fb_en()
    s1_dir.mkdir(parents=True, exist_ok=True)
    hops = BFS_HOPS_CWQ if dataset == "cwq" else BFS_HOPS_WEBQSP
    caps = BFS_CAPS_CWQ if dataset == "cwq" else BFS_CAPS_WEBQSP

    # Collect all unique seed MIDs across all records, BFS once per MID
    all_seeds: Set[str] = set()
    rec_seeds: Dict[str, List[str]] = {}
    for rec in records:
        rid = rec["id"]
        pkl_path = s1_dir / f"{rid}.pkl"
        if resume and pkl_path.exists():
            continue
        seeds = _get_seeds(rec, ptype, mid_name_idx)
        rec_seeds[rid] = seeds
        all_seeds.update(seeds)

    n_already = len(records) - len(rec_seeds)
    log.info(f"  [{dataset}] Stage1 BFS: {len(all_seeds)} unique MIDs to expand "
             f"({n_already} records already done, resuming)")

    # BFS per unique MID
    mid_triples: Dict[str, Set[Tuple[str, str, str]]] = {}
    t0 = time.time()
    for i, mid in enumerate(sorted(all_seeds)):
        mid_triples[mid] = _bfs_one_mid(mid, hops, caps)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(all_seeds) - i - 1) / rate if rate > 0 else 0
            log.info(f"  [{dataset}] BFS: {i+1}/{len(all_seeds)} MIDs | "
                     f"{rate:.2f}/s | ETA {eta/60:.1f}m")

    # Write per-question pkls
    done = 0
    for rec in records:
        rid = rec["id"]
        pkl_path = s1_dir / f"{rid}.pkl"
        if resume and pkl_path.exists():
            done += 1; continue
        seeds = rec_seeds.get(rid, [])
        all_triples: Set[Tuple[str, str, str]] = set()
        all_ents: Set[str] = set(seeds)
        for mid in seeds:
            t = mid_triples.get(mid, set())
            all_triples |= t
            for s, r, o in t:
                all_ents.add(s); all_ents.add(o)
        if len(all_triples) > S1_TRIPLE_CAP:
            all_triples = set(list(all_triples)[:S1_TRIPLE_CAP])
        with open(pkl_path, "wb") as f:
            pickle.dump((all_ents, all_triples), f, protocol=4)
        done += 1

    log.info(f"  [{dataset}] Stage1 BFS done: {done}/{len(records)} pkl files in {s1_dir}")


def _get_seeds(
    rec: dict,
    ptype: str,
    mid_name_idx: Optional[Dict[str, str]],
) -> List[str]:
    """Get seed MIDs for a record based on ptype."""
    gem = rec.get("gold_entity_map") or {}
    seeds = [m for m in gem.keys() if m and not m.startswith("g.")]
    if seeds:
        return seeds
    # Fallback: extract from SPARQL (p1 or WebQSP records without GEM)
    sparql = rec.get("sparql", "")
    seeds = extract_mids_from_sparql(sparql)
    return seeds


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: PPR (parallel, one pkl per question)
# ══════════════════════════════════════════════════════════════════════════════

# Worker globals
_g_s1_dir: Optional[Path] = None
_g_out_dir: Optional[Path] = None
_g_vocab: Optional[Dict] = None
_g_emb: Optional[np.ndarray] = None
_g_cap: int = 2000
_g_resume: bool = True


def _init_worker(s1_dir, out_dir, vocab, emb, cap, resume):
    global _g_s1_dir, _g_out_dir, _g_vocab, _g_emb, _g_cap, _g_resume
    _g_s1_dir = s1_dir
    _g_out_dir = out_dir
    _g_vocab = vocab
    _g_emb = emb
    _g_cap = cap
    _g_resume = resume


def _fmt_answers(answers) -> list:
    """Normalize answers to [{"kb_id": mid, "text": mid}] format."""
    if not answers:
        return []
    out = []
    for a in answers:
        if isinstance(a, dict):
            out.append(a)
        elif isinstance(a, str):
            out.append({"kb_id": a, "text": a})
    return out


def _ppr_one(record: dict) -> Tuple[str, str]:
    rid = record["id"]
    out_path = _g_out_dir / f"{rid}.json"
    if _g_resume and out_path.exists():
        return rid, "skipped"

    seeds = _get_seeds(record, record.get("perturbation_type", ""), None)
    gold_rel_map = record.get("gold_relation_map") or {}

    # Load pkl
    pkl = _g_s1_dir / f"{rid}.pkl"
    all_ents: Set[str] = set()
    all_triples: Set = set()
    if pkl.exists():
        try:
            with open(pkl, "rb") as f:
                all_ents, all_triples = pickle.load(f)
        except Exception:
            pass

    answers = _fmt_answers(record.get("answers", []))
    if not all_triples:
        _write_json(out_path, {
            "id": rid,
            "question": record.get("perturbed_question", ""),
            "original_question": record.get("original_question", ""),
            "answers": answers,
            "seed_mids": seeds, "entities": seeds, "triples": [],
            "perturbation_type": record.get("perturbation_type", ""),
        })
        return rid, "empty"

    # Compute relation scores from question embedding
    question = record.get("perturbed_question", "")
    rel_scores = compute_relation_scores(question, gold_rel_map, _g_vocab, _g_emb)

    # PPR
    kept_ents = graftnetv3_ppr(all_triples, seeds, rel_scores, top_m=_g_cap)
    kept_set = set(kept_ents) | {s for s in seeds if s}
    kept_trips = [[s, r, o] for s, r, o in all_triples if s in kept_set and o in kept_set]

    _write_json(out_path, {
        "id": rid,
        "question": question,
        "original_question": record.get("original_question", ""),
        "answers": answers,
        "seed_mids": seeds,
        "entities": sorted(kept_set),
        "triples": kept_trips,
        "perturbation_type": record.get("perturbation_type", ""),
    })
    return rid, "done"


def _write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def stage2_ppr(
    records: List[dict],
    dataset: str,
    s1_dir: Path,
    out_dir: Path,
    vocab: Dict,
    emb: np.ndarray,
    cap: int,
    resume: bool,
    n_workers: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  [{dataset}] Stage2 PPR: {len(records)} records, {n_workers} workers")
    done = skipped = empty = 0
    failed_records: List[dict] = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(s1_dir, out_dir, vocab, emb, cap, resume),
    ) as ex:
        futs = {ex.submit(_ppr_one, rec): rec for rec in records}
        for fut in as_completed(futs):
            rec = futs[fut]
            try:
                rid, status = fut.result()
                if status == "skipped": skipped += 1
                elif status == "empty": empty += 1
                else: done += 1
            except Exception as e:
                log.warning(f"  PPR worker error [{rec['id']}]: {e} — queued for retry")
                failed_records.append(rec)

    # Retry failures sequentially in main process
    if failed_records:
        log.info(f"  [{dataset}] Retrying {len(failed_records)} failed records sequentially")
        _init_worker(s1_dir, out_dir, vocab, emb, cap, False)
        for rec in failed_records:
            try:
                rid, status = _ppr_one(rec)
                if status == "empty": empty += 1
                else: done += 1
            except Exception as e:
                log.warning(f"  Sequential retry also failed [{rec['id']}]: {e}")

    log.info(f"  [{dataset}] Stage2 done: {done} ok, {empty} empty, {skipped} skipped, "
             f"{len(failed_records)} retried")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: Integer encoding → test_subgraph.json
# ══════════════════════════════════════════════════════════════════════════════

def stage3_encode(
    records: List[dict],
    dataset: str,
    subgraph_dir: Path,
    gnn_out_dir: Path,
    ptype: str,
) -> None:
    subdir = DATASET_SUBDIR[dataset]
    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}

    with open(GNN_DATA / subdir / "entities.txt") as f:
        for i, line in enumerate(f):
            e = line.strip()
            if e: ent2id[e] = i
    with open(GNN_DATA / subdir / "relations.txt") as f:
        for i, line in enumerate(f):
            r = line.strip()
            if r: rel2id[r] = i

    # Count OOV stats for logging (do NOT extend — GNN-RAG embedding matrix is fixed-size)
    n_oov_ent = n_oov_rel = 0
    for rec in records:
        jp = subgraph_dir / f"{rec['id']}.json"
        if not jp.exists(): continue
        d = json.loads(jp.read_text())
        for m in d.get("entities", []):
            if m and m not in ent2id: n_oov_ent += 1
        for t in d.get("triples", []):
            if len(t) >= 3 and t[1] not in rel2id: n_oov_rel += 1
    log.info(f"  [{dataset}] vocab: {len(ent2id):,} ents, {len(rel2id):,} rels "
             f"| OOV (dropped): {n_oov_ent} ent tokens, {n_oov_rel} rel tokens")

    # Symlink entities.txt / relations.txt from original GNN-RAG vocab (not extended)
    gnn_out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("entities.txt", "relations.txt"):
        src = GNN_DATA / subdir / fname
        dst = gnn_out_dir / fname
        if not dst.exists():
            os.symlink(src.resolve(), dst)

    out_path = gnn_out_dir / "test_subgraph.json"
    n_written = n_empty = 0
    with open(out_path, "w") as fout:
        for rec in records:
            rid = rec["id"]
            jp = subgraph_dir / f"{rid}.json"
            if not jp.exists(): continue
            d = json.loads(jp.read_text())
            seed_ids = [ent2id[m] for m in d.get("seed_mids", []) if m in ent2id]
            all_ent_ids = sorted(set(ent2id[e] for e in d.get("entities", []) if e in ent2id))
            tuples = []
            for t in d.get("triples", []):
                if len(t) < 3: continue
                s, r, o = str(t[0]), str(t[1]), str(t[2])
                if s in ent2id and r in rel2id and o in ent2id:
                    tuples.append([ent2id[s], rel2id[r], ent2id[o]])
            fout.write(json.dumps({
                "id": rid,
                "question": rec.get("perturbed_question", ""),
                "question_orig": rec.get("original_question", ""),
                "answers": _fmt_answers(rec.get("answers", [])),
                "entities": seed_ids,
                "subgraph": {"tuples": tuples, "entities": all_ent_ids},
                "perturbation_type": ptype,
            }, ensure_ascii=False) + "\n")
            n_written += 1
            if not tuples: n_empty += 1

    log.info(f"  [{dataset}] wrote {n_written} records ({n_empty} empty) → {out_path}")

    # Symlink vocab/word_emb
    for src_name, dst_name in [
        (f"{dataset}_pert_vocab.txt", "vocab.txt"),
        (f"{dataset}_pert_word_emb.npy", "word_emb.npy"),
    ]:
        src = EMB_DIR / src_name
        dst = gnn_out_dir / dst_name
        if not dst.exists() and src.exists():
            os.symlink(src.resolve(), dst)


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════════

def run(ptype: str, dataset: str, stages: List[str], resume: bool, n_workers: int,
        use_v2: bool, out_root: Optional[Path] = None):
    datasets = ["cwq", "webqsp"] if dataset == "both" else [dataset]

    for ds in datasets:
        log.info(f"\n{'═'*60}")
        log.info(f"GraftNetV3  ptype={ptype}  dataset={ds}")
        log.info(f"{'═'*60}")

        # Load dataset
        pert_file = PERT_DIR / f"{ds}_{ptype}.json"
        records: List[dict] = json.load(open(pert_file))
        log.info(f"  Loaded {len(records)} records from {pert_file.name}")

        # Output root: custom --out-dir or default new_exp/
        root = out_root if out_root else NEW_EXP
        s1_dir       = root / "subgraphs" / f".s1_{ds}_{ptype}_graftnetv3"
        subgraph_dir = root / "subgraphs" / f"{ds}_{ptype}_graftnetv3"
        gnn_out_dir  = root / "gnn_input"  / f"{ds}_{ptype}_graftnetv3"
        cap = CWQ_CAP_PPR if ds == "cwq" else WEBQSP_CAP_PPR

        # For p1: load entity name index for SPARQL MID extraction
        mid_name_idx = None
        if ptype == "p1":
            mid_name_idx = build_mid_name_index(TSV_NAMES)

        if "1" in stages:
            if use_v2:
                log.info(f"\n[Stage 1] Fast-path from v2 JSONs → {s1_dir}")
                bfs_fallback_records = stage1_from_v2(records, ds, ptype, s1_dir, resume)
                if len(bfs_fallback_records) == len(records):
                    log.info(f"  v2 fast-path unavailable for ({ds}, {ptype}) — running full BFS")
                    stage1_bfs(records, ptype, ds, s1_dir, resume, mid_name_idx)
                    _free_fb_en()
                elif bfs_fallback_records:
                    log.info(f"\n[Stage 1] BFS fallback for {len(bfs_fallback_records)} empty records → {s1_dir}")
                    stage1_bfs(bfs_fallback_records, ptype, ds, s1_dir, False, mid_name_idx)
                    _free_fb_en()
                    log.info("  fb_en.txt freed from memory")
                else:
                    log.info(f"  v2 fast-path complete — no BFS needed")
            else:
                log.info(f"\n[Stage 1] BFS from fb_en.txt → {s1_dir}")
                stage1_bfs(records, ptype, ds, s1_dir, resume, mid_name_idx)
                _free_fb_en()
                log.info("  fb_en.txt freed from memory")

        if "2" in stages:
            log.info(f"\n[Stage 2] GraftNetV3 PPR → {subgraph_dir}")
            vocab, emb = load_word_emb(ds)
            stage2_ppr(records, ds, s1_dir, subgraph_dir, vocab, emb, cap, resume, n_workers)
            del vocab, emb; gc.collect()
            log.info("  word embeddings freed from memory")

        if "3" in stages:
            log.info(f"\n[Stage 3] Integer encode → {gnn_out_dir}")
            stage3_encode(records, ds, subgraph_dir, gnn_out_dir, ptype)

        log.info(f"\n  [{ds}] GraftNetV3 {ptype} complete → {gnn_out_dir}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    p = argparse.ArgumentParser()
    p.add_argument("--ptype",    required=True,
                   help="perturbation type: p1, p4, p7, or all")
    p.add_argument("--dataset",  default="both",
                   choices=["cwq", "webqsp", "both"])
    p.add_argument("--stage",    default="123",
                   help="stages to run: any combo of 1,2,3 (default: 123)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers",  type=int, default=N_WORKERS)
    p.add_argument("--use-v2",   action="store_true",
                   help="Stage1 fast-path: load triples from existing graftnetv2/NSM JSON files "
                        "(saves ~5h BFS for p4/p7; falls back to BFS for empty records)")
    p.add_argument("--out-dir",  type=str, default="",
                   help="Root output directory (default: new_exp/). "
                        "Outputs: <out-dir>/subgraphs/<ds>_<ptype>_graftnetv3/ and "
                        "<out-dir>/gnn_input/<ds>_<ptype>_graftnetv3/")
    args = p.parse_args()

    ptypes = ["p1", "p4", "p7"] if args.ptype == "all" else [args.ptype]
    stages = list(args.stage)
    resume = not args.no_resume
    out_root = Path(args.out_dir) if args.out_dir else None

    for pt in ptypes:
        run(pt, args.dataset, stages, resume, args.workers, args.use_v2, out_root)


if __name__ == "__main__":
    main()