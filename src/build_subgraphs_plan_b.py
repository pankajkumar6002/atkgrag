#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
Plan B: Build dense subgraphs using ELQ-detected seed entities with GPU-accelerated BFS.

Uses subgraph_2hop_triple.npy (190M int32 triples, pre-built Freebase index) loaded
directly onto GPU (A100 80GB) via PyTorch for fast CSR-based BFS traversal.
Avoids slow text parsing entirely.

Hop depth:   WebQSP=2, CWQ=4
Entity cap:  MAX_ENTITIES=2000 per neighborhood (degree-weighted Approximate PPR)
Output:      Pre-process/plan_b/{dataset}/{strategy}/

Usage:
    conda run -n perturb python build_subgraphs_plan_b.py [--dataset cwq|webqsp|both]
    conda run -n perturb python build_subgraphs_plan_b.py --dataset webqsp --strategy s1_active_passive
    conda run -n perturb python build_subgraphs_plan_b.py --skip_kg_load  # cache-only mode
"""

import argparse
import csv
import json
import os
import pickle
import re
import sys
from collections import OrderedDict

import numpy as np
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = str(Path(__file__).resolve().parents[1])
GNN_DIR     = os.path.join(REPO_ROOT, "GNN")
RESULTS_DIR = os.path.join(GNN_DIR, "results")
OUTPUT_BASE = os.path.join(REPO_ROOT, "Pre-process", "plan_b")
MID_CACHE   = os.path.join(REPO_ROOT, "Pre-process", "mid_cache")
GLOVE_PATH  = os.path.join(REPO_ROOT, "freebase/GraftNet/preprocessing/downloads/glove.840B.300d.txt")

KG_NPY        = os.path.join(REPO_ROOT, "UniKGQA/data/MyFreebaseCache/subgraph_2hop_triple.npy")
ENT2ID_PKL    = os.path.join(REPO_ROOT, "UniKGQA/data/MyFreebaseCache/ent2id.pickle")
RELATIONS_TXT = os.path.join(REPO_ROOT, "UniKGQA/data/MyFreebaseCache/relations.txt")

ENTITY_LIST = {
    "cwq":    os.path.join(RESULTS_DIR, "entity_list_cwq.csv"),
    "webqsp": os.path.join(RESULTS_DIR, "entity_list_webqsp.csv"),
}
DATASETS = {
    "cwq":    {"orig_json": os.path.join(GNN_DIR, "CWQ/test.json"),    "hops": 4},
    "webqsp": {"orig_json": os.path.join(GNN_DIR, "webqsp/test.json"), "hops": 2},
}
STRATEGIES = [
    "s1_active_passive", "s2_reorder", "s3_relation_synonym",
    "s4_entity_alias", "s5_adversarial_para", "s6_grammatical_perturb",
    "s7_filler_injection",
]

MAX_ENTITIES = 2000
GLOVE_DIM    = 300
MID_PATTERN  = re.compile(r"^m\.[a-z0-9_]+$")
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def is_valid_mid(s):
    return isinstance(s, str) and bool(MID_PATTERN.match(s))


def load_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_orig_by_id(path):
    orig = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                orig[e["id"]] = e
    return orig


def load_seed_mids_from_csv(csv_path):
    seeds = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            mid = row.get("Freebase MID", "").strip()
            if is_valid_mid(mid):
                seeds.add(mid)
    return seeds


# ─── GPU KG Loading ───────────────────────────────────────────────────────────

class FreebaseKG:
    """
    GPU-accelerated Freebase KG using PyTorch CSR sparse matrices.
    Loads subgraph_2hop_triple.npy (190M × 3 int32) directly onto GPU.
    """

    def __init__(self, npy_path, ent2id_path, relations_txt, device=DEVICE):
        self.device = device

        # Load ent2id (MID string → integer index)
        print(f"  Loading ent2id from {ent2id_path} ...", flush=True)
        with open(ent2id_path, "rb") as f:
            self.ent2id = pickle.load(f)
        # Build reverse: int → MID string
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        print(f"  ent2id: {len(self.ent2id)} entities.", flush=True)

        # Load relations: integer rel_id → relation name string
        print(f"  Loading relations from {relations_txt} ...", flush=True)
        self.id2rel = []
        with open(relations_txt) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.id2rel.append(line)
        print(f"  Relations: {len(self.id2rel)} loaded.", flush=True)

        # Load triple array onto GPU
        print(f"  Loading {npy_path} onto {device} ...", flush=True)
        triples_np = np.load(npy_path)         # shape: (N, 3), dtype int32
        print(f"  Triples shape: {triples_np.shape}", flush=True)

        self.triples = torch.from_numpy(triples_np).to(torch.int32).to(device)
        # columns: heads, rels, tails
        self.heads = self.triples[:, 0].long()
        self.rels  = self.triples[:, 1].long()
        self.tails = self.triples[:, 2].long()
        self.N     = triples_np.shape[0]
        self.n_ent = len(self.ent2id)

        # Compute entity degree (in+out) for Approximate PPR pruning — stays on GPU
        print("  Computing entity degrees ...", flush=True)
        out_degree = torch.zeros(self.n_ent + 1, dtype=torch.long, device=device)
        in_degree  = torch.zeros(self.n_ent + 1, dtype=torch.long, device=device)
        safe_heads = self.heads.clamp(0, self.n_ent)
        safe_tails = self.tails.clamp(0, self.n_ent)
        out_degree.scatter_add_(0, safe_heads, torch.ones(self.N, dtype=torch.long, device=device))
        in_degree.scatter_add_(0, safe_tails, torch.ones(self.N, dtype=torch.long, device=device))
        self.degree = (out_degree + in_degree)[:self.n_ent]   # shape: (n_ent,)
        print("  Degrees computed.", flush=True)

        # Sort triples by head for fast contiguous head lookup
        print("  Sorting triples by head (for BFS) ...", flush=True)
        sort_idx = torch.argsort(self.heads)
        self.sorted_heads = self.heads[sort_idx]
        self.sorted_rels  = self.rels[sort_idx]
        self.sorted_tails = self.tails[sort_idx]

        # Compute head→range index (like CSR row pointers)
        # head_ptr[h] = start index in sorted arrays for head h
        self.head_ptr = torch.zeros(self.n_ent + 1, dtype=torch.long, device=device)
        count = torch.bincount(self.sorted_heads.clamp(0, self.n_ent - 1), minlength=self.n_ent)
        self.head_ptr[1:] = count.cumsum(0)

        # Sort by tail for backward BFS
        print("  Sorting triples by tail (for backward BFS) ...", flush=True)
        sort_idx_t = torch.argsort(self.tails)
        self.sorted_tails_bwd  = self.tails[sort_idx_t]
        self.sorted_heads_bwd  = self.heads[sort_idx_t]
        self.sorted_rels_bwd   = self.rels[sort_idx_t]
        self.tail_ptr = torch.zeros(self.n_ent + 1, dtype=torch.long, device=device)
        count_t = torch.bincount(self.sorted_tails_bwd.clamp(0, self.n_ent - 1), minlength=self.n_ent)
        self.tail_ptr[1:] = count_t.cumsum(0)

        print("  KG ready on GPU.", flush=True)

    def mid_to_id(self, mid):
        return self.ent2id.get(mid, None)

    def id_to_mid(self, idx):
        return self.id2ent.get(int(idx), None)

    def _ranges_to_idx(self, starts, ends):
        """Vectorized: build flat index array from (starts, ends) pairs on GPU.
        Replaces the Python-level [torch.arange(s,e) for s,e in ...] loop."""
        lengths = (ends - starts).clamp(min=0)
        total = int(lengths.sum().item())
        if total == 0:
            return None
        # Cumulative lengths for computing per-element offsets
        cum = torch.zeros(lengths.shape[0] + 1, dtype=torch.long, device=self.device)
        cum[1:] = lengths.cumsum(0)
        # For each output position k, which group does it belong to?
        # starts_rep[k] = starts[group(k)], offset[k] = k - cum[group(k)]
        starts_rep = torch.repeat_interleave(starts, lengths)
        global_arange = torch.arange(total, dtype=torch.long, device=self.device)
        group_start = torch.repeat_interleave(cum[:-1], lengths)
        return starts_rep + (global_arange - group_start)

    def _forward_triples(self, entity_ids_tensor):
        """Fetch all forward triples (h, r, t) where h ∈ entity_ids_tensor."""
        entity_ids_tensor = entity_ids_tensor.clamp(0, self.n_ent - 1)
        starts = self.head_ptr[entity_ids_tensor]
        ends   = self.head_ptr[entity_ids_tensor + 1]
        idx = self._ranges_to_idx(starts, ends)
        if idx is None:
            return torch.empty((0, 3), dtype=torch.long, device=self.device)
        return torch.stack([self.sorted_heads[idx].long(),
                            self.sorted_rels[idx].long(),
                            self.sorted_tails[idx].long()], dim=1)

    def _backward_triples(self, entity_ids_tensor):
        """Fetch all backward triples (h, r, t) where t ∈ entity_ids_tensor."""
        entity_ids_tensor = entity_ids_tensor.clamp(0, self.n_ent - 1)
        starts = self.tail_ptr[entity_ids_tensor]
        ends   = self.tail_ptr[entity_ids_tensor + 1]
        idx = self._ranges_to_idx(starts, ends)
        if idx is None:
            return torch.empty((0, 3), dtype=torch.long, device=self.device)
        return torch.stack([self.sorted_heads_bwd[idx].long(),
                            self.sorted_rels_bwd[idx].long(),
                            self.sorted_tails_bwd[idx].long()], dim=1)

    def extract_neighborhood(self, seed_mid, max_hops, max_entities=MAX_ENTITIES):
        """
        BFS n-hop neighborhood from seed_mid using degree-weighted pruning.
        Returns:
            entities: list of MID strings
            tuples:   list of (head_mid, rel_int, tail_mid) — rel as integer
        """
        seed_id = self.mid_to_id(seed_mid)
        if seed_id is None:
            return [seed_mid], []

        entity_set_ids = torch.tensor([seed_id], dtype=torch.long, device=self.device)
        all_triples    = torch.empty((0, 3), dtype=torch.long, device=self.device)
        frontier_ids   = entity_set_ids.clone()

        for hop in range(max_hops):
            fwd = self._forward_triples(frontier_ids)
            bwd = self._backward_triples(frontier_ids)
            hop_triples = torch.cat([fwd, bwd], dim=0) if fwd.shape[0] + bwd.shape[0] > 0 else \
                          (fwd if fwd.shape[0] > 0 else bwd)

            if hop_triples.shape[0] == 0:
                break

            all_triples = torch.cat([all_triples, hop_triples], dim=0)

            # Candidate new entities = all heads + tails not yet visited
            new_cands = torch.cat([hop_triples[:, 0], hop_triples[:, 2]])
            new_cands = new_cands.unique()
            # Filter to not-yet-visited
            visited_mask = torch.isin(new_cands, entity_set_ids)
            new_cands = new_cands[~visited_mask]

            if new_cands.shape[0] == 0:
                break

            # Degree-weighted pruning: keep top max_entities by degree
            remaining = max_entities - entity_set_ids.shape[0]
            if remaining <= 0:
                break
            if new_cands.shape[0] > remaining:
                cand_degrees = self.degree[new_cands.clamp(0, self.n_ent - 1)]
                _, top_idx = torch.topk(cand_degrees, min(remaining, new_cands.shape[0]))
                new_cands = new_cands[top_idx]

            entity_set_ids = torch.cat([entity_set_ids, new_cands])
            frontier_ids = new_cands

        # Filter triples to entity_set
        entity_set_cpu = set(entity_set_ids.cpu().tolist())
        mask = torch.isin(all_triples[:, 0], entity_set_ids) & \
               torch.isin(all_triples[:, 2], entity_set_ids)
        filtered = all_triples[mask].unique(dim=0)

        # Convert to MID strings / relation name strings
        ent_mids = [self.id_to_mid(i) or f"unk_{i}" for i in entity_set_ids.cpu().tolist()]
        tuples   = []
        for row in filtered.cpu():
            h_str = self.id_to_mid(row[0].item()) or f"unk_{row[0].item()}"
            r_int = int(row[1].item())
            r_str = self.id2rel[r_int] if 0 <= r_int < len(self.id2rel) else f"rel_{r_int}"
            t_str = self.id_to_mid(row[2].item()) or f"unk_{row[2].item()}"
            tuples.append((h_str, r_str, t_str))
        return ent_mids, tuples


# ─── Cache ────────────────────────────────────────────────────────────────────

def build_neighborhood_cache(seed_mids, kg, max_hops, cache_path,
                             chunk_id=-1, n_chunks=1):
    """Extract neighborhood cache, optionally processing only 1/n_chunks of missing MIDs.
    chunk_id >= 0 writes to a chunk-specific file; chunk_id=-1 writes to cache_path directly."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Determine actual write path
    if chunk_id >= 0:
        write_path = cache_path.replace(".pkl", f"_chunk{chunk_id}.pkl")
    else:
        write_path = cache_path

    # Load existing cache (shared base + any prior partial run)
    cache = {}
    for path in ([cache_path, write_path] if chunk_id >= 0 else [cache_path]):
        if os.path.exists(path):
            print(f"  Loading existing cache: {path}", flush=True)
            try:
                with open(path, "rb") as f:
                    partial = pickle.load(f)
                cache.update(partial)
                print(f"  Loaded {len(partial)} entries from {path}.", flush=True)
            except Exception as e:
                print(f"  WARNING: could not load {path}: {e}", flush=True)

    missing = sorted(m for m in seed_mids if m not in cache)

    # Chunk slicing: take only 1/n_chunks of the remaining work
    if chunk_id >= 0 and n_chunks > 1:
        chunk_size = (len(missing) + n_chunks - 1) // n_chunks
        missing = missing[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]
        print(f"  Chunk {chunk_id}/{n_chunks}: extracting {len(missing)} neighborhoods "
              f"({len(cache)} already cached)...", flush=True)
    else:
        print(f"  Extracting {len(missing)} neighborhoods ({len(cache)} already cached)...",
              flush=True)

    for i, mid in enumerate(missing):
        ent_mids, tuples = kg.extract_neighborhood(mid, max_hops)
        cache[mid] = {"entities": ent_mids, "tuples": tuples}
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(missing)} done ...", flush=True)
            with open(write_path, "wb") as f:
                pickle.dump(cache, f)

    with open(write_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Cache saved ({len(cache)} entries) → {write_path}", flush=True)
    return cache


def merge_chunk_caches(base_cache_path, n_chunks):
    """Merge chunk pkl files into the main cache file."""
    merged = {}
    # Load base cache first (if exists)
    if os.path.exists(base_cache_path):
        with open(base_cache_path, "rb") as f:
            merged = pickle.load(f)
        print(f"  Base cache: {len(merged)} entries.", flush=True)
    for cid in range(n_chunks):
        chunk_path = base_cache_path.replace(".pkl", f"_chunk{cid}.pkl")
        if os.path.exists(chunk_path):
            with open(chunk_path, "rb") as f:
                chunk = pickle.load(f)
            merged.update(chunk)
            print(f"  Chunk {cid}: +{len(chunk)} entries → total {len(merged)}.", flush=True)
    with open(base_cache_path, "wb") as f:
        pickle.dump(merged, f)
    print(f"  Merged cache saved: {len(merged)} entries → {base_cache_path}", flush=True)
    return merged


# ─── ID maps ──────────────────────────────────────────────────────────────────

def build_id_maps(cache, kg):
    """Build string entity/relation maps from all cached neighborhoods."""
    all_ents = set()
    all_rels = set()
    for nbhd in cache.values():
        all_ents.update(nbhd["entities"])
        for h, r, t in nbhd["tuples"]:
            all_ents.add(h)
            all_ents.add(t)
            all_rels.add(r)   # r is already an integer

    id2ent = sorted(all_ents)
    id2rel = sorted(all_rels, key=lambda x: str(x))
    ent2id = {e: i for i, e in enumerate(id2ent)}
    rel2id = {r: i for i, r in enumerate(id2rel)}
    return ent2id, rel2id, id2ent, id2rel


def write_id_files(id2ent, id2rel, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "entities.txt"), "w") as f:
        for e in id2ent:
            f.write(str(e) + "\n")
    with open(os.path.join(out_dir, "relations.txt"), "w") as f:
        for r in id2rel:
            f.write(str(r) + "\n")
    print(f"  entities.txt: {len(id2ent)} | relations.txt: {len(id2rel)}", flush=True)


# ─── Vocab + Embeddings ───────────────────────────────────────────────────────

def tokenize(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def build_vocab(questions, rel_list):
    vocab = OrderedDict()
    for q in questions:
        for tok in tokenize(q):
            if tok and tok not in vocab:
                vocab[tok] = len(vocab)
    for r in rel_list:
        r_str = str(r)
        for part in re.split(r"[._]", r_str):
            part = part.strip().lower()
            if part and part not in vocab:
                vocab[part] = len(vocab)
    return vocab


def load_glove(glove_path, dim=300):
    if not os.path.exists(glove_path):
        print("  WARNING: GloVe not found. Zero vectors will be used.", flush=True)
        return {}
    print(f"  Loading GloVe (one-time) ...", flush=True)
    word2emb = {}
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < dim + 1:
                continue
            try:
                emb = np.array([float(x) for x in parts[-dim:]], dtype=np.float32)
                if len(emb) == dim:
                    word2emb[parts[0]] = emb
            except ValueError:
                continue
    print(f"  GloVe: {len(word2emb)} vectors loaded.", flush=True)
    return word2emb


def build_emb_matrix(vocab, word2emb, dim=300):
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    found = 0
    for word, idx in vocab.items():
        if word in word2emb:
            matrix[idx] = word2emb[word]
            found += 1
    print(f"    word_emb: {found}/{len(vocab)} words in GloVe.", flush=True)
    return matrix


def build_dep_parses(entries, out_path):
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        print("    WARNING: spaCy not available. Skipping test.dep.", flush=True)
        return
    with open(out_path, "w") as f:
        for entry in entries:
            doc = nlp(entry["question"])
            dep = [[t.text.lower(), t.tag_,
                    str(0 if t.head == t else t.head.i + 1),
                    t.dep_] for t in doc]
            f.write(json.dumps({"id": entry["id"], "question": entry["question"], "dep": dep}) + "\n")
    print(f"    test.dep → {out_path}", flush=True)


# ─── Assemble per-strategy ────────────────────────────────────────────────────

def process_strategy(dataset, strategy, cache, ent2id, rel2id, orig_by_id, word2emb, id2rel):
    elq_path = os.path.join(RESULTS_DIR, "output_req", dataset, strategy, "test_elq.jsonl")
    if not os.path.exists(elq_path):
        print(f"  SKIP: {elq_path} not found.", flush=True)
        return 0

    out_dir = os.path.join(OUTPUT_BASE, dataset, strategy)
    os.makedirs(out_dir, exist_ok=True)

    perturbed = load_jsonl(elq_path)
    print(f"  {dataset}/{strategy}: {len(perturbed)} entries", flush=True)

    merged = []
    fallback = 0
    no_cache = 0
    for entry in perturbed:
        qid  = entry["id"]
        orig = orig_by_id.get(qid)

        # Determine seed
        seed = entry.get("elq_seed_mid", "")
        if not is_valid_mid(seed):
            gt = entry.get("gt_seed_mids", [])
            seed = gt[0] if gt and is_valid_mid(gt[0]) else ""
            fallback += 1

        nbhd = cache.get(seed, {"entities": [seed] if seed else [], "tuples": []})
        if seed and seed not in cache:
            no_cache += 1

        # Convert to local integer IDs
        seed_ids = [ent2id[seed]] if seed and seed in ent2id else []
        tuples_int = []
        for h, r, t in nbhd["tuples"]:
            if h in ent2id and r in rel2id and t in ent2id:
                tuples_int.append([ent2id[h], rel2id[r], ent2id[t]])
        ent_ids = [ent2id[e] for e in nbhd["entities"] if e in ent2id]

        out = {
            "id": qid,
            "question": entry["question"],
            "question_orig": entry.get("question_orig", orig["question"] if orig else ""),
            "answers": orig["answers"] if orig else [],
            "entities": seed_ids,
            "subgraph": {"tuples": tuples_int, "entities": ent_ids},
            "elq_seed_mid": entry.get("elq_seed_mid", ""),
            "gt_seed_mids": entry.get("gt_seed_mids", []),
            "elq_gt_mid_match": entry.get("elq_gt_mid_match", None),
            "strategy": entry.get("strategy", strategy),
        }
        merged.append(out)

    subgraph_path = os.path.join(out_dir, "test_subgraph.json")
    with open(subgraph_path, "w") as f:
        for e in merged:
            f.write(json.dumps(e) + "\n")
    print(f"    test_subgraph.json: {len(merged)} → {subgraph_path}", flush=True)
    print(f"    (gt fallback: {fallback}, no-cache: {no_cache})", flush=True)

    # Vocab + embeddings
    questions = [e["question"] for e in merged]
    vocab = build_vocab(questions, id2rel)
    vocab_path = os.path.join(out_dir, "vocab.txt")
    with open(vocab_path, "w") as f:
        for word in vocab:
            f.write(word + "\n")
    print(f"    vocab.txt: {len(vocab)} words", flush=True)

    emb = build_emb_matrix(vocab, word2emb, GLOVE_DIM)
    np.save(os.path.join(out_dir, "word_emb.npy"), emb)
    print(f"    word_emb.npy: {emb.shape}", flush=True)

    build_dep_parses(merged, os.path.join(out_dir, "test.dep"))
    return len(merged)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cwq", "webqsp", "both"], default="both")
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--skip_kg_load", action="store_true",
                        help="Use existing mid_cache only, skip GPU KG load")
    # Parallel chunk extraction
    parser.add_argument("--chunk_id", type=int, default=-1,
                        help="Chunk index (0-based). -1 = no chunking.")
    parser.add_argument("--n_chunks", type=int, default=1,
                        help="Total number of parallel chunks.")
    parser.add_argument("--skip_assembly", action="store_true",
                        help="Only extract neighborhoods; skip entities.txt and test_subgraph.json.")
    parser.add_argument("--merge_chunks", action="store_true",
                        help="Merge chunk caches then run assembly. No KG load needed.")
    args = parser.parse_args()

    global DEVICE
    DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    datasets   = ["cwq", "webqsp"] if args.dataset == "both" else [args.dataset]
    strategies = STRATEGIES if args.strategy == "all" else [args.strategy]

    # --merge_chunks: merge chunk pkl files then run assembly; no KG needed
    if args.merge_chunks:
        print("Merge-chunks mode.", flush=True)
        word2emb = load_glove(GLOVE_PATH, GLOVE_DIM)
        for dataset in datasets:
            cache_path  = os.path.join(MID_CACHE, f"neighborhoods_{dataset}.pkl")
            dataset_out = os.path.join(OUTPUT_BASE, dataset)
            orig_by_id  = load_orig_by_id(DATASETS[dataset]["orig_json"])
            cache = merge_chunk_caches(cache_path, args.n_chunks)
            ent2id, rel2id, id2ent, id2rel = build_id_maps(cache, None)
            write_id_files(id2ent, id2rel, dataset_out)
            total = 0
            for strategy in strategies:
                print(f"\n  Strategy: {strategy}", flush=True)
                total += process_strategy(dataset, strategy, cache, ent2id, rel2id,
                                          orig_by_id, word2emb, id2rel)
            print(f"\n  {dataset.upper()} done. Total: {total}", flush=True)
        print(f"\nAll done. Outputs in: {OUTPUT_BASE}", flush=True)
        return

    print(f"Using device: {DEVICE}", flush=True)

    # Load KG onto GPU (one-time, shared)
    if not args.skip_kg_load:
        kg = FreebaseKG(KG_NPY, ENT2ID_PKL, RELATIONS_TXT, device=DEVICE)
    else:
        kg = None
        print("Skipping KG load (--skip_kg_load). Cache-only mode.", flush=True)

    # Load GloVe (only needed for assembly, not chunk extraction)
    word2emb = {} if args.skip_assembly else load_glove(GLOVE_PATH, GLOVE_DIM)

    for dataset in datasets:
        print(f"\n{'='*60}", flush=True)
        print(f"Dataset: {dataset.upper()}  ({DATASETS[dataset]['hops']}-hop)", flush=True)
        print(f"{'='*60}", flush=True)

        hops        = DATASETS[dataset]["hops"]
        cache_path  = os.path.join(MID_CACHE, f"neighborhoods_{dataset}.pkl")
        dataset_out = os.path.join(OUTPUT_BASE, dataset)
        orig_by_id  = load_orig_by_id(DATASETS[dataset]["orig_json"])
        print(f"  Loaded {len(orig_by_id)} original entries.", flush=True)

        # Collect all unique seed MIDs
        seed_mids = load_seed_mids_from_csv(ENTITY_LIST[dataset])
        for strat in strategies:
            p = os.path.join(RESULTS_DIR, "output_req", dataset, strat, "test_elq.jsonl")
            if os.path.exists(p):
                for e in load_jsonl(p):
                    for mid in e.get("gt_seed_mids", []):
                        if is_valid_mid(mid):
                            seed_mids.add(mid)
        print(f"  Unique seed MIDs to extract: {len(seed_mids)}", flush=True)

        # Extract/load neighborhood cache
        if kg is not None:
            cache = build_neighborhood_cache(seed_mids, kg, hops, cache_path,
                                             chunk_id=args.chunk_id,
                                             n_chunks=args.n_chunks)
        else:
            if not os.path.exists(cache_path):
                print("ERROR: No KG and no cache. Run without --skip_kg_load.", flush=True)
                sys.exit(1)
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"  Cache loaded: {len(cache)} entries.", flush=True)

        if args.skip_assembly:
            print("  --skip_assembly: skipping entities.txt and test_subgraph builds.", flush=True)
            continue

        # Build fresh entity/relation ID maps (per dataset, shared across strategies)
        ent2id, rel2id, id2ent, id2rel = build_id_maps(cache, kg)
        write_id_files(id2ent, id2rel, dataset_out)
        print(f"  ID maps: {len(ent2id)} ents, {len(rel2id)} rels", flush=True)

        # Process each strategy
        total = 0
        for strategy in strategies:
            print(f"\n  Strategy: {strategy}", flush=True)
            total += process_strategy(dataset, strategy, cache, ent2id, rel2id,
                                      orig_by_id, word2emb, id2rel)

        print(f"\n  {dataset.upper()} done. Total: {total}", flush=True)

    print(f"\nAll done. Outputs in: {OUTPUT_BASE}", flush=True)


if __name__ == "__main__":
    main()
