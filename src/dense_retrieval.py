#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
dense_retrieval.py — 3-tier subgraph neighborhood lookup for perturbed questions.

Three EL-source modes:
  elq     — use data/perturbations/el_elq/{dataset}_{p_type}_el.json
  agent   — use data/perturbations/el_agent/{dataset}_{p_type}_el.json
  hybrid  — UNION of ELQ + ELA linked MIDs per question (best recall)

For each question, given linked entity MIDs, retrieve subgraph via 3-tier hierarchy:

  Tier 1 (Primary): Pre-process/mid_cache/neighborhoods_{dataset}.pkl
    Dict[MID → subgraph dict], loaded once. Fastest — full 4-hop PKL cache.

  Tier 2 (Secondary):
    CWQ:    Pre-process/cwq_4hop_nxhd/{question_id}*.nxhd   (per-question 4-hop files)
    WebQSP: freebase/data/dataset/freebase_2hops/stagg.neighborhoods/{id}.nxhd

  Tier 3 (Tertiary): data/resources/manual_fb_filter.txt
    Full filtered Freebase text scan — last resort, opt-in via --tier3.

Output: data/perturbations/subgraphs/{dataset}_{p_type}_{el_source}_subgraph.pkl
  Dict[question_id → {mids, subgraph|triples, tier_used, el_source}]

Usage:
  python dense_retrieval.py --dataset cwq --p-type p1 --el-source elq
  python dense_retrieval.py --dataset cwq --p-type p1 --el-source agent
  python dense_retrieval.py --dataset cwq --p-type p1 --el-source hybrid
  python dense_retrieval.py --all --el-source hybrid
  python dense_retrieval.py --dataset cwq --p-type p1 --el-source elq --dry-run
"""

from __future__ import annotations

import argparse
import glob
import gzip
import pickle
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DATASET_DIR    = REPO_ROOT / "data" / "perturbations"
MID_CACHE_DIR  = REPO_ROOT / "data" / "mid_cache"
NXHD_CWQ_DIR   = REPO_ROOT / "data" / "cwq_4hop_nxhd"
NXHD_WEBQSP_DIR = Path(
    str(REPO_ROOT / "data" / "freebase_2hops" / "stagg.neighborhoods")
)
# Tier 1.5: per-MID nxhd files for new seeds (ELQ + ELA) not in original PKL
NEW_SEED_NXHD_DIR = REPO_ROOT / "data" / "new_seed_nxhd"
FB_FILTER_PATH = REPO_ROOT / "data" / "resources" / "manual_fb_filter.txt"
OUT_DIR = DATASET_DIR / "subgraphs"

PERT_TYPES = ["p1", "p2", "p3", "p4", "p5", "p6"]


# ─── Tier 1: PKL neighborhood cache ──────────────────────────────────────────

_pkl_cache: Dict[str, Any] = {}  # dataset → loaded dict


def _load_pkl(dataset: str) -> Dict[str, Any]:
    if dataset not in _pkl_cache:
        path = MID_CACHE_DIR / f"neighborhoods_{dataset}.pkl"
        print(f"  Loading {path} ...")
        with open(path, "rb") as f:
            _pkl_cache[dataset] = pickle.load(f)
        print(f"  Loaded {len(_pkl_cache[dataset]):,} MID entries.")
    return _pkl_cache[dataset]


def lookup_tier1(dataset: str, mid: str) -> Optional[Dict]:
    nb = _load_pkl(dataset)
    return nb.get(mid)


# ─── Tier 1.5: Per-MID nxhd files (new seeds not in PKL) ────────────────────
# These files are keyed by Freebase MID (e.g. m.02mjmr.nxhd) and contain
# the Freebase neighborhood extracted from manual_fb_filter.txt.
# This is the correct fallback for ELA/ELQ seeds not in the original PKL.

def lookup_tier1_5(mids: List[str]) -> List[Tuple[str, str, str]]:
    """Look up per-MID nxhd files in new_seed_nxhd/ for MIDs not in PKL."""
    triples: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for mid in mids:
        p = NEW_SEED_NXHD_DIR / f"{mid}.nxhd"
        if not p.exists():
            continue
        for t in _parse_nxhd(p):
            if t not in seen:
                seen.add(t)
                triples.append(t)
    return triples


# ─── Tier 2: Per-question .nxhd files (original question neighborhood) ───────
# NOTE: These are indexed by question-ID and contain the ORIGINAL question's
# neighborhood — only used for ELQ el-source where seeds match the original.
# Do NOT use for ELA/hybrid where seeds differ from the original question.

def _nxhd_path_cwq(question_id: str) -> Optional[Path]:
    """Find the .nxhd file for a CWQ question ID (prefix match)."""
    pattern = str(NXHD_CWQ_DIR / f"{question_id}*.nxhd")
    matches = glob.glob(pattern)
    return Path(matches[0]) if matches else None


def _nxhd_path_webqsp(question_id: str) -> Optional[Path]:
    """WebQSP nxhd: {question_id}.nxhd (e.g., WebQTest-0.nxhd)"""
    p = NXHD_WEBQSP_DIR / f"{question_id}.nxhd"
    return p if p.exists() else None


def _parse_nxhd(path: Path) -> List[Tuple[str, str, str]]:
    """Parse a .nxhd file → list of (subject_mid, predicate, object_mid_or_value)."""
    triples: List[Tuple[str, str, str]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    triples.append((parts[0], parts[1], parts[2]))
    except Exception:
        pass
    return triples


def lookup_tier2(dataset: str, question_id: str) -> List[Tuple[str, str, str]]:
    """Original question-ID nxhd — only valid when seeds == original seeds (ELQ)."""
    if dataset == "cwq":
        path = _nxhd_path_cwq(question_id)
    else:
        base_id = question_id.split("_")[0] if "_" in question_id else question_id
        path = _nxhd_path_webqsp(base_id)
    if path is None:
        return []
    return _parse_nxhd(path)


# ─── Tier 3: manual_fb_filter.txt line scan ──────────────────────────────────

def lookup_tier3(mids: Set[str], max_lines: int = 50_000_000) -> List[Tuple[str, str, str]]:
    """Scan manual_fb_filter.txt for lines containing any of the given MIDs."""
    triples: List[Tuple[str, str, str]] = []
    mid_set = {f"ns:{m}" for m in mids} | {f"<http://rdf.freebase.com/ns/{m}>" for m in mids}
    try:
        open_fn = gzip.open if str(FB_FILTER_PATH).endswith(".gz") else open
        with open_fn(FB_FILTER_PATH, "rt", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                if any(m in line for m in mid_set):
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        triples.append((parts[0], parts[1], parts[2]))
    except Exception:
        pass
    return triples


# ─── Main retrieval logic ────────────────────────────────────────────────────

def retrieve_subgraph(
    dataset: str,
    question_id: str,
    mids: List[str],
    use_tier3: bool = False,
    el_source: str = "elq",
) -> Dict[str, Any]:
    """
    Return subgraph dict for a question given linked entity MIDs.

    Tier hierarchy (MID-based — correct for ELA/hybrid/ELQ from-scratch):
      Tier 1  : PKL (neighborhoods_{dataset}.pkl keyed by MID)
      Tier 1.5: new_seed_nxhd/{mid}.nxhd  — pre-built for new seeds
      Tier 2  : question-ID nxhd (ORIGINAL neighborhood) — ELQ only
      Tier 3  : manual_fb_filter.txt scan (opt-in, slow)
    """
    # Tier 1: PKL lookup per MID
    tier1_triples: List[Tuple[str, str, str]] = []
    tier1_entities: List[str] = []
    seen_triples: Set[Tuple[str, str, str]] = set()
    pklhit_mids: List[str] = []
    for mid in mids:
        nb = lookup_tier1(dataset, mid)
        if nb:
            pklhit_mids.append(mid)
            for t in nb.get("tuples", []):
                key = (t[0], t[1], t[2])
                if key not in seen_triples:
                    seen_triples.add(key)
                    tier1_triples.append(key)
            for e in nb.get("entities", []):
                tier1_entities.append(e)

    if tier1_triples:
        return {
            "mids": mids, "triples": tier1_triples,
            "entities": list(dict.fromkeys(tier1_entities)),
            "tier_used": 1,
        }

    # Tier 1.5: per-MID nxhd (new seeds not in PKL) — correct for any el_source
    t15_triples = lookup_tier1_5(mids)
    if t15_triples:
        return {"mids": mids, "triples": t15_triples, "tier_used": 1.5}

    # Tier 2: question-ID nxhd — original neighborhood, valid only for ELQ
    # (where perturbed seeds should match original seeds for most questions)
    if el_source == "elq":
        triples = lookup_tier2(dataset, question_id)
        if triples:
            return {"mids": mids, "triples": triples, "tier_used": 2}

    # Tier 3: expensive full scan (opt-in)
    if use_tier3 and mids:
        triples = lookup_tier3(set(mids))
        if triples:
            return {"mids": mids, "triples": triples, "tier_used": 3}

    return {"mids": mids, "triples": [], "tier_used": 0}


def load_el_records(dataset: str, p_type: str, el_source: str) -> Optional[List[Dict]]:
    """Load EL records for a single source; return None if file missing.

    Special el_source values:
      agent_v1 → data/perturbations/el_agent/ELA_v1/{dataset}_{p_type}_el.json
      hybrid_v1 → union of elq + agent_v1
    """
    import json
    if el_source == "agent_v1":
        path = DATASET_DIR / "el_agent" / "ELA_v1" / f"{dataset}_{p_type}_el.json"
    else:
        path = DATASET_DIR / f"el_{el_source}" / f"{dataset}_{p_type}_el.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def merge_hybrid_records(
    elq_records: List[Dict], agent_records: List[Dict]
) -> List[Dict]:
    """
    Merge ELQ + ELA records per question.
    For each question ID: union of linked_entities from both sources.
    MID deduplication by mid value; ELQ entities kept first (higher precision),
    ELA appended (adds recall for aliases and paraphrased mentions).
    """
    agent_by_id = {r["id"]: r for r in agent_records}
    merged = []
    for rec in elq_records:
        qid = rec["id"]
        elq_ents = rec.get("linked_entities", [])
        agent_ents = agent_by_id.get(qid, {}).get("linked_entities", [])

        # Union by MID (ELQ first for precision, ELA adds recall)
        seen_mids: Set[str] = set()
        combined: List[Dict] = []
        for ent in elq_ents:
            mid = ent.get("mid", "")
            if mid and mid not in seen_mids:
                combined.append({**ent, "el_source": "elq"})
                seen_mids.add(mid)
        for ent in agent_ents:
            mid = ent.get("mid", "")
            if mid and mid not in seen_mids:
                combined.append({**ent, "el_source": "agent"})
                seen_mids.add(mid)

        merged.append({**rec, "linked_entities": combined, "el_source": "hybrid"})
    return merged


def process_dataset(
    dataset: str,
    p_type: str,
    el_source: str,
    dry_run: bool,
    use_tier3: bool,
) -> None:
    import json

    out_path = OUT_DIR / f"{dataset}_{p_type}_{el_source}_subgraph.pkl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load records based on el_source ──────────────────────────────────────
    if el_source in ("hybrid", "hybrid_v1"):
        agent_src = "agent_v1" if el_source == "hybrid_v1" else "agent"
        elq_recs   = load_el_records(dataset, p_type, "elq")
        agent_recs = load_el_records(dataset, p_type, agent_src)
        if elq_recs is None and agent_recs is None:
            print(f"  [SKIP] Neither elq nor {agent_src} files found for {dataset} {p_type}")
            return
        if elq_recs is None:
            print(f"  [WARN] ELQ file missing — {el_source} falls back to {agent_src} only")
            records = agent_recs
        elif agent_recs is None:
            print(f"  [WARN] {agent_src} file missing — {el_source} falls back to ELQ only")
            records = elq_recs
        else:
            records = merge_hybrid_records(elq_recs, agent_recs)
            elq_only   = sum(1 for r in records for e in r["linked_entities"] if e.get("el_source") == "elq")
            agent_only = sum(1 for r in records for e in r["linked_entities"] if e.get("el_source") == "agent")
            print(f"  {el_source}: {len(records)} questions, ELQ={elq_only}, ELA={agent_only}")
    else:
        records = load_el_records(dataset, p_type, el_source)
        if records is None:
            print(f"  [SKIP] {DATASET_DIR}/el_{el_source}/{dataset}_{p_type}_el.json "
                  f"not found — run el_{el_source}.py first")
            return

    if dry_run:
        records = records[:5]

    print(f"  Processing {len(records)} records ...")

    results: Dict[str, Any] = {}
    tier_counts: Dict[Any, int] = {0: 0, 1: 0, 1.5: 0, 2: 0, 3: 0}

    for i, rec in enumerate(records):
        qid = rec["id"]
        mids = [e["mid"] for e in rec.get("linked_entities", []) if e.get("mid")]
        sg = retrieve_subgraph(dataset, qid, mids, use_tier3=use_tier3, el_source=el_source)
        sg["el_source"] = el_source
        results[qid] = sg
        t = sg["tier_used"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(records)}  tiers={tier_counts}")

    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=4)  # protocol 4 avoids LONG_BINGET overflow

    print(f"  Saved → {out_path}")
    print(f"  Tier coverage: T1={tier_counts[1]} T1.5={tier_counts[1.5]} "
          f"T2={tier_counts[2]} T3={tier_counts[3]} T0(miss)={tier_counts[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="3-tier subgraph neighborhood retrieval")
    parser.add_argument("--dataset", choices=["cwq", "webqsp"], default=None)
    parser.add_argument("--p-type", choices=PERT_TYPES, default=None)
    parser.add_argument("--el-source", choices=["elq", "agent", "hybrid", "agent_v1", "hybrid_v1"], required=True,
                        help="elq=ELQ only, agent=ELA only, hybrid=union of both (best recall)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tier3", action="store_true",
                        help="Enable tier-3 (manual_fb_filter.txt) fallback (slow)")
    args = parser.parse_args()

    if args.all:
        datasets = ["cwq", "webqsp"]
        p_types = PERT_TYPES
    else:
        if not args.dataset or not args.p_type:
            parser.error("Specify --dataset and --p-type, or use --all")
        datasets = [args.dataset]
        p_types = [args.p_type]

    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        for p_type in p_types:
            print(f"\n--- {dataset} {p_type} ({args.el_source}) ---")
            process_dataset(dataset, p_type, args.el_source,
                            dry_run=args.dry_run, use_tier3=args.tier3)


if __name__ == "__main__":
    main()