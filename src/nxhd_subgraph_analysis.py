#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
nxhd_subgraph_analysis.py — Subgraph analysis for 10 correct-seed samples per p-type.

For each question with correct ELQ seed (any-GT match):
  Build perturbed subgraph:
    CWQ:    UNION of
              new_seed_nxhd/{pred_mid}.nxhd    (seed-keyed 2-hop)
            + cwq_4hop_nxhd/{question_id}.nxhd (question-keyed 4-hop)
            capped at MAX_ENTITIES_CWQ = 2500 unique MIDs
    WebQSP: stagg.neighborhoods/WebQTest-{N}.nxhd  (question-keyed)
            capped at MAX_ENTITIES_WQ = 2000 unique MIDs

  Build orig subgraph (same question, GT seed):
    CWQ:    new_seed_nxhd/{gt_mid}.nxhd + cwq_4hop_nxhd/{question_id}.nxhd
    WebQSP: same file as pert (same question ID, so identical subgraph)

Metrics per sample:
  - answer_coverage   : answer MID reachable in pert subgraph entities
  - jaccard           : |pert_ents ∩ orig_ents| / |pert_ents ∪ orig_ents|
  - delta_SR          : 1 - jaccard (Structural Ratio difference)
  - GED               : |pert_triples △ orig_triples| (symmetric difference of edge sets)
  - golden_path_pert  : BFS 1-4 hops from pred_mid → answer_mid in pert subgraph
  - golden_path_orig  : BFS 1-4 hops from gt_mid   → answer_mid in orig subgraph
  - golden_retained   : golden_path_orig AND golden_path_pert (path retained after perturbation)
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

REPO_ROOT = REPO_ROOT
DATASET     = REPO_ROOT / "pert" / "dataset"
RESULTS     = REPO_ROOT / "results"
GNN_DATA    = REPO_ROOT / "external" / "GNN-RAG" / "data"
CWQ_NXHD    = REPO_ROOT / "data" / "cwq_4hop_nxhd"
NEW_SEED    = REPO_ROOT / "data" / "new_seed_nxhd"
WEBQSP_NXHD = REPO_ROOT / "data" / "freebase_2hops" / "stagg.neighborhoods"

MAX_ENTITIES_CWQ = 2500
MAX_ENTITIES_WQ  = 2000

RESULTS.mkdir(exist_ok=True)

_FB = re.compile(r"^<fb:(.+)>$")


def strip_fb(s: str) -> Optional[str]:
    m = _FB.match(s.strip())
    return m.group(1) if m else None


# ── nxhd parsing ─────────────────────────────────────────────────────────────

def parse_nxhd(path: Path, entity_cap: int, existing_ents: Optional[Set[str]] = None
               ) -> Tuple[Set[str], Set[Tuple[str,str,str]]]:
    """
    Parse one .nxhd file.  Returns (entities, triples).
    entity_cap: stop adding NEW entities once this many unique MIDs are seen.
    existing_ents: already-seen entities from previously parsed files (for union capping).
    Format: <fb:subj>\t<fb:rel>\t<fb:obj_or_literal>
    Only MID subjects/objects (m.xxx or g.xxx) are counted toward entity_cap.
    """
    entities: Set[str] = set()
    triples:  Set[Tuple[str,str,str]] = set()

    if existing_ents is None:
        existing_ents = set()
    combined_ents = existing_ents | entities   # live view

    if not path.exists():
        return entities, triples

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            subj = strip_fb(parts[0])
            rel  = strip_fb(parts[1])
            obj_raw = parts[2].strip()
            obj  = strip_fb(obj_raw)

            if not subj or not rel:
                continue

            # Subject must be an entity MID
            if not (subj.startswith("m.") or subj.startswith("g.")):
                continue

            # Enforce entity cap
            total_ents = len(combined_ents) + len(entities)
            if subj not in combined_ents and subj not in entities:
                if total_ents >= entity_cap:
                    continue   # cap reached, skip triples adding new entities
                entities.add(subj)

            # Object: may be entity MID or literal string
            obj_str = obj if obj else obj_raw.strip('"').strip()
            if obj and (obj.startswith("m.") or obj.startswith("g.")):
                if obj not in combined_ents and obj not in entities:
                    total_ents = len(combined_ents) + len(entities)
                    if total_ents < entity_cap:
                        entities.add(obj)

            triples.add((subj, rel, obj_str))

    return entities, triples


def build_subgraph(
    qid:        str,
    seed_mid:   str,
    dataset:    str,
    entity_cap: int,
) -> Tuple[Set[str], Set[Tuple[str,str,str]]]:
    """
    Build UNION subgraph from available nxhd sources for one (question, seed).
    Returns (entities, triples) capped at entity_cap entities.
    """
    all_ents:    Set[str]                    = set()
    all_triples: Set[Tuple[str,str,str]]     = set()

    if dataset == "cwq":
        # Source 1: new_seed_nxhd/{seed_mid}.nxhd  (seed-keyed 2-hop)
        seed_file = NEW_SEED / f"{seed_mid}.nxhd"
        e1, t1 = parse_nxhd(seed_file, entity_cap, existing_ents=all_ents)
        all_ents.update(e1);  all_triples.update(t1)

        # Source 2: cwq_4hop_nxhd/{qid}.nxhd  (question-keyed 4-hop)
        q_file = CWQ_NXHD / f"{qid}.nxhd"
        e2, t2 = parse_nxhd(q_file, entity_cap, existing_ents=all_ents)
        all_ents.update(e2);  all_triples.update(t2)

    else:  # webqsp
        # stagg.neighborhoods/WebQTest-{N}.nxhd
        base = qid.split("_")[0]   # "WebQTest-832"
        q_file = WEBQSP_NXHD / f"{base}.nxhd"
        e1, t1 = parse_nxhd(q_file, entity_cap, existing_ents=all_ents)
        all_ents.update(e1);  all_triples.update(t1)

    return all_ents, all_triples


# ── BFS: can we reach answer_mid from seed_mid in ≤ max_hops? ────────────────

def build_adj(triples: Set[Tuple[str,str,str]]) -> Dict[str, List[str]]:
    """Build adjacency list: subj → [obj, …]"""
    adj: Dict[str, List[str]] = defaultdict(list)
    for subj, _, obj in triples:
        adj[subj].append(obj)
    return adj


def bfs_path_exists(adj: Dict[str, List[str]], start: str, target: str,
                    max_hops: int = 4) -> bool:
    """BFS from start → target with max_hops depth."""
    if start == target:
        return True
    if start not in adj:
        return False
    visited = {start}
    queue   = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for nb in adj.get(node, []):
            if nb == target:
                return True
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, depth + 1))
    return False


# ── Subgraph comparison metrics ───────────────────────────────────────────────

def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def ged(a: Set, b: Set) -> int:
    """Graph edit distance approximation: |symmetric difference| of edge sets."""
    return len(a.symmetric_difference(b))


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse_ptype(
    dataset:   str,
    p_type:    str,
    n_sample:  int,
    rng:       random.Random,
    id2mid:    Dict[int, str],
    gt_map:    Dict[str, Dict],   # qid → {"gt_mids": set, "answers": list}
    el_by_id:  Dict[str, Dict],
) -> Dict:
    """
    Select n_sample correct-seed questions for this (dataset, p_type),
    build subgraphs, compute all metrics.
    """
    ds_path = DATASET / f"{dataset}_{p_type}.json"
    if not ds_path.exists():
        return {}

    records = json.load(open(ds_path))

    # Filter to correct-seed (any ELQ predicted MID ∈ GT)
    correct = []
    for rec in records:
        qid = rec["id"]
        gt  = gt_map.get(qid, {}).get("gt_mids", set())
        el  = el_by_id.get(qid, {})
        pred_mids = {e.get("mid","") for e in el.get("linked_entities",[]) if e.get("mid")}
        if pred_mids & gt:
            correct.append(rec)

    if not correct:
        print(f"    [SKIP] no correct-seed samples for {dataset} {p_type}")
        return {}

    sample = rng.sample(correct, min(n_sample, len(correct)))
    entity_cap = MAX_ENTITIES_CWQ if dataset == "cwq" else MAX_ENTITIES_WQ

    per_q = []
    for rec in sample:
        qid     = rec["id"]
        gt_info = gt_map.get(qid, {})
        gt_mids = gt_info.get("gt_mids", set())
        answers = gt_info.get("answers", [])

        el      = el_by_id.get(qid, {})
        linked  = el.get("linked_entities", [])
        pred_mids = [e.get("mid","") for e in linked if e.get("mid") and e.get("mid") in gt_mids]
        if not pred_mids:
            # fallback: any linked MID
            pred_mids = [e.get("mid","") for e in linked if e.get("mid")]
        pred_mid = pred_mids[0] if pred_mids else ""

        gt_mid   = next(iter(gt_mids), "")  # representative GT seed
        ans_mids = [a.get("kb_id","") for a in answers if a.get("kb_id")]

        # Build pert subgraph (predicted seed)
        pert_ents, pert_trips = build_subgraph(qid, pred_mid, dataset, entity_cap)

        # Build orig subgraph (GT seed, same question ID)
        orig_ents, orig_trips = build_subgraph(qid, gt_mid, dataset, entity_cap)

        # Answer coverage in pert subgraph
        ans_in_pert = any(a in pert_ents for a in ans_mids if a)

        # Subgraph similarity
        ent_jacc  = jaccard(pert_ents, orig_ents)
        trip_jacc = jaccard(pert_trips, orig_trips)
        delta_sr  = 1.0 - trip_jacc
        g_ged     = ged(pert_trips, orig_trips)

        # Golden path in orig (gt_mid → answer, 1-4 hops)
        orig_adj = build_adj(orig_trips)
        orig_golden = any(bfs_path_exists(orig_adj, gt_mid, a, max_hops=4)
                          for a in ans_mids if a) if gt_mid else False

        # Golden path in pert (pred_mid → answer, 1-4 hops)
        pert_adj  = build_adj(pert_trips)
        pert_golden = any(bfs_path_exists(pert_adj, pred_mid, a, max_hops=4)
                          for a in ans_mids if a) if pred_mid else False

        # Retention: orig had a path AND pert still has a path
        retained = orig_golden and pert_golden

        per_q.append({
            "qid":              qid,
            "pred_mid":         pred_mid,
            "gt_mid":           gt_mid,
            "pert_n_ents":      len(pert_ents),
            "orig_n_ents":      len(orig_ents),
            "pert_n_trips":     len(pert_trips),
            "orig_n_trips":     len(orig_trips),
            "ent_jaccard":      round(ent_jacc,  4),
            "trip_jaccard":     round(trip_jacc, 4),
            "delta_SR":         round(delta_sr,  4),
            "GED":              g_ged,
            "ans_in_pert":      ans_in_pert,
            "orig_golden_path": orig_golden,
            "pert_golden_path": pert_golden,
            "golden_retained":  retained,
        })

    # Aggregate
    def avg(lst):  return round(sum(lst)/len(lst), 4) if lst else 0.0
    def pct(lst):  return round(sum(1 for v in lst if v)/len(lst)*100, 2) if lst else 0.0

    return {
        "dataset": dataset, "p_type": p_type,
        "n_correct_pool": len(correct),
        "n_sample":  len(per_q),
        "avg_pert_ents":  avg([q["pert_n_ents"]  for q in per_q]),
        "avg_orig_ents":  avg([q["orig_n_ents"]  for q in per_q]),
        "avg_pert_trips": avg([q["pert_n_trips"] for q in per_q]),
        "avg_orig_trips": avg([q["orig_n_trips"] for q in per_q]),
        "ent_jaccard_mean":  avg([q["ent_jaccard"]  for q in per_q]),
        "trip_jaccard_mean": avg([q["trip_jaccard"] for q in per_q]),
        "delta_SR_mean":     avg([q["delta_SR"]     for q in per_q]),
        "GED_mean":          avg([q["GED"]          for q in per_q]),
        "answer_coverage_pct":     pct([q["ans_in_pert"]      for q in per_q]),
        "orig_golden_path_pct":    pct([q["orig_golden_path"] for q in per_q]),
        "pert_golden_path_pct":    pct([q["pert_golden_path"] for q in per_q]),
        "golden_retained_pct":     pct([q["golden_retained"]  for q in per_q]),
        "samples": per_q,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-sample", type=int, default=10)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--dataset",  choices=["cwq","webqsp"])
    p.add_argument("--p-type",   choices=["orig","p1","p2","p3","p4","p5","p6"])
    args = p.parse_args()

    rng = random.Random(args.seed)

    all_results: Dict = {}

    for dataset in (["cwq","webqsp"] if not args.dataset else [args.dataset]):
        print(f"\n{'='*60}\n  Dataset: {dataset}\n{'='*60}", flush=True)
        id2mid = {}
        subdir = "CWQ" if dataset == "cwq" else "webqsp"
        with open(GNN_DATA / subdir / "entities.txt") as f:
            for i, line in enumerate(f):
                if line.strip():
                    id2mid[i] = line.strip()

        # Build GT map from orig
        orig_records = json.load(open(DATASET / f"{dataset}_orig.json"))
        gt_map = {
            r["id"]: {
                "gt_mids": {id2mid[i] for i in r.get("gnn_entities",[]) if i in id2mid},
                "answers": r.get("answers", []),
            }
            for r in orig_records
        }

        ptypes = [args.p_type] if args.p_type else ["p1","p2","p3","p4","p5","p6"]
        for p_type in ptypes:
            print(f"\n  ── {p_type} ──", flush=True)

            # Load ELQ
            el_path = DATASET / "el_elq" / f"{dataset}_{p_type}_el.json"
            if not el_path.exists():
                print(f"    [SKIP] ELQ missing")
                continue
            el_by_id = {r["id"]: r for r in json.load(open(el_path))}

            result = analyse_ptype(dataset, p_type, args.n_sample, rng,
                                   id2mid, gt_map, el_by_id)
            if not result:
                continue

            key = f"{dataset}_{p_type}"
            all_results[key] = result

            # Print summary
            r = result
            print(f"    n={r['n_sample']} (pool={r['n_correct_pool']})")
            print(f"    subgraph  pert_ents={r['avg_pert_ents']:.0f}  "
                  f"orig_ents={r['avg_orig_ents']:.0f}  "
                  f"pert_trips={r['avg_pert_trips']:.0f}  "
                  f"orig_trips={r['avg_orig_trips']:.0f}")
            print(f"    entity_Jaccard={r['ent_jaccard_mean']:.3f}  "
                  f"trip_Jaccard={r['trip_jaccard_mean']:.3f}  "
                  f"ΔSR={r['delta_SR_mean']:.3f}  GED={r['GED_mean']:.0f}")
            print(f"    answer_coverage={r['answer_coverage_pct']}%  "
                  f"orig_golden={r['orig_golden_path_pct']}%  "
                  f"pert_golden={r['pert_golden_path_pct']}%  "
                  f"retained={r['golden_retained_pct']}%")

            # Per-sample detail
            print(f"    {'qid':<48} {'pred':>10} {'ans':>4} {'orig_gp':>8} {'pert_gp':>8} {'ret':>4}")
            for q in r["samples"]:
                print(f"    {q['qid'][:47]:<48} {q['pred_mid']:>10}  "
                      f"{'Y' if q['ans_in_pert'] else 'N':>3}  "
                      f"{'Y' if q['orig_golden_path'] else 'N':>7}  "
                      f"{'Y' if q['pert_golden_path'] else 'N':>7}  "
                      f"{'Y' if q['golden_retained'] else 'N':>3}")

    # Save
    out = RESULTS / "nxhd_subgraph_analysis.json"
    # Strip per-sample list for top-level file to keep it readable;
    # save full version with samples
    json.dump(all_results, open(out, "w"), indent=2)
    print(f"\nFull results saved → {out}", flush=True)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY: nxhd subgraph analysis ({args.n_sample} correct-seed samples/ptype)")
    print(f"{'='*80}")
    print(f"  {'DS':<8} {'PT':<5} {'n':>4}  {'AnsC%':>6}  "
          f"{'EntJac':>7}  {'TripJac':>8}  {'ΔSR':>5}  "
          f"{'OGP%':>5}  {'PGP%':>5}  {'Ret%':>5}")
    print("  " + "-"*75)
    for key in sorted(all_results):
        r = all_results[key]
        print(f"  {r['dataset']:<8} {r['p_type']:<5} {r['n_sample']:>4}  "
              f"{r['answer_coverage_pct']:>5.1f}%  "
              f"{r['ent_jaccard_mean']:>7.3f}  "
              f"{r['trip_jaccard_mean']:>8.3f}  "
              f"{r['delta_SR_mean']:>5.3f}  "
              f"{r['orig_golden_path_pct']:>5.1f}  "
              f"{r['pert_golden_path_pct']:>5.1f}  "
              f"{r['golden_retained_pct']:>5.1f}")


if __name__ == "__main__":
    main()