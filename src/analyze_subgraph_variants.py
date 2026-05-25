#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
analyze_subgraph_variants.py — Correct subgraph overlap for all custom variants.

Compares NSM, GraftNetV2, GraftNetV3 subgraphs against the original GNN-RAG
test.json using MID-string triple sets (NOT raw integer IDs).

Why MID-strings: NSM/GraftNet/GraftNetV2 use extended entity vocabs whose integer
IDs are NOT comparable to the baseline vocab used in the original test.json.
GraftNetV3 uses a symlinked baseline vocab, so its integers ARE comparable — but
we normalise to strings anyway for a single consistent code path.

Metrics per question:
  jaccard, delta_SR, GED_norm   — on MID-string triple sets {(s_mid, r_str, o_mid)}
  ent_jaccard                   — on entity MID sets (from tuples + seed entities)
  seeds_same                    — whether seed MID sets match original
  answer_in_orig / answer_in_pert — gold answer MID present in subgraph
  hit                           — em_pert from inference result

Usage:
  python tmlr/scripts/analyze_subgraph_variants.py 2>&1 | tee new_exp/logs/subgraph_analysis.log
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

GNN_DATA    = REPO_ROOT / "external" / "GNN-RAG" / "data"
GNN_INPUT   = REPO_ROOT / "data" / "perturbations" / "gnn_input"
NEW_EXP     = REPO_ROOT / "data" / "new_exp_gnn_input"
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR     = RESULTS_DIR / "subgraph_overlap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DS_SUBDIR = {"cwq": "CWQ", "webqsp": "webqsp"}


# ── Vocab helpers ──────────────────────────────────────────────────────────────

def load_vocab(path: Path) -> Tuple[Dict[str, int], List[str]]:
    """Returns (str→id, id→str list)."""
    s2i: Dict[str, int] = {}
    i2s: List[str] = []
    with open(path) as f:
        for i, line in enumerate(f):
            tok = line.strip()
            s2i[tok] = i
            i2s.append(tok)
    return s2i, i2s


def get_vocabs(dataset: str, variant_dir: Path):
    """Load baseline vocab + variant-specific extended vocab.
    Returns (ent_i2s, rel_i2s) for the VARIANT's own integer space."""
    subdir = DS_SUBDIR[dataset]
    baseline_ent = GNN_DATA / subdir / "entities.txt"
    baseline_rel = GNN_DATA / subdir / "relations.txt"

    # Variant may have its own entities.txt (extended) or symlink to baseline
    var_ent = variant_dir / "entities.txt"
    var_rel = variant_dir / "relations.txt"

    ent_path = var_ent if var_ent.exists() else baseline_ent
    rel_path = var_rel if var_rel.exists() else baseline_rel

    _, ent_i2s = load_vocab(ent_path)
    _, rel_i2s = load_vocab(rel_path)
    return ent_i2s, rel_i2s


# ── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["id"]] = r
    return records


def decode_tuples(tuples: List, ent_i2s: List[str], rel_i2s: List[str]) -> Set[Tuple[str, str, str]]:
    """Convert integer triples to MID-string triples, dropping OOV."""
    out: Set[Tuple[str, str, str]] = set()
    ne, nr = len(ent_i2s), len(rel_i2s)
    for t in tuples:
        if len(t) < 3:
            continue
        s, r, o = int(t[0]), int(t[1]), int(t[2])
        if s < ne and r < nr and o < ne:
            out.add((ent_i2s[s], rel_i2s[r], ent_i2s[o]))
    return out


def decode_seeds(seed_ids: List[int], ent_i2s: List[str]) -> Set[str]:
    ne = len(ent_i2s)
    return {ent_i2s[i] for i in seed_ids if i < ne}


def ent_set_from_str_triples(triples: Set[Tuple[str, str, str]]) -> Set[str]:
    out: Set[str] = set()
    for s, _, o in triples:
        out.add(s); out.add(o)
    return out


# ── Metrics ────────────────────────────────────────────────────────────────────

def overlap_metrics(A: Set, B: Set) -> dict:
    if not A and not B:
        return dict(jaccard=1.0, delta_SR=0.0, GED=0, GED_norm=0.0,
                    orig_triples=0, pert_triples=0, shared=0)
    inter = len(A & B)
    union = len(A | B)
    ged   = len(A) + len(B) - 2 * inter
    maxAB = max(len(A), len(B))
    j = inter / union if union else 0.0
    return dict(jaccard=round(j, 6), delta_SR=round(1 - j, 6),
                GED=ged, GED_norm=round(ged / maxAB if maxAB else 0.0, 6),
                orig_triples=len(A), pert_triples=len(B), shared=inter)


# ── Core analysis ──────────────────────────────────────────────────────────────

def analyze_variant(
    dataset: str,
    ptype: str,
    variant: str,
    variant_base: Path,          # GNN_INPUT or NEW_EXP
    hit_path: Optional[Path] = None,
) -> Tuple[List[dict], dict]:
    subdir      = DS_SUBDIR[dataset]
    label       = f"{dataset}_{ptype}_{variant}"
    variant_dir = variant_base / f"{dataset}_{ptype}_{variant}"
    print(f"\n[{label}]")

    pert_path = variant_dir / "test_subgraph.json"
    if not pert_path.exists():
        print(f"  SKIP: {pert_path} not found")
        return [], {}

    # Load vocabs — use variant's own extended vocab for decoding its integer IDs
    orig_ent_i2s, orig_rel_i2s = get_vocabs(dataset, GNN_DATA / subdir)  # baseline only
    var_ent_i2s,  var_rel_i2s  = get_vocabs(dataset, variant_dir)

    # Load records
    orig = load_jsonl(GNN_DATA / subdir / "test.json")
    pert = load_jsonl(pert_path)
    print(f"  orig={len(orig)}  pert={len(pert)}")

    n_empty = sum(1 for r in pert.values() if not r.get("subgraph", {}).get("tuples"))
    print(f"  empty={n_empty}/{len(pert)}")

    # Load hit results
    hit_by_id: Dict[str, float] = {}
    if hit_path and hit_path.exists():
        try:
            with open(hit_path) as hf:
                data = json.load(hf)
            if isinstance(data, list):
                for rec in data:
                    hit_by_id[rec["id"]] = float(rec.get("em_pert", 0.0))
        except Exception as e:
            print(f"  Warning: could not load hit results: {e}")

    common_ids = sorted(set(orig) & set(pert))
    print(f"  common={len(common_ids)}")
    if not common_ids:
        return [], {}

    per_q: List[dict] = []
    jacs, dsrs, geds, gns, ejacs = [], [], [], [], []
    same_seed = ans_orig = ans_pert = hit_sum = n_hit = n_empty_q = 0

    for qid in common_ids:
        o_rec = orig[qid]
        p_rec = pert[qid]

        o_tuples = o_rec.get("subgraph", {}).get("tuples", [])
        p_tuples = p_rec.get("subgraph", {}).get("tuples", [])
        if not p_tuples:
            n_empty_q += 1

        # Decode to MID-string triple sets
        A = decode_tuples(o_tuples, orig_ent_i2s, orig_rel_i2s)
        B = decode_tuples(p_tuples, var_ent_i2s,  var_rel_i2s)

        m = overlap_metrics(A, B)
        jacs.append(m["jaccard"]); dsrs.append(m["delta_SR"])
        geds.append(m["GED"]);     gns.append(m["GED_norm"])

        # Entity Jaccard (MID strings)
        oe = ent_set_from_str_triples(A)
        pe = ent_set_from_str_triples(B)
        eu = len(oe | pe)
        ej = len(oe & pe) / eu if eu else (1.0 if not oe and not pe else 0.0)
        ejacs.append(ej)

        # Seed comparison (decode original integer seeds to MIDs)
        o_seeds = decode_seeds(o_rec.get("entities", []), orig_ent_i2s)
        p_seeds = decode_seeds(p_rec.get("entities", []), var_ent_i2s)
        if o_seeds == p_seeds:
            same_seed += 1

        # Answer coverage — gold MIDs from original record
        gold_mids: Set[str] = set()
        for a in o_rec.get("answers", []):
            kb = a.get("kb_id", "") if isinstance(a, dict) else str(a)
            if kb:
                gold_mids.add(kb)

        orig_all = oe | o_seeds
        pert_all = pe | p_seeds
        in_o = bool(gold_mids & orig_all)
        in_p = bool(gold_mids & pert_all)
        if in_o: ans_orig += 1
        if in_p: ans_pert += 1

        hit = hit_by_id.get(qid)
        if hit is not None:
            hit_sum += int(hit == 1.0); n_hit += 1

        per_q.append({
            "id": qid, "dataset": dataset, "ptype": ptype, "variant": variant,
            **m,
            "ent_jaccard": round(ej, 6),
            "seeds_same": o_seeds == p_seeds,
            "answer_in_orig": in_o,
            "answer_in_pert": in_p,
            "hit": hit,
            "pert_empty": not bool(p_tuples),
        })

    n = len(common_ids)
    hr = hit_sum / n_hit if n_hit else None

    summary = {
        "dataset": dataset, "ptype": ptype, "variant": variant, "n": n,
        "n_empty_pert": n_empty_q,
        "pct_empty_pert":     round(100.0 * n_empty_q / n, 2) if n else 0.0,
        "jaccard_mean":       round(statistics.mean(jacs),  4),
        "jaccard_median":     round(statistics.median(jacs), 4),
        "ent_jaccard_mean":   round(statistics.mean(ejacs), 4),
        "delta_SR_mean":      round(statistics.mean(dsrs),  4),
        "GED_mean":           round(statistics.mean(geds),  2),
        "GED_norm_mean":      round(statistics.mean(gns),   4),
        "pct_seeds_same":     round(100.0 * same_seed / n,  2) if n else 0.0,
        "pct_answer_in_orig": round(100.0 * ans_orig / n,   2) if n else 0.0,
        "pct_answer_in_pert": round(100.0 * ans_pert / n,   2) if n else 0.0,
        "hit_rate_gnn_rag":   round(hr, 4) if hr is not None else None,
        "n_with_hit": n_hit,
    }

    print(f"  jaccard(trip)={summary['jaccard_mean']:.4f}  "
          f"jaccard(ent)={summary['ent_jaccard_mean']:.4f}  "
          f"delta_SR={summary['delta_SR_mean']:.4f}")
    print(f"  seeds_same={summary['pct_seeds_same']:.1f}%  "
          f"ans_orig={summary['pct_answer_in_orig']:.1f}%  "
          f"ans_pert={summary['pct_answer_in_pert']:.1f}%  "
          f"hit={f'{hr*100:.2f}%' if hr is not None else 'N/A'}")

    return per_q, summary


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    RES = RESULTS_DIR

    VARIANTS = [
        # (dataset, ptype, variant, variant_base, hit_path)
        ("cwq",    "p7", "graftnet",   GNN_INPUT, RES/"e1_cwq_gnn_rag/full_graftnet/p7_per_question.json"),
        ("webqsp", "p7", "graftnet",   GNN_INPUT, RES/"e2_webqsp_gnn_rag/full_graftnet/p7_per_question.json"),
        ("cwq",    "p7", "nsm",        GNN_INPUT, RES/"e1_cwq_gnn_rag/full_nsm/p7_per_question.json"),
        ("webqsp", "p7", "nsm",        GNN_INPUT, RES/"e2_webqsp_gnn_rag/full_nsm/p7_per_question.json"),
        ("cwq",    "p4", "nsm",        GNN_INPUT, RES/"e1_cwq_gnn_rag/full_nsm/p4_per_question.json"),
        ("webqsp", "p4", "nsm",        GNN_INPUT, RES/"e2_webqsp_gnn_rag/full_nsm/p4_per_question.json"),
        ("cwq",    "p7", "graftnetv3", NEW_EXP,   RES/"e1_cwq_gnn_rag/full_graftnetv3/p7_per_question.json"),
        ("webqsp", "p7", "graftnetv3", NEW_EXP,   RES/"e2_webqsp_gnn_rag/full_graftnetv3/p7_per_question.json"),
        ("cwq",    "p4", "graftnetv3", NEW_EXP,   RES/"e1_cwq_gnn_rag/full_graftnetv3/p4_per_question.json"),
        ("webqsp", "p4", "graftnetv3", NEW_EXP,   RES/"e2_webqsp_gnn_rag/full_graftnetv3/p4_per_question.json"),
    ]

    all_results: Dict[str, dict] = {}
    all_per_q:   List[dict] = []

    for ds, pt, var, vbase, hp in VARIANTS:
        pq, sm = analyze_variant(ds, pt, var, vbase, hp)
        if sm:
            all_results[f"{ds}_{pt}_{var}"] = sm
            all_per_q.extend(pq)

    # Save
    out_s = OUT_DIR / "all_variants_summary.json"
    out_p = OUT_DIR / "all_variants_per_question.json"
    with open(out_s, "w") as f: json.dump(all_results, f, indent=2)
    with open(out_p, "w") as f: json.dump(all_per_q,   f, indent=2)
    print(f"\nSaved → {out_s}")

    # Print table
    print("\n" + "="*100)
    print(f"{'Key':<35} {'n':>5} {'empty%':>7} {'trip_jac':>9} {'ent_jac':>8} "
          f"{'seeds%':>7} {'ans_pert%':>10} {'ans_orig%':>10} {'hit%':>7}")
    print("-"*100)
    for key, s in all_results.items():
        hr = f"{s['hit_rate_gnn_rag']*100:.2f}%" if s.get('hit_rate_gnn_rag') is not None else "N/A"
        print(f"{key:<35} {s['n']:>5} {s['pct_empty_pert']:>6.1f}% "
              f"{s['jaccard_mean']:>9.4f} {s['ent_jaccard_mean']:>8.4f} "
              f"{s['pct_seeds_same']:>6.1f}% {s['pct_answer_in_pert']:>9.1f}% "
              f"{s['pct_answer_in_orig']:>9.1f}% {hr:>7}")


if __name__ == "__main__":
    main()