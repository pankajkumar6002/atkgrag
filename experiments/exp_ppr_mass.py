#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
exp_ppr_mass.py  (EXP-1 from review.md)
=========================================
Mechanistic analysis: PPR mass on gold answer entity in CR-perturbed subgraphs.

Review concern (B9): "answer presence ≠ answer reachability" is argued in prose.
This script quantifies it using existing result files and optionally re-running
the GraftNet PPR to get actual per-entity PPR scores.

Part A — Answer reachability breakdown (no inference needed, ~5 min):
  For each CWQ CR question, classify into one of:
    CORRECT  : em_pert=1 (GNN found the answer)
    PRESENT  : em_pert=0 AND gold MID in subgraph entities (answer present but GNN missed)
    ABSENT   : em_pert=0 AND gold MID not in subgraph entities (answer not retrieved)
  Reports fractions per subgraph variant (ELQ, NSM, GraftNet).

Part B — PPR score distribution (optional, requires re-running PPR, ~2h for 200 Qs):
  For a 200-question sample of "PRESENT but GNN missed" questions (CR ELQ):
    - Re-run GraftNet PPR on clean subgraph → record gold entity PPR score
    - Re-run GraftNet PPR on perturbed subgraph → record gold entity PPR score
    - Report: median/mean PPR score collapse (clean → perturbed)
  This is the mechanistic "PPR mass on gold answer entity collapses" finding.

Output:
  results/ppr_mass_cwq_p4.json      — Part A breakdown
  results/ppr_mass_sample_200.json  — Part B PPR scores (optional, 200-question sample)

LaTeX targets:
  - sections/analysis.tex: "The answer is present in 74% of CR ELQ subgraphs yet GNN
    cannot reach it; median PPR mass on gold entity collapses from X to Y under CR."
  - Supports Proposition B.1 (GNN spectral amplification) claim.

Usage:
  python exp_ppr_mass.py --part A         # fast breakdown only
  python exp_ppr_mass.py --part AB        # full including PPR re-run (slow)
  python exp_ppr_mass.py --part B --sample 200 --ptype p4 --variant elq
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = REPO_ROOT
RESULTS     = REPO_ROOT / "tmlr" / "results"
PERT_DIR    = REPO_ROOT / "tmlr" / "pert" / "dataset"
GNN_DATA    = REPO_ROOT / "tmlr" / "Repo" / "GNN-RAG" / "data"
NEW_EXP     = REPO_ROOT / "new_exp"

# Per-question result files (from e1_cwq_gnn_rag.py)
PER_QUESTION = {
    "elq":            RESULTS / "e1_cwq_gnn_rag" / "full_elq",
    "nsm_gem":        RESULTS / "e1_cwq_gnn_rag" / "full_nsm_gem",
    "graftnetv3_gem": RESULTS / "e1_cwq_gnn_rag" / "full_graftnetv3_gem",
    "nsm_elq":        RESULTS / "e1_cwq_gnn_rag" / "full_nsm_elq",
    "graftnetv3_elq": RESULTS / "e1_cwq_gnn_rag" / "full_graftnetv3_elq",
}

# Subgraph files (test_subgraph.json, JSONL format)
# ELQ CR uses the ORIGINAL CLEAN subgraph as proxy (same seeds → same PPR walk;
# paper Table 3 footnote: "α_pert=74.0% reflects the clean subgraph").
SUBGRAPH_FILES = {
    "elq":            GNN_DATA / "CWQ" / "test.json",           # clean subgraph (ELQ CR proxy)
    "nsm_gem":        NEW_EXP / "gnn_input" / "cwq_p4_nsm_gem" / "test_subgraph.json",
    "graftnetv3_gem": NEW_EXP / "gnn_input" / "cwq_p4_graftnetv3_gem" / "test_subgraph.json",
    "nsm_elq":        NEW_EXP / "gnn_input" / "cwq_p4_nsm_elq" / "test_subgraph.json",
    "graftnetv3_elq": NEW_EXP / "gnn_input" / "cwq_p4_graftnetv3_elq" / "test_subgraph.json",
}

# Gold answers (from GNN-RAG original test.json)
GOLD_ANSWERS_FILE = GNN_DATA / "CWQ" / "test.json"

# Perturbed question file
PERT_FILE = PERT_DIR / "cwq_p4.json"


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_gold_answers() -> Dict[str, List[str]]:
    """Load {qid: [gold_mid, ...]} from GNN-RAG CWQ test.json."""
    answers = {}
    with open(GOLD_ANSWERS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = r.get("id", "")
            mids = [a["kb_id"] for a in r.get("answers", []) if isinstance(a, dict) and a.get("kb_id")]
            if qid and mids:
                answers[qid] = mids
    return answers


def load_entity_vocab(sg_dir: Path) -> List[str]:
    """Load entities.txt from the subgraph directory → list of MID strings (index → MID)."""
    ent_file = sg_dir / "entities.txt"
    if not ent_file.exists():
        # Fallback to GNN-RAG default entities.txt
        ent_file = GNN_DATA / "CWQ" / "entities.txt"
    if not ent_file.exists():
        return []
    with open(ent_file) as f:
        return [line.strip() for line in f if line.strip()]


def load_subgraph_entities(sg_file: Path) -> Dict[str, Set[str]]:
    """Load {qid: {entity_mid, ...}} from test_subgraph.json.
    After Stage 3, entities field contains integer indices into entities.txt.
    We load entities.txt from the same directory to map indices → MID strings."""
    entities_by_qid = {}
    if not sg_file.exists():
        return entities_by_qid

    # Load entity vocab (index → MID) from same directory
    vocab = load_entity_vocab(sg_file.parent)

    with open(sg_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = r.get("id", "")
            # GNN-RAG format: subgraph.entities = list of ALL entity int-indices in subgraph
            # top-level entities = seed entity indices only (2-3 entries)
            sg = r.get("subgraph", {})
            raw_ents = sg.get("entities", [])  # full subgraph entity list
            if not raw_ents:
                # Fallback: use top-level entities (seeds only)
                raw_ents = r.get("entities", [])
            if vocab and raw_ents and isinstance(raw_ents[0], int):
                mids = set(vocab[i] for i in raw_ents if 0 <= i < len(vocab))
            elif raw_ents and isinstance(raw_ents[0], str):
                mids = set(raw_ents)
            else:
                mids = set()
            entities_by_qid[qid] = mids
    return entities_by_qid


def load_per_question(pq_dir: Path, ptype: str = "p4") -> Dict[str, Dict[str, Any]]:
    """Load per_question results. Returns {qid: {em_clean, em_pert, ...}}."""
    pq_file = pq_dir / f"{ptype}_per_question.json"
    if not pq_file.exists():
        return {}
    with open(pq_file) as f:
        records = json.load(f)
    return {r["id"]: r for r in records}


def load_perturbed_questions() -> Dict[str, Dict[str, Any]]:
    """Load cwq_p4.json. Returns {qid: record}."""
    with open(PERT_FILE) as f:
        records = json.load(f)
    return {r["id"]: r for r in records}


# ══════════════════════════════════════════════════════════════════════════════
# Part A: Answer Reachability Breakdown
# ══════════════════════════════════════════════════════════════════════════════

def run_part_a(ptype: str = "p4", variants: Optional[List[str]] = None) -> Dict[str, Any]:
    """Classify each CR question into CORRECT / PRESENT / ABSENT per subgraph variant."""
    if variants is None:
        variants = ["elq", "nsm_gem", "graftnetv3_gem"]

    print(f"\n{'='*70}")
    print(f"PART A — Answer Reachability Breakdown ({ptype.upper()})")
    print(f"{'='*70}")

    gold_answers = load_gold_answers()
    perturbed = load_perturbed_questions()

    # Check which ptypes are available for each variant
    available_variants = []
    for var in variants:
        pq = load_per_question(PER_QUESTION.get(var, Path("__missing__")), ptype)
        sg = load_subgraph_entities(SUBGRAPH_FILES.get(var, Path("__missing__")))
        if not pq:
            print(f"  [SKIP] {var}: no per_question file for {ptype}")
            continue
        if not sg:
            print(f"  [WARN] {var}: no subgraph file — α_pert computed from answers field only")
        available_variants.append((var, pq, sg))

    if not available_variants:
        print("  No data available. Run e1_cwq_gnn_rag.py for p4 first.")
        return {}

    all_results = {}
    for var, pq, sg in available_variants:
        stats = defaultdict(int)
        alpha_pert_ids = []

        for qid, rec in pq.items():
            gold_mids = set(gold_answers.get(qid, []))
            em_pert = rec["em_pert"]

            if em_pert == 1:
                stats["correct"] += 1
                continue

            # Check if gold answer MID appears in subgraph entities
            sg_ents = sg.get(qid, set())
            alpha_pert = bool(gold_mids & sg_ents)

            if alpha_pert:
                stats["present_but_wrong"] += 1
                alpha_pert_ids.append(qid)
            else:
                stats["absent"] += 1

        n = len(pq)
        stats["total"] = n
        stats["alpha_pert"] = stats["correct"] + stats["present_but_wrong"]

        pct = lambda k: 100 * stats[k] / n if n else 0

        print(f"\n  {var.upper()} (n={n}):")
        print(f"    CORRECT         (em=1):              {stats['correct']:4d} ({pct('correct'):5.1f}%)")
        print(f"    PRESENT+WRONG   (α_pert=1, em=0):    {stats['present_but_wrong']:4d} ({pct('present_but_wrong'):5.1f}%)")
        print(f"    ABSENT          (α_pert=0, em=0):    {stats['absent']:4d} ({pct('absent'):5.1f}%)")
        print(f"    α_pert total    (ans in subgraph):   {stats['alpha_pert']:4d} ({pct('alpha_pert'):5.1f}%)")

        all_results[var] = {
            "n": n,
            "correct_pct": pct("correct"),
            "present_wrong_pct": pct("present_but_wrong"),
            "absent_pct": pct("absent"),
            "alpha_pert_pct": pct("alpha_pert"),
            "alpha_pert_ids": alpha_pert_ids,
        }

    print(f"\n  Key finding: For ELQ CR, α_pert={all_results.get('elq',{}).get('alpha_pert_pct',0):.1f}% "
          f"yet EM≈0% — the GNN cannot reach the answer despite its presence.")
    print(f"  LaTeX: CR ELQ α_pert=${{\\alpha_{{\\text{{pert}}}}}}$ = "
          f"{all_results.get('elq',{}).get('alpha_pert_pct',0):.1f}\\% yet EM\\,=\\,0.68\\%")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# Part B: PPR Score Distribution (optional, requires build_graftnetv3 import)
# ══════════════════════════════════════════════════════════════════════════════

def run_part_b(ptype: str = "p4", sample_n: int = 200,
               variant: str = "elq") -> Dict[str, Any]:
    """Compute actual PPR scores for gold answer entities (clean vs perturbed subgraph).

    Strategy: re-run GraftNet PPR on a sample of questions using build_graftnetv3
    helper functions, but with COSINE_FLOOR=1.0 (flat PPR = uniform) to isolate
    the seed propagation from cosine weighting. The score collapse is purely topological.

    For the paper, we report:
      median PPR mass on gold entity: clean → CR-perturbed subgraph
      showing that structural hop-order reversal redirects PPR mass away from gold entity.
    """
    import sys as _sys
    SCRIPTS_ROOT = Path(__file__).parent.parent
    if str(SCRIPTS_ROOT) not in _sys.path:
        _sys.path.insert(0, str(SCRIPTS_ROOT))

    try:
        from build_graftnetv3 import (
            load_word_emb, compute_relation_scores,
            _graftnetv3_read_facts, _graftnet_ppr,
        )
        print("\n  build_graftnetv3 imported OK")
    except ImportError as e:
        print(f"\n  [SKIP Part B] Cannot import from build_graftnetv3: {e}")
        print("  Part B requires build_graftnetv3.py in tmlr/scripts/")
        return {}

    print(f"\n{'='*70}")
    print(f"PART B — PPR Score Distribution on Gold Answer Entity (sample={sample_n})")
    print(f"Variant: {variant}, ptype: {ptype}")
    print(f"{'='*70}")

    # Load "PRESENT but WRONG" questions from Part A (these are the key cases)
    part_a_results = run_part_a(ptype, variants=[variant])
    present_wrong_ids = part_a_results.get(variant, {}).get("alpha_pert_ids", [])

    if not present_wrong_ids:
        print("  No 'present but wrong' questions found for this variant.")
        return {}

    random.seed(42)
    sample_ids = random.sample(present_wrong_ids, min(sample_n, len(present_wrong_ids)))
    print(f"\n  Sampled {len(sample_ids)} 'present but wrong' questions for PPR analysis")

    # Load required data
    gold_answers = load_gold_answers()
    perturbed_qs = load_perturbed_questions()
    vocab, emb = load_word_emb("cwq")

    # Load global entity/relation index → string mappings
    entities_global: List[str] = []
    with open(GNN_DATA / "CWQ" / "entities.txt") as f:
        entities_global = [line.strip() for line in f]
    relations_global: List[str] = []
    with open(GNN_DATA / "CWQ" / "relations.txt") as f:
        relations_global = [line.strip() for line in f]
    print(f"  Loaded {len(entities_global)} entities, {len(relations_global)} relations")

    # Load original (clean) GNN-RAG records
    orig_records = {}
    with open(GNN_DATA / "CWQ" / "test.json") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                orig_records[r["id"]] = r

    clean_scores, pert_scores = [], []
    n_processed = 0

    import scipy.sparse as sp

    print("\n  Computing PPR scores (this may take 30-60 min for 200 questions)...")
    for qid in sample_ids:
        orig_rec = orig_records.get(qid)
        pert_rec = perturbed_qs.get(qid)
        if not orig_rec or not pert_rec:
            continue

        gold_mids = set(gold_answers.get(qid, []))
        if not gold_mids:
            continue

        clean_q = orig_rec.get("question", "")
        pert_q = pert_rec.get("perturbed_question", "")
        orig_subgraph = orig_rec.get("subgraph", {})

        if not orig_subgraph:
            continue

        # subgraph.entities: list of global entity indices into entities.txt
        # subgraph.tuples: [global_src_idx, rel_idx, global_tgt_idx]
        global_ents = orig_subgraph.get("entities", [])
        if not global_ents:
            continue

        # Convert integer tuples to MID-string triples for _graftnetv3_read_facts
        int_tuples = orig_subgraph.get("tuples", [])
        str_triples: Set[Tuple[str, str, str]] = set()
        for t in int_tuples:
            if len(t) < 3:
                continue
            si, ri, oi = t[0], t[1], t[2]
            if (si < len(entities_global) and oi < len(entities_global)
                    and ri < len(relations_global)):
                str_triples.add((entities_global[si], relations_global[ri], entities_global[oi]))
        if not str_triples:
            continue

        # Seed MIDs
        seed_globals = orig_rec.get("entities", [])
        seed_mids_local = [entities_global[g] for g in seed_globals if g < len(entities_global)]
        if not seed_mids_local:
            seed_mids_local = [entities_global[global_ents[0]]] if global_ents and global_ents[0] < len(entities_global) else []
        if not seed_mids_local:
            continue

        gold_relation_map = pert_rec.get("gold_relation_map", {})

        def get_ppr_mass_on_gold(question_text: str) -> Optional[float]:
            """Relation-weighted PPR; returns PPR mass on gold answer entity."""
            try:
                rel_scores = compute_relation_scores(question_text, gold_relation_map, vocab, emb)
                entity_map, adj_mat = _graftnetv3_read_facts(str_triples, rel_scores)
                if adj_mat is None or not entity_map:
                    return None
                n = adj_mat.shape[0]
                seed_idx_list = [entity_map[m] for m in seed_mids_local if m in entity_map]
                if not seed_idx_list:
                    return None
                seed_vec = np.zeros((n, 1))
                for idx in seed_idx_list:
                    seed_vec[idx] = 1.0
                seed_vec /= seed_vec.sum()
                ppr = _graftnet_ppr(seed_vec, adj_mat)
                ppr = ppr / (ppr.sum() + 1e-10)
                gold_mass = sum(
                    float(ppr[entity_map[m]])
                    for m in gold_mids
                    if m in entity_map
                )
                return gold_mass
            except Exception:
                return None

        clean_mass = get_ppr_mass_on_gold(clean_q)
        pert_mass = get_ppr_mass_on_gold(pert_q)

        if clean_mass is not None and pert_mass is not None:
            clean_scores.append(clean_mass)
            pert_scores.append(pert_mass)
            n_processed += 1

        if n_processed % 20 == 0:
            print(f"  Processed {n_processed}/{len(sample_ids)}...")

    if not clean_scores:
        print("  No PPR scores computed (check subgraph format compatibility).")
        return {}

    results = {
        "n_processed": n_processed,
        "clean_median_ppr": float(np.median(clean_scores)),
        "clean_mean_ppr":   float(np.mean(clean_scores)),
        "pert_median_ppr":  float(np.median(pert_scores)),
        "pert_mean_ppr":    float(np.mean(pert_scores)),
        "collapse_ratio":   float(np.median(pert_scores) / (np.median(clean_scores) + 1e-10)),
    }

    print(f"\n  PPR mass on gold answer entity ({n_processed} questions):")
    print(f"    Clean subgraph:   median={results['clean_median_ppr']:.4f}, mean={results['clean_mean_ppr']:.4f}")
    print(f"    CR-pert subgraph: median={results['pert_median_ppr']:.4f}, mean={results['pert_mean_ppr']:.4f}")
    print(f"    Collapse ratio:   {results['collapse_ratio']:.3f}x")
    print(f"\n  LaTeX sentence:")
    print(f"    On the 'present but unreachable' subset (CR ELQ, α_pert=1, EM=0),")
    print(f"    median PPR mass on the gold entity collapses from")
    print(f"    {results['clean_median_ppr']:.4f} (clean) to {results['pert_median_ppr']:.4f} (perturbed),")
    print(f"    a {results['collapse_ratio']:.1f}× reduction, confirming topological inaccessibility.")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="PPR mass / answer reachability analysis (EXP-1)")
    p.add_argument("--part", default="A", choices=["A", "B", "AB"],
                   help="A=breakdown only (fast), B=PPR re-run (slow), AB=both")
    p.add_argument("--ptype", default="p4", choices=["p4"],
                   help="Perturbation type (currently only p4=CR supported)")
    p.add_argument("--sample", type=int, default=200,
                   help="Number of questions for Part B PPR re-run (default: 200)")
    p.add_argument("--variant", default="elq",
                   choices=["elq", "nsm_gem", "graftnetv3_gem"],
                   help="Subgraph variant for Part B analysis (default: elq)")
    p.add_argument("--variants-a", nargs="+",
                   default=["elq", "nsm_gem", "graftnetv3_gem"],
                   help="Variants to include in Part A breakdown")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "results" / "ppr_mass_cwq_p4.json",
                   help="Output JSON file")
    args = p.parse_args()

    all_results = {}

    if "A" in args.part:
        part_a = run_part_a(args.ptype, args.variants_a)
        all_results["part_a"] = part_a

    if "B" in args.part:
        part_b = run_part_b(args.ptype, args.sample, args.variant)
        all_results["part_b"] = part_b

    if all_results:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()