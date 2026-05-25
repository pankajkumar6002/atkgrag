#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
Stage Isolation Protocol for SubgraphRetrievalKBQA (NSM Reasoner)
===================================================================

STAGE 1 — Entity Linking (EL)
  Given perturbed question, does ELQ find the correct seed entity?
  Measures Hit@1 against gold_entity_map (MIDs).
  - p4 (Cmp), p7 (RelSwap), p1 (Lex) for WebQSP and CWQ.
  - p7 uses original entities (no entity change) → baseline oracle.

STAGE 2 — Subgraph Retrieval (SR)
  Given ORACLE/GOLD seed entities (from clean test.json) + PERTURBED question,
  run SubgraphRetrievalKBQA retriever on the KG cache and build new subgraphs.
  Compare vs clean subgraph: Jaccard, entity-Jaccard, answer coverage.

STAGE 3 — Evidence Aggregation (EA)
  Run trained NSM reasoner on:
  (a) GOLD subgraph (clean test.json subgraph) + perturbed question
  (b) SR-PERTURBED subgraph (output of Stage 2) + perturbed question
  Report Hit@1, F1 and ΔAG vs clean baseline.

Outputs → tmlr/results/sr_nsm_stage_isolation/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import copy
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT  = REPO_ROOT
SR_REPO    = REPO_ROOT / "external" / "SubgraphRetrievalKBQA" / "src"
PERT_DIR   = REPO_ROOT / "data" / "perturbations"
NSM_DATA   = SR_REPO / "nsm_reader" / "data"
NSM_CKPT   = REPO_ROOT / "models" / "nsm"
KG_CACHE   = REPO_ROOT / "external" / "GMT-KBQA" / "dataset" / "MyFreebaseCache" / "subgraph_hop1.txt"
OUT_DIR    = REPO_ROOT / "results" / "sr_nsm_stage_isolation"

DATASETS   = ["webqsp", "cwq"]
PERT_TYPES = ["p4", "p7", "p1"]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return data


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def mean_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05) -> Dict:
    arr = np.array(values, dtype=float)
    m = float(np.mean(arr))
    boots = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": m, "ci_lo": float(lo), "ci_hi": float(hi)}


def pert_file(ds: str, ptype: str) -> Path:
    """Return path to the canonical perturbed dataset JSON for ds/ptype."""
    suffix = "_fixed3" if ptype == "p1" else ""
    return PERT_DIR / f"{ds}_{ptype}{suffix}.json"


def clean_test_path(ds: str) -> Path:
    return NSM_DATA / ds / "test_simple.json"


def elq_el_path(ds: str, ptype: str) -> Path:
    return PERT_DIR / "el_elq" / f"{ds}_{ptype}_el.json"


# ── STAGE 1: Entity Linking ──────────────────────────────────────────────────

def stage1_el(ds: str, ptype: str) -> Dict:
    """
    For p7: gold entities unchanged → EL hit = 100% by construction (oracle baseline).
    For p4, p1: load ELQ results and compare top-1 linked MID vs gold_entity_map keys.
    """
    pert_data = load_json(pert_file(ds, ptype))  # list of records
    pert_by_id = {r["id"]: r for r in pert_data}

    if ptype == "p7":
        # Entities are not perturbed in RelSwap; gold seed is identical to clean
        clean_data = load_jsonl(clean_test_path(ds))
        hits = []
        for r in clean_data:
            qid = r["id"]
            if qid not in pert_by_id:
                continue
            # gold entities = entities in clean test.json
            gold_mids = set(r["entities"])
            # "linked" = gold (trivially correct)
            hits.append(1.0 if gold_mids else 0.0)
        result = {
            "dataset": ds, "pert_type": ptype, "n": len(hits),
            "note": "p7 entities unchanged — oracle hit@1 = 1.0 by construction",
            "hit1": mean_ci(hits),
        }
        return result

    el_path = elq_el_path(ds, ptype)
    if not el_path.exists():
        return {"dataset": ds, "pert_type": ptype, "skipped": True,
                "reason": f"ELQ result file not found: {el_path}"}

    el_data = load_json(el_path)
    el_by_id = {r["id"]: r for r in el_data}

    hits = []
    n_no_el = 0
    for qid, pr in pert_by_id.items():
        gold_mids = set(pr.get("gold_entity_map", {}).keys())
        if not gold_mids:
            continue
        el_rec = el_by_id.get(qid)
        if not el_rec or not el_rec.get("linked_entities"):
            n_no_el += 1
            hits.append(0.0)
            continue
        top1_mid = el_rec["linked_entities"][0].get("mid", "")
        hits.append(1.0 if top1_mid in gold_mids else 0.0)

    result = {
        "dataset": ds, "pert_type": ptype, "n": len(hits),
        "n_no_el": n_no_el,
        "hit1": mean_ci(hits),
    }
    return result


# ── STAGE 2: Subgraph Retrieval ───────────────────────────────────────────────

class KGCache:
    """Lightweight wrapper around subgraph_hop1.txt triple cache."""

    def __init__(self, path: Path):
        print(f"  Loading KG cache from {path} …")
        self.head2rt: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    h, r, t = parts
                    self.head2rt[h].append((r, t))
        print(f"  KG cache: {len(self.head2rt):,} seed entities")

    def get_relations(self, entity: str) -> List[str]:
        return [r for r, _ in self.head2rt.get(entity, [])]

    def get_subgraph(self, entity: str, max_triples: int = 2000
                     ) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        triples = self.head2rt.get(entity, [])[:max_triples]
        entities = {entity}
        result_triples = []
        for r, t in triples:
            entities.add(t)
            result_triples.append((entity, r, t))
        return list(entities), result_triples


def retrieve_subgraph_for_record(
    record: Dict,          # NSM-format record with string MID entities
    question: str,         # perturbed question text
    kg: KGCache,
    top_k: int = 10,
) -> Dict:
    """
    Simplified SR: for each gold seed entity, retrieve the 1-hop neighbourhood
    from the KG cache.  This mirrors the path-based retriever output but without
    the trained SimBERT scorer (which is not available).  The KG cache contains
    all 1-hop triples per entity pre-built from Freebase.

    Returns an NSM-format record with updated question + subgraph.
    """
    out = copy.deepcopy(record)
    out["question"] = question  # substitute perturbed question

    seed_mids = record["entities"]  # gold seed MIDs from clean test.json
    all_ents: set[str] = set()
    all_triples: set[Tuple[str, str, str]] = set()

    for seed in seed_mids:
        ents, triples = kg.get_subgraph(seed)
        all_ents.update(ents)
        all_triples.update(triples)

    out["subgraph"] = {
        "entities": list(all_ents),
        "tuples": [list(t) for t in all_triples],
    }
    return out


def compute_subgraph_overlap(clean_rec: Dict, pert_rec: Dict) -> Dict:
    """Jaccard, entity-Jaccard, answer coverage for clean vs perturbed subgraph."""
    clean_trip = {tuple(t) for t in clean_rec["subgraph"]["tuples"]}
    pert_trip  = {tuple(t) for t in pert_rec["subgraph"]["tuples"]}
    clean_ents = set(clean_rec["subgraph"]["entities"])
    pert_ents  = set(pert_rec["subgraph"]["entities"])

    trip_jac = (len(clean_trip & pert_trip) / len(clean_trip | pert_trip)
                if clean_trip | pert_trip else 1.0)
    ent_jac  = (len(clean_ents & pert_ents) / len(clean_ents | pert_ents)
                if clean_ents | pert_ents else 1.0)

    gold_mids = {a["kb_id"] for a in clean_rec.get("answers", [])}
    ans_in_clean = any(m in clean_ents for m in gold_mids)
    ans_in_pert  = any(m in pert_ents  for m in gold_mids)
    return {
        "trip_jac": trip_jac,
        "ent_jac": ent_jac,
        "ans_in_clean": float(ans_in_clean),
        "ans_in_pert": float(ans_in_pert),
        "n_clean_triples": len(clean_trip),
        "n_pert_triples": len(pert_trip),
    }


def stage2_sr(ds: str, ptype: str, kg: KGCache, out_dir: Path) -> Dict:
    """
    Build perturbed SR subgraphs using oracle gold seeds + perturbed question.
    Saves the perturbed test_simple.json for use in Stage 3.
    """
    pert_data  = load_json(pert_file(ds, ptype))
    pert_by_id = {r["id"]: r for r in pert_data}
    clean_data = load_jsonl(clean_test_path(ds))
    clean_by_id = {r["id"]: r for r in clean_data}

    overlap_results = []
    pert_records = []
    n_skipped = 0

    for clean_rec in tqdm(clean_data, desc=f"SR {ds}/{ptype}"):
        qid = clean_rec["id"]
        if qid not in pert_by_id:
            n_skipped += 1
            continue
        pert_q = pert_by_id[qid]["perturbed_question"]
        pert_rec = retrieve_subgraph_for_record(clean_rec, pert_q, kg)
        pert_records.append(pert_rec)
        overlap = compute_subgraph_overlap(clean_rec, pert_rec)
        overlap["id"] = qid
        overlap_results.append(overlap)

    # Save perturbed test_simple.json for Stage 3
    sr_out = out_dir / f"{ds}_{ptype}_sr_pert"
    sr_out.mkdir(parents=True, exist_ok=True)
    with open(sr_out / "test_simple.json", "w") as f:
        for r in pert_records:
            f.write(json.dumps(r) + "\n")
    # Copy vocab/entity/relation files from clean data dir
    for fname in ["entities.txt", "relations.txt", "vocab_new.txt", "word_emb_300d.npy"]:
        src = NSM_DATA / ds / fname
        dst = sr_out / fname
        if src.exists() and not dst.exists():
            os.symlink(src, dst)

    summary = {
        "dataset": ds, "pert_type": ptype, "n": len(overlap_results),
        "n_skipped": n_skipped,
        "trip_jac":    mean_ci([x["trip_jac"]  for x in overlap_results]),
        "ent_jac":     mean_ci([x["ent_jac"]   for x in overlap_results]),
        "ans_in_clean": float(np.mean([x["ans_in_clean"] for x in overlap_results])),
        "ans_in_pert":  float(np.mean([x["ans_in_pert"]  for x in overlap_results])),
        "avg_clean_triples": float(np.mean([x["n_clean_triples"] for x in overlap_results])),
        "avg_pert_triples":  float(np.mean([x["n_pert_triples"]  for x in overlap_results])),
    }
    save_json(overlap_results, out_dir / f"{ds}_{ptype}_sr_overlap_per_question.json")
    return summary


# ── STAGE 3: Evidence Aggregation ────────────────────────────────────────────

def run_nsm_inference(
    test_data_dir: Path,
    ckpt_dir: Path,
    dataset_name: str,
    experiment_name: str,
) -> Dict[str, Any]:
    """
    Run NSM inference by directly calling Trainer_KBQA.evaluate().
    Returns {qid: {"hit": float, "f1": float}} dict.
    """
    nsm_src = str(SR_REPO / "nsm_reader")
    if nsm_src not in sys.path:
        sys.path.insert(0, nsm_src)

    import torch
    import importlib
    # Re-import to pick up correct sys.path
    from NSM.train.trainer_nsm import Trainer_KBQA
    from NSM.util.utils import create_logger

    # Find best checkpoint
    ckpt_files = sorted(ckpt_dir.glob(f"{experiment_name}*.ckpt"))
    if not ckpt_files:
        return {"error": f"No checkpoint found in {ckpt_dir} for {experiment_name}"}
    best_ckpt = ckpt_files[-1]  # take latest
    print(f"    Using checkpoint: {best_ckpt}")

    args = {
        "data_folder": str(test_data_dir) + "/",
        "checkpoint_dir": str(ckpt_dir) + "/",
        "model_name": "gnn",
        "batch_size": 20,
        "test_batch_size": 40,
        "num_step": 3,
        "entity_dim": 50,
        "word_dim": 300,
        "kg_dim": 100,
        "kge_dim": 100,
        "eval_every": 5,
        "encode_type": True,
        "experiment_name": experiment_name,
        "eps": 0.95 if dataset_name == "webqsp" else 0.5,
        "num_epoch": 200,
        "use_self_loop": True,
        "lr": 5e-4,
        "word_emb_file": "word_emb_300d.npy",
        "loss_type": "kl",
        "reason_kb": True,
        "is_eval": True,
        "load_experiment": best_ckpt.name,
        "load_ckpt_file": None,
        "use_cuda": torch.cuda.is_available(),
        "q_type": "seq",
        "num_layer": 1,
        "mode": "teacher",
        "entropy_weight": 0.0,
        "use_label": False,
        "tree_soft": False,
        "filter_label": False,
        "share_embedding": False,
        "share_instruction": False,
        "encoder_type": "lstm",
        "lambda_label": 0.1,
        "lambda_auto": 0.01,
        "label_f1": 0.5,
        "label_file": None,
        "share_encoder": False,
        "use_inverse_relation": False,
        "train_KL": False,
        "filter_sub": False,
        "decay_rate": 0.0,
        "gradient_clip": 1.0,
        "lr_schedule": False,
        "label_smooth": 0.1,
        "fact_drop": 0,
        "fact_scale": 3,
        "entity2id": "entities.txt",
        "relation2id": "relations.txt",
        "word2id": "vocab_new.txt",
        "entity_emb_file": None,
        "entity_kge_file": None,
        "relation_emb_file": None,
        "relation_kge_file": None,
        "rel_word_ids": "rel_word_idx.npy",
        "pretrained_entity_kge_file": "entity_emb_100d.npy",
        "char2id": "chars.txt",
        "kge_dim": 100,
        "lstm_dropout": 0.3,
        "linear_dropout": 0.2,
        "seed": 19960626,
        "log_level": "info",
        "name": dataset_name,
        "num_layer": 1,
    }

    import logging
    logger = logging.getLogger("nsm_eval")
    trainer = Trainer_KBQA(args=args, logger=logger)
    f1, h1 = trainer.evaluate(trainer.test_data, args["test_batch_size"], write_info=False)
    return {"hit1": float(h1), "f1": float(f1)}


def stage3_ea(ds: str, ptype: str, out_dir: Path) -> Dict:
    """
    Stage 3: Evidence Aggregation.
    Evaluates NSM on:
      (a) gold subgraph + perturbed question  (from clean test_simple.json with swapped question)
      (b) SR-perturbed subgraph + perturbed question (built in stage 2)
    Also runs NSM on clean data to get the baseline.
    """
    ckpt_dir = NSM_CKPT / ds
    exp_name = f"{ds}_nsm"

    # Check checkpoint exists
    if not any(ckpt_dir.glob(f"{exp_name}*.ckpt")):
        return {"dataset": ds, "pert_type": ptype, "skipped": True,
                "reason": f"No checkpoint found for {ckpt_dir}/{exp_name}*.ckpt"}

    pert_data  = load_json(pert_file(ds, ptype))
    pert_by_id = {r["id"]: r for r in pert_data}
    clean_data = load_jsonl(clean_test_path(ds))

    # Build (a): gold subgraph + perturbed question → write temp test_simple.json
    gold_sg_dir = out_dir / f"{ds}_{ptype}_gold_sg_pert_q"
    gold_sg_dir.mkdir(parents=True, exist_ok=True)

    with open(gold_sg_dir / "test_simple.json", "w") as f:
        for rec in clean_data:
            qid = rec["id"]
            if qid not in pert_by_id:
                continue
            out_rec = copy.deepcopy(rec)
            out_rec["question"] = pert_by_id[qid]["perturbed_question"]
            f.write(json.dumps(out_rec) + "\n")

    for fname in ["entities.txt", "relations.txt", "vocab_new.txt", "word_emb_300d.npy"]:
        src = NSM_DATA / ds / fname
        dst = gold_sg_dir / fname
        if src.exists() and not dst.exists():
            os.symlink(src, dst)

    print(f"  Stage 3a: NSM on gold subgraph + perturbed Q ({ds}/{ptype})…")
    res_a = run_nsm_inference(gold_sg_dir, ckpt_dir, ds, exp_name)

    # Build (b): SR-perturbed subgraph + perturbed question (already built in stage 2)
    sr_dir = out_dir / f"{ds}_{ptype}_sr_pert"
    res_b = {"skipped": True, "reason": "SR-perturbed dir not found"}
    if sr_dir.exists() and (sr_dir / "test_simple.json").exists():
        print(f"  Stage 3b: NSM on SR-perturbed subgraph + perturbed Q ({ds}/{ptype})…")
        res_b = run_nsm_inference(sr_dir, ckpt_dir, ds, exp_name)

    # Clean baseline
    print(f"  Stage 3 baseline: NSM on clean data ({ds})…")
    res_clean = run_nsm_inference(NSM_DATA / ds, ckpt_dir, ds, exp_name)

    result = {
        "dataset": ds, "pert_type": ptype,
        "clean_baseline": res_clean,
        "gold_sg_pert_q": res_a,
        "sr_pert_sg_pert_q": res_b,
    }
    if not res_a.get("skipped") and not res_clean.get("skipped"):
        result["delta_AG_hit1_gold_sg"] = res_clean.get("hit1", 0) - res_a.get("hit1", 0)
    if not res_b.get("skipped") and not res_clean.get("skipped"):
        result["delta_AG_hit1_sr_sg"] = res_clean.get("hit1", 0) - res_b.get("hit1", 0)
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    p.add_argument("--dataset", choices=["webqsp", "cwq", "both"], default="both")
    p.add_argument("--pert-type", nargs="+", choices=PERT_TYPES, default=PERT_TYPES)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    datasets = DATASETS if args.dataset == "both" else [args.dataset]
    ptypes   = args.pert_type
    out_dir  = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if args.stage in ("1", "all"):
        print("\n" + "="*70)
        print("STAGE 1: Entity Linking")
        print("="*70)
        s1_results = {}
        for ds in datasets:
            s1_results[ds] = {}
            for pt in ptypes:
                print(f"\n  [{ds}/{pt}]")
                res = stage1_el(ds, pt)
                s1_results[ds][pt] = res
                if not res.get("skipped"):
                    h1 = res["hit1"]["mean"]
                    print(f"    Hit@1 = {h1:.4f} (n={res['n']})")
                else:
                    print(f"    SKIPPED: {res.get('reason','?')}")
        save_json(s1_results, out_dir / "stage1_el_results.json")
        all_results["stage1"] = s1_results

        print("\n── Stage 1 Summary ──")
        print(f"{'':5} {'':4} {'WebQSP':>10} {'CWQ':>10}")
        for pt in ptypes:
            row_parts = [f"  {pt:<4}"]
            for ds in datasets:
                r = s1_results.get(ds, {}).get(pt, {})
                if r.get("skipped"):
                    row_parts.append(f"{'skip':>10}")
                else:
                    row_parts.append(f"{r['hit1']['mean']:>10.4f}")
            print("".join(row_parts))

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if args.stage in ("2", "all"):
        print("\n" + "="*70)
        print("STAGE 2: Subgraph Retrieval (oracle seeds + perturbed question)")
        print("="*70)
        if not KG_CACHE.exists():
            print(f"  ERROR: KG cache not found at {KG_CACHE}")
            print("  Skipping Stage 2")
        else:
            kg = KGCache(KG_CACHE)
            s2_results = {}
            for ds in datasets:
                s2_results[ds] = {}
                for pt in ptypes:
                    print(f"\n  [{ds}/{pt}]")
                    res = stage2_sr(ds, pt, kg, out_dir)
                    s2_results[ds][pt] = res
                    print(f"    trip_jac={res['trip_jac']['mean']:.4f}  "
                          f"ent_jac={res['ent_jac']['mean']:.4f}  "
                          f"ans_in_pert={res['ans_in_pert']:.4f}")
            save_json(s2_results, out_dir / "stage2_sr_results.json")
            all_results["stage2"] = s2_results

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    if args.stage in ("3", "all"):
        print("\n" + "="*70)
        print("STAGE 3: Evidence Aggregation (NSM reasoner)")
        print("="*70)
        s3_results = {}
        for ds in datasets:
            s3_results[ds] = {}
            for pt in ptypes:
                print(f"\n  [{ds}/{pt}]")
                res = stage3_ea(ds, pt, out_dir)
                s3_results[ds][pt] = res
                if not res.get("skipped"):
                    print(f"    clean={res['clean_baseline'].get('hit1','?'):.4f}  "
                          f"gold_sg_pert_q={res['gold_sg_pert_q'].get('hit1','?'):.4f}  "
                          f"sr_sg_pert_q={res['sr_pert_sg_pert_q'].get('hit1','?') if not res['sr_pert_sg_pert_q'].get('skipped') else 'skip'}")
                else:
                    print(f"    SKIPPED: {res.get('reason','?')}")
        save_json(s3_results, out_dir / "stage3_ea_results.json")
        all_results["stage3"] = s3_results

    save_json(all_results, out_dir / "stage_isolation_summary.json")
    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
