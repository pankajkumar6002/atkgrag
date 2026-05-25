#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
exp_nsm_isolation.py  (EXP-3 from review.md)
==============================================
NSM Decoder Isolation Experiment.

Review concern (B6): "GNN decoder robust" claim rests on ReaRev only.
This script runs NSM's GNN on its own clean subgraph with perturbed CR question
text, measuring NSM Hit@1 (entity-name internal metric) to test whether decoder
robustness generalises beyond ReaRev.

Design mirrors Table 5 (tab:rearev_hits) but for NSM instead of ReaRev:
  - Input: perturbed question text (CR=P4, RS=P7)
  - Subgraph: CLEAN NSM subgraph from original (unperturbed) run
  - Metric: GNN-internal entity-name Hit@1 (same as Table 5 S2/S3 rows)

Expected finding:
  If NSM Hit@1 stays near its clean baseline (~50% CWQ), this confirms that GNN
  instruction decoders are structurally robust to compositional/relational
  perturbation when the subgraph is intact — not just ReaRev, but NSM too.
  The bottleneck truly is subgraph retrieval, not reasoning.

NSM checkpoints:
  CWQ:    tmlr/Repo/EPR-KGQA/NSM_H/checkpoint/pretrain/CWQ_nsm-h1.ckpt
  WebQSP: tmlr/Repo/EPR-KGQA/NSM_H/checkpoint/pretrain/WebQSP_nsm-h1.ckpt

NSM dataset format: EPR-KGQA NSM_H format (see datasets/ in NSM_H repo).
Clean NSM subgraphs: from tmlr/results/e1_cwq_gnn_rag/full_nsm_gem/ (p4 clean run).

Usage:
  # 200-question CWQ CR sample:
  python exp_nsm_isolation.py --dataset cwq --ptype p4 --sample 200

  # Both datasets, CR and RS:
  python exp_nsm_isolation.py --dataset both --ptype p4 p7 --sample 200

  # Dry run to check setup:
  python exp_nsm_isolation.py --dataset cwq --ptype p4 --sample 10 --dry-run

Output:
  results/nsm_isolation_{ds}_{ptype}.json
  Columns: Hit@1_clean, Hit@1_pert, Ins.cos. (if computable)

Paper target: Add NSM row to Table 5 (tab:rearev_hits).
  "NSM (200-q sample) & [Hit@1] & [cos] & [Hit@1] & [cos] \\\\"
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = REPO_ROOT
# SubgraphRetrievalKBQA NSM — the working NSM implementation (used by stage_isolation_nsm.py)
SR_REPO     = REPO_ROOT / "tmlr" / "Repo" / "SubgraphRetrievalKBQA" / "src"
NSM_GNN     = SR_REPO / "nsm_reader"          # sys.path entry for NSM module
NSM_DATA    = NSM_GNN / "data"                 # test_simple.json lives here
NSM_CKPT    = REPO_ROOT / "tmlr" / "models" / "nsm" # cwq/ and webqsp/ subdirs
PERT_DIR    = REPO_ROOT / "tmlr" / "pert" / "dataset"
GNN_DATA    = REPO_ROOT / "tmlr" / "Repo" / "GNN-RAG" / "data"
RESULTS     = REPO_ROOT / "tmlr" / "results"
NEW_EXP     = REPO_ROOT / "new_exp"

# Best Hit@1 checkpoint per dataset
CKPT = {
    "cwq":    NSM_CKPT / "cwq"    / "cwq_nsm-h1.ckpt",
    "webqsp": NSM_CKPT / "webqsp" / "webqsp_nsm-h1.ckpt",
}

# NSM vocab/data directory per dataset (for entities.txt, vocab_new.txt, etc.)
NSM_DATASET = {
    "cwq":    NSM_DATA / "cwq",
    "webqsp": NSM_DATA / "webqsp",
}

# Clean NSM subgraph: use SubgraphRetrievalKBQA's own test_simple.json (gold subgraph)
CLEAN_NSM_SUBGRAPH = {
    "cwq":    NSM_DATA / "cwq",
    "webqsp": NSM_DATA / "webqsp",
}

# Perturbation dataset files
PERT_FILE = {
    "cwq": {
        "p4": PERT_DIR / "cwq_p4.json",
        "p7": PERT_DIR / "cwq_p7.json",
    },
    "webqsp": {
        "p4": PERT_DIR / "webqsp_p4.json",
        "p7": PERT_DIR / "webqsp_p7.json",
    },
}

PTYPE_LABELS = {"p4": "CR — Compositional Restructuring", "p7": "RS — Relation Synonym"}


# ══════════════════════════════════════════════════════════════════════════════
# NSM inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_pert_questions(ds: str, ptype: str) -> Dict[str, str]:
    """Returns {qid: perturbed_question_text}."""
    fpath = PERT_FILE[ds][ptype]
    if not fpath.exists():
        raise FileNotFoundError(f"Pert file not found: {fpath}")
    with open(fpath) as f:
        records = json.load(f)
    return {r["id"]: r["perturbed_question"] for r in records
            if r.get("perturbed_question")}


def load_clean_subgraph_jsonl(ds: str, ptype: str) -> Dict[str, Dict]:
    """Load clean NSM test records from SubgraphRetrievalKBQA's test_simple.json.

    These are the gold-subgraph records used by stage_isolation_nsm.py Stage 3a:
    original (unperturbed) question + clean NSM subgraph. We substitute the
    question field with the perturbed question in run_nsm_on_records().
    """
    sg_file = CLEAN_NSM_SUBGRAPH[ds] / "test_simple.json"
    if not sg_file.exists():
        raise FileNotFoundError(
            f"NSM clean test file not found: {sg_file}\n"
            f"  Expected: SubgraphRetrievalKBQA/src/nsm_reader/data/{ds}/test_simple.json"
        )
    records = {}
    with open(sg_file) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["id"]] = r
    return records


def check_nsm_setup(ds: str) -> bool:
    """Verify NSM checkpoint and dataset files exist, and NSM module is importable."""
    ok = True
    ckpt = CKPT[ds]
    if not ckpt.exists():
        print(f"  [ERROR] NSM checkpoint missing: {ckpt}")
        ok = False
    nsm_ds = NSM_DATASET[ds]
    if not nsm_ds.exists():
        print(f"  [ERROR] NSM dataset directory missing: {nsm_ds}")
        ok = False
    if str(NSM_GNN) not in sys.path:
        sys.path.insert(0, str(NSM_GNN))
    try:
        from NSM.train.trainer_nsm import Trainer_KBQA  # noqa: F401
        print("  [OK] NSM Trainer_KBQA importable")
    except ImportError as e:
        print(f"  [ERROR] Cannot import NSM: {e}")
        print(f"  Ensure {NSM_GNN} is in PYTHONPATH and NSM deps installed")
        ok = False
    return ok


def run_nsm_on_records(
    records: List[Dict],
    ds: str,
    perturbed_qs: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> Dict[str, Dict]:
    """
    Run NSM inference on a list of records using SubgraphRetrievalKBQA NSM.
    If perturbed_qs is provided, replace question text with perturbed versions.
    Returns {qid: {"hit1": float, "f1": float}}.

    Mirrors stage_isolation_nsm.py:run_nsm_inference() — uses the same args dict,
    same data format (test_simple.json), same checkpoint glob, same Trainer_KBQA API.
    """
    if str(NSM_GNN) not in sys.path:
        sys.path.insert(0, str(NSM_GNN))

    if dry_run:
        print(f"    [DRY RUN] would run NSM on {len(records)} records")
        return {r["id"]: {"hit1": 0.5, "f1": 0.4} for r in records}

    try:
        from NSM.train.trainer_nsm import Trainer_KBQA
        import torch
        import logging
    except ImportError as e:
        print(f"  [ERROR] NSM import failed: {e}")
        return {}

    ckpt_path  = CKPT[ds]
    ckpt_dir   = ckpt_path.parent
    nsm_ds_dir = NSM_DATASET[ds]

    # Substitute question text when running perturbed pass
    inference_records = []
    for r in records:
        rec = copy.deepcopy(r)
        qid = rec.get("id", "")
        if perturbed_qs and qid in perturbed_qs:
            rec["question"] = perturbed_qs[qid]
        inference_records.append(rec)

    tmp_dir = Path(tempfile.mkdtemp(prefix="nsm_isolation_"))
    try:
        # Write records as test_simple.json (NSM SubgraphRetrievalKBQA format)
        tmp_test = tmp_dir / "test_simple.json"
        with open(tmp_test, "w") as f:
            for rec in inference_records:
                f.write(json.dumps(rec) + "\n")

        # Symlink vocab/embedding files from the dataset directory
        for fname in ["entities.txt", "relations.txt", "vocab_new.txt",
                      "word_emb_300d.npy", "dev_simple.json", "train_simple.json"]:
            src = nsm_ds_dir / fname
            dst = tmp_dir / fname
            if src.exists() and not dst.exists():
                os.symlink(src, dst)

        # NSM trainer expects args as a dict (same structure as stage_isolation_nsm.py)
        exp_name = "nsm_isolation"
        args = {
            "name":             ds,
            "data_folder":      str(tmp_dir) + "/",
            "checkpoint_dir":   str(ckpt_dir) + "/",
            "model_name":       "gnn",
            "batch_size":       20,
            "test_batch_size":  40,
            "num_step":         3,
            "entity_dim":       50,
            "word_dim":         300,
            "kg_dim":           100,
            "kge_dim":          100,
            "eval_every":       5,
            "encode_type":      True,
            "experiment_name":  exp_name,
            "eps":              0.95 if ds == "webqsp" else 0.5,
            "num_epoch":        200,
            "use_self_loop":    True,
            "lr":               5e-4,
            "word_emb_file":    "word_emb_300d.npy",
            "loss_type":        "kl",
            "reason_kb":        True,
            "is_eval":          True,
            "load_experiment":  ckpt_path.name,
            "load_ckpt_file":   None,
            "use_cuda":         torch.cuda.is_available(),
            "q_type":           "seq",
            "num_layer":        1,
            "mode":             "teacher",
            "entropy_weight":   0.0,
            "use_label":        False,
            "tree_soft":        False,
            "filter_label":     False,
            "share_embedding":  False,
            "share_instruction": False,
            "encoder_type":     "lstm",
            "lambda_label":     0.1,
            "lambda_auto":      0.01,
            "label_f1":         0.5,
            "label_file":       None,
            "share_encoder":    False,
            "use_inverse_relation": False,
            "train_KL":         False,
            "filter_sub":       False,
            "decay_rate":       0.0,
            "gradient_clip":    1.0,
            "lr_schedule":      False,
            "label_smooth":     0.1,
            "fact_drop":        0,
            "fact_scale":       3,
            "entity2id":        "entities.txt",
            "relation2id":      "relations.txt",
            "word2id":          "vocab_new.txt",
            "entity_emb_file":  None,
            "entity_kge_file":  None,
            "relation_emb_file": None,
            "relation_kge_file": None,
            "rel_word_ids":     "rel_word_idx.npy",
            "pretrained_entity_kge_file": "entity_emb_100d.npy",
            "char2id":          "chars.txt",
            "lstm_dropout":     0.3,
            "linear_dropout":   0.2,
            "seed":             19960626,
            "log_level":        "info",
        }

        logger = logging.getLogger(f"nsm_{exp_name}")
        logger.setLevel(logging.WARNING)
        trainer = Trainer_KBQA(args=args, logger=logger)
        f1, h1 = trainer.evaluate(trainer.test_data, args["test_batch_size"],
                                  write_info=False)

        # Return aggregate metrics (positional matching not needed — trainer returns scalars)
        agg = {"hit1": float(h1), "f1": float(f1)}
        return {r["id"]: agg for r in inference_records}

    finally:
        try:
            del trainer
        except NameError:
            pass
        import gc
        gc.collect()
        if sys.modules.get("torch"):
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main isolation experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_nsm_isolation(ds: str, ptype: str, sample_n: int,
                      dry_run: bool = False) -> Dict[str, Any]:
    """Run NSM decoder isolation for one (dataset, ptype) cell."""
    print(f"\n{'='*70}")
    print(f"NSM Isolation: {ds.upper()} {ptype.upper()} ({PTYPE_LABELS.get(ptype, ptype)})")
    print(f"Sample: {sample_n} questions | Clean NSM subgraph + Perturbed question")
    print(f"{'='*70}")

    if not check_nsm_setup(ds):
        print("  [SKIP] NSM setup check failed")
        return {}

    clean_records = load_clean_subgraph_jsonl(ds, ptype)
    print(f"  Loaded {len(clean_records)} clean NSM subgraph records")

    pert_qs = load_pert_questions(ds, ptype)
    print(f"  Loaded {len(pert_qs)} perturbed questions for {ptype}")

    # Intersect: questions with both clean subgraph and perturbed question
    common_ids = sorted(set(clean_records.keys()) & set(pert_qs.keys()))
    print(f"  Intersection: {len(common_ids)} questions")

    if not common_ids:
        print("  [ERROR] No overlapping question IDs between subgraph and perturbation file")
        return {}

    random.seed(42)
    sample_ids = random.sample(common_ids, min(sample_n, len(common_ids)))
    print(f"  Using {len(sample_ids)} randomly sampled questions")

    sample_records = [clean_records[qid] for qid in sample_ids]

    # Run 1: Clean baseline (original question, clean NSM subgraph)
    print(f"\n  Run 1: NSM on CLEAN question + clean NSM subgraph...")
    t0 = time.time()
    clean_preds = run_nsm_on_records(sample_records, ds, perturbed_qs=None, dry_run=dry_run)
    print(f"  Done in {(time.time()-t0)/60:.1f}min")

    # Run 2: Perturbed question, same clean NSM subgraph
    print(f"\n  Run 2: NSM on PERTURBED question ({ptype}) + clean NSM subgraph...")
    t0 = time.time()
    pert_preds = run_nsm_on_records(sample_records, ds, perturbed_qs=pert_qs, dry_run=dry_run)
    print(f"  Done in {(time.time()-t0)/60:.1f}min")

    # Aggregate metrics
    hit1_clean = [clean_preds.get(qid, {}).get("hit1", 0.0) for qid in sample_ids
                  if qid in clean_preds]
    hit1_pert  = [pert_preds.get(qid, {}).get("hit1", 0.0) for qid in sample_ids
                  if qid in pert_preds]

    n = len(hit1_clean)
    if n == 0:
        print("  [WARN] No results to aggregate")
        return {}

    results = {
        "dataset": ds, "ptype": ptype,
        "n": n,
        "sample_n_requested": sample_n,
        "hit1_clean": round(100 * np.mean(hit1_clean), 2),
        "hit1_pert":  round(100 * np.mean(hit1_pert), 2),
        "delta_hit1": round(100 * (np.mean(hit1_clean) - np.mean(hit1_pert)), 2),
    }

    clean_baseline = {"cwq": 52.9, "webqsp": 74.3}[ds]   # ReaRev clean EM for reference
    nsm_clean_baseline = {"cwq": 50.0, "webqsp": 68.0}[ds]  # approximate NSM clean baseline

    print(f"\n  Results:")
    print(f"    Hit@1 (clean question):    {results['hit1_clean']:.1f}%")
    print(f"    Hit@1 (pert question {ptype}): {results['hit1_pert']:.1f}%")
    print(f"    Drop:                      {results['delta_hit1']:.1f}pp")
    print(f"\n  Interpretation:")
    if results["hit1_pert"] >= 0.9 * nsm_clean_baseline:
        print(f"    NSM Hit@1 ({results['hit1_pert']:.1f}%) stays near clean baseline (~{nsm_clean_baseline}%)")
        print(f"    → NSM decoder is robust to {ptype.upper()} when subgraph is intact.")
        print(f"    → Generalises the ReaRev finding (Table 5): decoder is NOT the bottleneck.")
    else:
        print(f"    NSM Hit@1 drops to {results['hit1_pert']:.1f}% under {ptype.upper()}")
        print(f"    → NSM decoder is more sensitive than ReaRev to this perturbation type.")

    print(f"\n  LaTeX row for Table 5:")
    print(f"    NSM ({n}q sample) & {results['hit1_pert']:.1f} & — & {results['hit1_pert']:.1f} & — \\\\")
    print(f"    (CWQ and WebQSP columns; Ins.cos. requires extracting NSM instruction vectors)")

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="NSM decoder isolation experiment (EXP-3)")
    p.add_argument("--dataset", default="cwq", choices=["cwq", "webqsp", "both"])
    p.add_argument("--ptype", nargs="+", default=["p4"],
                   choices=["p4", "p7"],
                   help="CR (p4) and/or RS (p7) perturbation types")
    p.add_argument("--sample", type=int, default=200,
                   help="Number of questions to sample (default: 200)")
    p.add_argument("--dry-run", action="store_true",
                   help="Check setup and show what would run without GPU inference")
    p.add_argument("--out-dir", type=Path, default=RESULTS,
                   help="Output directory for JSON results")
    args = p.parse_args()

    datasets = ["cwq", "webqsp"] if args.dataset == "both" else [args.dataset]

    for ds in datasets:
        for ptype in args.ptype:
            results = run_nsm_isolation(ds, ptype, args.sample, args.dry_run)
            if results and not args.dry_run:
                out = args.out_dir / f"nsm_isolation_{ds}_{ptype}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\n  Saved: {out}")

    print("\n[PAPER UPDATE]")
    print("  After collecting results, update:")
    print("  1. sections/exp.tex Table 5 (tab:rearev_hits): add NSM row")
    print("  2. sections/disc.tex Limitations: change 'scope to ReaRev' → 'ReaRev and NSM'")
    print("  3. Hit@1 metric note: NSM uses entity-name Hit@1, same as ReaRev")


if __name__ == "__main__":
    main()
