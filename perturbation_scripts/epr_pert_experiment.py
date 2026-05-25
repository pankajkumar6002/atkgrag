#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
epr_pert_experiment.py — EPR-KGQA on perturbed questions (p1 / p4 / p7)

Pipeline (mirrors the original EPR-KGQA README, inference-only):
  0. prep  — build perturbed test JSON + sr_info in EPR format; write pert config YAML
  1. ap    — atomic pattern retrieval (biencoder inference, pretrained model)
  2. ep    — evidence pattern construction + ranking (BERT ranker, predict only)
  3. subg  — subgraph extraction via Virtuoso (port 3002)
  4. nsm   — NSM reasoning + hit@1 evaluation

Usage:
  # Full run CWQ p4:
  python new_exp/scripts/epr_pert_experiment.py --dataset cwq --ptype p4 --stage all

  # All pert types, both datasets:
  python new_exp/scripts/epr_pert_experiment.py --dataset cwq webqsp --ptype p1 p4 p7 --stage all

  # Individual stages:
  python new_exp/scripts/epr_pert_experiment.py --dataset cwq --ptype p4 --stage prep
  python new_exp/scripts/epr_pert_experiment.py --dataset cwq --ptype p4 --stage ap
  python new_exp/scripts/epr_pert_experiment.py --dataset cwq --ptype p4 --stage ep
  python new_exp/scripts/epr_pert_experiment.py --dataset cwq --ptype p4 --stage subg
  python new_exp/scripts/epr_pert_experiment.py --dataset cwq --ptype p4 --stage nsm

Required checkpoints (download from Google Drive links in EPR-KGQA/README.md):
  data/cache/rel_conn_fb.jsonl
  data/cache/rr_aps_fb.json
  data/cache/rr_aps_forward_reverse_dict.json
  data/cache/rr_aps_tag_dict.json
  data/cache/CWQ/cached_paths.jsonl
  data/cache/WebQSP/cached_paths.jsonl
  data/CWQ/ap_retrieval/model/CWQ_ep_5.pt        ← biencoder CWQ (Google Drive)
  data/CWQ/ap_retrieval/model/CWQ_ep_5/flat.index ← faiss index (built after download)
  data/CWQ/ep_retrieval/model/epoch_7/            ← BERT ranker CWQ
  data/CWQ/ep_retrieval/CWQ_entity_name_dict.json
  data/WebQSP/ap_retrieval/model/WebQSP_ep_6.pt
  data/WebQSP/ep_retrieval/model/epoch_6/
  data/WebQSP/ep_retrieval/WebQSP_entity_name_dict.json
  NSM_H/datasets/CWQ_EPR/                         ← NSM data (entities, relations, vocab)
  NSM_H/datasets/WebQSP_EPR/
  NSM_H/checkpoint/pretrain/CWQ_nsm-h1.ckpt
  NSM_H/checkpoint/pretrain/WebQSP_nsm-h1.ckpt    ← (name may differ; check run_webqsp.sh)
  NSM_H/word_emb_300d.npy

ODBC setup (needed for biencoder ER-AP retrieval via FreebaseODBC):
  Add to ~/.odbc.ini:
    [freebase]
    Driver=Virtuoso
    Host=127.0.0.1
    Port=1111     # Virtuoso ISQL port (check virtuoso.ini for ServerPort)
    UID=dba
    PWD=dba
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = REPO_ROOT
EPR_ROOT    = REPO_ROOT / "tmlr/Repo/EPR-KGQA"
NEW_EXP     = REPO_ROOT / "new_exp"
PERT_DIR    = REPO_ROOT / "data" / "perturbations"
EPR_PYTHON  = os.environ.get("EPR_PYTHON", "python")

SPARQL_ENDPOINT = "http://127.0.0.1:3001/sparql"   # Freebase Virtuoso HTTP port
ODBC_DSN        = "freebase"  # DSN in ~/.odbc.ini (Driver=virtodbc.so, Address=localhost:13001)

# Biencoder epoch and EP ranker epoch (from README)
BIENCODER_EPOCH = {"cwq": 5, "webqsp": 6}
EP_RANKER_EPOCH = {"cwq": 7, "webqsp": 6}
EP_TOPK         = 100

# NSM checkpoint names (adjust if downloaded file has a different name)
NSM_CKPT = {
    "cwq":    "CWQ_nsm-h1.ckpt",
    "webqsp": "WebQSP_nsm-h1.ckpt",
}

# Mapping from our dataset names to EPR tag names
DATASET_TAG = {"cwq": "CWQ", "webqsp": "WebQSP"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: List[str], cwd: Path, env: dict = None, check: bool = True) -> int:
    merged_env = {**os.environ, **(env or {})}
    merged_env["SPARQL_ENDPOINT"] = SPARQL_ENDPOINT
    merged_env["ODBC_DSN"]        = ODBC_DSN
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), env=merged_env)
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def epr_python(script: str, extra_args: List[str], config_path: Path, env: dict = None) -> int:
    return run(
        [EPR_PYTHON, script] + extra_args + ["--config", str(config_path)],
        cwd=EPR_ROOT,
        env=env,
    )


# ── Stage 0: Prep — build perturbed test files + EPR configs ──────────────────

def load_pert_dataset(dataset: str, ptype: str) -> List[Dict]:
    path = PERT_DIR / f"{dataset}_{ptype}.json"
    with open(path) as f:
        return json.load(f)


def load_original_sr_info(dataset: str) -> Dict[str, Dict]:
    """Load CWQ_full_with_int_id.jsonl keyed by ID."""
    path = EPR_ROOT / "data/dataset/CWQ/CWQ_full_with_int_id.jsonl"
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                result[obj["id"]] = obj
    return result


def create_cwq_pert_test_json(pert_records: List[Dict], out_path: Path) -> None:
    """Write perturbed CWQ test JSON in EPR format (replaces question field)."""
    original_test = json.load(open(EPR_ROOT / "data/dataset/CWQ/ComplexWebQuestions_test.json"))
    orig_by_id = {r["ID"]: r for r in original_test}
    pert_by_id = {r["id"]: r for r in pert_records}

    result = []
    for orig in original_test:
        qid = orig["ID"]
        if qid in pert_by_id:
            rec = dict(orig)
            rec["question"] = pert_by_id[qid]["perturbed_question"]
            result.append(rec)
        else:
            result.append(orig)  # keep original if not perturbed (e.g. p1 partial coverage)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"  [prep] wrote {len(result)} records → {out_path}")


def create_cwq_pert_sr_info(pert_records: List[Dict], ptype: str, out_path: Path) -> None:
    """
    Write perturbed CWQ_full_with_int_id.jsonl.
    For p1: update topic entities from gold_entity_map keys.
    For p4/p7: topic entities unchanged; only answers updated from pert answers field.
    """
    original_sr = load_original_sr_info("cwq")
    pert_by_id = {r["id"]: r for r in pert_records}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for qid, orig in original_sr.items():
            if qid not in pert_by_id:
                f.write(json.dumps(orig) + "\n")
                continue
            pert = pert_by_id[qid]
            rec = dict(orig)
            if ptype == "p1":
                # Update topic entities from gold_entity_map (new substituted entity MIDs)
                new_entity_mids = list(pert.get("gold_entity_map", {}).keys())
                rec["entities"] = [{"kb_id": mid, "text": name}
                                   for mid, name in pert.get("gold_entity_map", {}).items()]
            # Update answers
            new_answers = pert.get("answers", [])
            if new_answers:
                rec["answers"] = [{"kb_id": a, "text": ""} if isinstance(a, str) else a
                                  for a in new_answers]
            f.write(json.dumps(rec) + "\n")
    print(f"  [prep] wrote sr_info → {out_path}")


def create_webqsp_pert_test_jsonl(pert_records: List[Dict], ptype: str, out_path: Path) -> None:
    """Write perturbed WebQSP test_simple.jsonl in EPR format."""
    original_jsonl_path = EPR_ROOT / "data/dataset/WebQSP/test_simple.jsonl"
    orig_by_id: Dict[str, Dict] = {}
    with open(original_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                orig_by_id[obj["id"]] = obj

    pert_by_id = {r["id"]: r for r in pert_records}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for qid, orig in orig_by_id.items():
            if qid not in pert_by_id:
                f.write(json.dumps(orig) + "\n")
                continue
            pert = pert_by_id[qid]
            rec = dict(orig)
            rec["question"] = pert["perturbed_question"]
            if ptype == "p1":
                # Update entities from gold_entity_map keys
                rec["entities"] = list(pert.get("gold_entity_map", {}).keys())
            # Update answers
            new_answers = pert.get("answers", [])
            if new_answers:
                rec["answers"] = [{"kb_id": a, "text": ""} if isinstance(a, str) else a
                                  for a in new_answers]
            f.write(json.dumps(rec) + "\n")
    print(f"  [prep] wrote WebQSP pert test jsonl → {out_path}")


def create_pert_config(dataset: str, ptype: str,
                       test_file: Path, sr_file: Path) -> Path:
    """Write a pert-specific config YAML for EPR-KGQA."""
    tag = DATASET_TAG[dataset]
    pert_tag = f"{tag}_{ptype}"  # e.g. CWQ_p4
    cache_path = f"data/cache/{tag}/cached_paths.jsonl"

    if dataset == "cwq":
        config = {
            "dataset": {
                "tag": tag,
                "train": f"data/dataset/CWQ/ComplexWebQuestions_train.json",
                "dev":   f"data/dataset/CWQ/ComplexWebQuestions_dev.json",
                "test":  str(test_file.relative_to(EPR_ROOT)),
                "SR_cache_all": str(sr_file.relative_to(EPR_ROOT)) if sr_file else f"data/dataset/CWQ/CWQ_full_with_int_id.jsonl",
            },
            "cache": {"searched_path": cache_path},
            "log": {"process_log": f"data/log/{pert_tag}_process.log", "level": "DEBUG"},
            "global_var": {"query_time_limit": 1, "max_hop_limit": 2,
                           "bulk_path_search": False, "worker_num": 4},
            "ap_retrieval": {"work_dir": f"data/{pert_tag}/ap_retrieval/", "ap_topk": EP_TOPK},
            "ep_retrieval": {"work_dir": f"data/{pert_tag}/ep_retrieval/",
                             "max_combine_rels": 5, "sample_size": 64},
            "subgraph_extraction": {"work_dir": f"data/{pert_tag}/subgraph_extraction/", "ep_topk": 1},
            "NSM_reasoning": {"work_dir": f"NSM_H/datasets/{pert_tag}_EPR/"},
        }
    else:  # webqsp
        config = {
            "dataset": {
                "tag": tag,
                "train": "data/dataset/WebQSP/train_simple.jsonl",
                "dev":   "data/dataset/WebQSP/dev_simple.jsonl",
                "test":  str(test_file.relative_to(EPR_ROOT)),
                "SR_cache_all": "data/dataset/WebQSP/test_simple.jsonl",
            },
            "cache": {"searched_path": cache_path},
            "log": {"process_log": f"data/log/{pert_tag}_process.log", "level": "DEBUG"},
            "global_var": {"query_time_limit": 1, "max_hop_limit": 2,
                           "bulk_path_search": False, "worker_num": 4},
            "ap_retrieval": {"work_dir": f"data/{pert_tag}/ap_retrieval/", "ap_topk": EP_TOPK},
            "ep_retrieval": {"work_dir": f"data/{pert_tag}/ep_retrieval/",
                             "max_combine_rels": 5, "sample_size": 100},
            "subgraph_extraction": {"work_dir": f"data/{pert_tag}/subgraph_extraction/", "ep_topk": 1},
            "NSM_reasoning": {"work_dir": f"NSM_H/datasets/{pert_tag}_EPR/"},
        }

    import yaml
    config_path = EPR_ROOT / f"config_{pert_tag}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  [prep] wrote config → {config_path}")
    return config_path


def stage_prep(dataset: str, ptype: str):
    print(f"\n=== PREP: {dataset} {ptype} ===")
    tag = DATASET_TAG[dataset]
    pert_tag = f"{tag}_{ptype}"
    pert_records = load_pert_dataset(dataset, ptype)
    print(f"  {len(pert_records)} perturbed records loaded")

    pert_data_dir = EPR_ROOT / "data" / "pert"
    pert_data_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "cwq":
        test_file = pert_data_dir / f"{pert_tag}_test.json"
        sr_file   = pert_data_dir / f"{pert_tag}_sr_info.jsonl"
        if not test_file.exists():
            create_cwq_pert_test_json(pert_records, test_file)
        else:
            print(f"  [prep] {test_file.name} already exists — skipping")
        if not sr_file.exists():
            create_cwq_pert_sr_info(pert_records, ptype, sr_file)
        else:
            print(f"  [prep] {sr_file.name} already exists — skipping")
    else:
        test_file = pert_data_dir / f"{pert_tag}_test.jsonl"
        sr_file   = None
        if not test_file.exists():
            create_webqsp_pert_test_jsonl(pert_records, ptype, test_file)
        else:
            print(f"  [prep] {test_file.name} already exists — skipping")

    config_path = EPR_ROOT / f"config_{pert_tag}.yaml"
    if not config_path.exists():
        create_pert_config(dataset, ptype, test_file, sr_file)
    else:
        print(f"  [prep] {config_path.name} already exists — skipping")

    return config_path, test_file, sr_file


# ── Stage 1: AP Retrieval (biencoder inference) ────────────────────────────────

def _data_tag(tag: str, ptype: str) -> str:
    """Returns the data directory tag: plain tag for clean, tag_ptype for perturbations."""
    return tag if ptype == "clean" else f"{tag}_{ptype}"


def stage_ap(dataset: str, ptype: str, config_path: Path, test_file: Path):
    print(f"\n=== AP RETRIEVAL: {dataset} {ptype} ===")
    tag = DATASET_TAG[dataset]
    pert_tag = _data_tag(tag, ptype)
    epoch = BIENCODER_EPOCH[dataset]

    out_dir = EPR_ROOT / "data" / pert_tag / "ap_retrieval"
    out_file = out_dir / f"{tag}_test_retrieved_ap.json"
    if out_file.exists():
        print(f"  AP retrieval output exists — skipping: {out_file}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (EPR_ROOT / "data" / "log").mkdir(parents=True, exist_ok=True)

    # The biencoder uses the pretrained model from data/{tag}/ap_retrieval/model/
    # and the FAISS index built during clean training.
    clean_model_base = str(EPR_ROOT / "data" / tag / "ap_retrieval" / "model")
    env = {
        "SPARQL_ENDPOINT": SPARQL_ENDPOINT,
        # Point to the clean model/index dir so perturbed runs skip re-encoding APs
        "EPR_BIENCODER_MODEL_DIR": clean_model_base,
    }
    if ptype == "clean":
        extra_args = []  # run dev + test + train for clean baseline
    else:
        env["EPR_PERT_TEST_FILE"] = str(test_file)
        extra_args = ["--splits", "test"]  # only test split needed for pert runs
    epr_python(
        "atomic_pattern_retrieval/biencoder/biencoder_inference.py",
        extra_args=extra_args,
        config_path=config_path,
        env=env,
    )


# ── Stage 2: EP Construction + Ranking ────────────────────────────────────────

def stage_ep(dataset: str, ptype: str, config_path: Path):
    print(f"\n=== EP CONSTRUCTION + RANKING: {dataset} {ptype} ===")
    tag = DATASET_TAG[dataset]
    pert_tag = _data_tag(tag, ptype)
    epoch = EP_RANKER_EPOCH[dataset]

    pert_splits = [] if ptype == "clean" else ["--splits", "test"]

    # For clean runs, train+dev candidate EPs are also needed by subgraph_extraction.
    candi_splits_needed = ["test"] if ptype != "clean" else ["test", "dev", "train"]
    ep_dir = EPR_ROOT / "data" / pert_tag / "ep_retrieval"
    missing_candi = [
        s for s in candi_splits_needed
        if not (ep_dir / f"{tag}_{s}_top{EP_TOPK}_ap_candi_eps.json").exists()
    ]
    if missing_candi:
        print(f"  Generating candidate EPs (splits: {candi_splits_needed})...")
        epr_python(
            "evidence_pattern_retrieval/generate_candidate_eps.py",
            extra_args=pert_splits,
            config_path=config_path,
        )
    else:
        print(f"  Candidate EPs exist for all splits — skipping")

    ep_pert_dir = EPR_ROOT / "data" / pert_tag / "ep_retrieval"
    ep_pert_dir.mkdir(parents=True, exist_ok=True)
    model_dir = EPR_ROOT / "data" / tag / "ep_retrieval" / "model" / f"epoch_{epoch}"
    sample_size = 64 if dataset == "cwq" else 100

    # For clean runs, dev+train ranked EPs are required by subgraph_extraction
    # (generate_subgraph_for_training builds NSM vocab from train+dev subgraphs).
    splits_to_rank = ["test"] if ptype != "clean" else ["test", "dev", "train"]
    missing_ranked = [
        s for s in splits_to_rank
        if not (ep_pert_dir / f"{tag}_{s}_top{EP_TOPK}_ap_ranked_ep.json").exists()
    ]

    if missing_ranked:
        print(f"  Generating EP ranking features (splits: {splits_to_rank})...")
        epr_python(
            "evidence_pattern_retrieval/generate_ep_ranking_data.py",
            extra_args=pert_splits,
            config_path=config_path,
            env={"SPARQL_ENDPOINT": SPARQL_ENDPOINT},
        )
        # Run ep_ranking.py for each missing split
        for split in missing_ranked:
            ranked_f = ep_pert_dir / f"{tag}_{split}_top{EP_TOPK}_ap_ranked_ep.json"
            if ranked_f.exists():
                print(f"  Ranked EPs exist for {split} — skipping")
                continue
            predict_file = str(ep_pert_dir / "training_data" / f"{tag}_{split}_top{EP_TOPK}_ap_candi_eps_feature_cache_for_prediction")
            print(f"  Running EP ranking prediction for {split}...")
            run(
                [EPR_PYTHON, "evidence_pattern_retrieval/ep_ranking.py",
                 "--dataset", tag,
                 "--model_type", "bert",
                 "--model_name_or_path", str(model_dir),
                 "--do_lower_case",
                 "--do_predict",
                 "--predict_file", predict_file,
                 "--overwrite_output_dir",
                 "--max_seq_length", "256",
                 "--output_dir", str(ep_pert_dir),
                 "--per_gpu_eval_batch_size", "16",
                 "--sample_size", str(sample_size),
                 "--topk", str(EP_TOPK)],
                cwd=EPR_ROOT,
            )
    else:
        print(f"  Ranked EPs exist for all splits — skipping EP ranking")

    # For clean runs, subgraph_extraction.py also sweeps topk=80 on test.
    # Run ep_ranking at topk=80 for test if missing.
    if ptype == "clean":
        ranked_80 = ep_pert_dir / f"{tag}_test_top80_ap_ranked_ep.json"
        if not ranked_80.exists():
            predict_file_80 = str(ep_pert_dir / "training_data" / f"{tag}_test_top80_ap_candi_eps_feature_cache_for_prediction")
            print(f"  Running EP ranking prediction for test at topk=80...")
            run(
                [EPR_PYTHON, "evidence_pattern_retrieval/ep_ranking.py",
                 "--dataset", tag,
                 "--model_type", "bert",
                 "--model_name_or_path", str(model_dir),
                 "--do_lower_case",
                 "--do_predict",
                 "--predict_file", predict_file_80,
                 "--overwrite_output_dir",
                 "--max_seq_length", "256",
                 "--output_dir", str(ep_pert_dir),
                 "--per_gpu_eval_batch_size", "16",
                 "--sample_size", str(sample_size),
                 "--topk", "80"],
                cwd=EPR_ROOT,
            )
        else:
            print(f"  Ranked EPs (top80 test) exist — skipping")


# ── Stage 3: Subgraph Extraction ───────────────────────────────────────────────

def stage_subg(dataset: str, ptype: str, config_path: Path):
    print(f"\n=== SUBGRAPH EXTRACTION: {dataset} {ptype} ===")
    tag = DATASET_TAG[dataset]
    pert_tag = _data_tag(tag, ptype)

    pert_splits = [] if ptype == "clean" else ["--splits", "test"]
    # Pert runs only have top100 ranked EPs; clean runs sweep [80, 100]
    topk_range = [] if ptype == "clean" else ["--test_topk_range", "100", "100"]

    subg_out = EPR_ROOT / "data" / pert_tag / "subgraph_extraction" / f"{tag}_test_top{EP_TOPK}_ap_top1_ep_instantiated_subg.json"
    if subg_out.exists():
        print(f"  Subgraph exists — skipping: {subg_out.name}")
    else:
        epr_python(
            "subgraph_extraction/subgraph_extraction.py",
            extra_args=pert_splits + topk_range,
            config_path=config_path,
            env={"SPARQL_ENDPOINT": SPARQL_ENDPOINT},
        )

    # Run convert_to_nsm_input.py — skip if test_simple.json already exists
    # freeze_relations: keep relations.txt fixed to match pretrained NSM checkpoint vocab
    nsm_test_simple = EPR_ROOT / "NSM_H" / "datasets" / f"{pert_tag}_EPR" / "test_simple.json"
    if nsm_test_simple.exists():
        print(f"  NSM test_simple.json exists — skipping convert_to_nsm_input.py")
    else:
        epr_python(
            "subgraph_extraction/convert_to_nsm_input.py",
            extra_args=pert_splits + topk_range + ["--freeze_relations"],
            config_path=config_path,
        )


# ── Stage 4: NSM Inference + Evaluation ──────────────────────────────────────

def stage_nsm(dataset: str, ptype: str, config_path: Path):
    print(f"\n=== NSM INFERENCE: {dataset} {ptype} ===")
    tag = DATASET_TAG[dataset]
    pert_tag = _data_tag(tag, ptype)

    ckpt_name = NSM_CKPT[dataset]
    nsm_data_dir = f"./datasets/{pert_tag}_EPR/"
    ckpt_dir = "checkpoint/pretrain/"

    # Check NSM data directory exists
    nsm_data_path = EPR_ROOT / "NSM_H" / "datasets" / f"{pert_tag}_EPR"
    if not nsm_data_path.exists():
        print(f"[ERROR] NSM data not found: {nsm_data_path}")
        print("  Run stage_subg first (convert_to_nsm_input.py creates this directory)")
        return

    # Skip if NSM log already has result
    nsm_log_file = EPR_ROOT / "NSM_H" / "checkpoint" / "pretrain" / f"{pert_tag}_nsm.log"
    if nsm_log_file.exists():
        with open(nsm_log_file) as _f:
            _log = _f.read()
        if "TEST F1" in _log:
            lines = [l for l in _log.splitlines() if "TEST F1" in l]
            print(f"  NSM already done — skipping. Result: {lines[-1].split('INFO - ')[-1]}")
            return

    # Ensure vocab_new.txt and word_emb_300d.npy are in the dataset dir
    import shutil
    for fname in ["vocab_new.txt", "word_emb_300d.npy"]:
        src = EPR_ROOT / "NSM_H" / fname
        dst = nsm_data_path / fname
        if not dst.exists() and src.exists():
            shutil.copy(src, dst)

    # Generate dep files from question text if missing
    # NSM's _prepare_dep() uses tokens from dep entries; empty dep → seq_len=0 → LSTM crash
    import re as _re
    def _tokenize(q):
        q = q.strip().lower()
        q = _re.sub("'s", " s", q)
        return [_re.sub(r'^[^a-z0-9]|[^a-z0-9]$', '', w) for w in q.split() if _re.sub(r'^[^a-z0-9]|[^a-z0-9]$', '', w)]

    for split in ["train", "dev", "test"]:
        dep_file = nsm_data_path / f"{split}.dep"
        if not dep_file.exists():
            src_json = nsm_data_path / f"{split}_simple.json"
            if src_json.exists():
                with open(src_json) as fin, open(dep_file, "w") as fout:
                    for line in fin:
                        d = json.loads(line)
                        q = d.get("question", "")
                        tokens = _tokenize(q)
                        fake_dep = [[w, "NN", "0", "root"] for w in tokens] if tokens else [["unk", "NN", "0", "root"]]
                        fout.write(json.dumps({"id": d["id"], "dep": fake_dep, "question": q}) + "\n")
                print(f"  Generated dep {dep_file.name} from question tokens")

    result_dir = NEW_EXP / "results" / "epr_kgqa" / f"{dataset}_{ptype}"
    result_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            EPR_PYTHON, "main_nsm.py",
            "--name", pert_tag,
            "--model_name", "gnn",
            "--data_folder", nsm_data_dir,
            "--checkpoint_dir", ckpt_dir,
            "--batch_size", "20",
            "--test_batch_size", "40",
            "--num_step", "4",
            "--entity_dim", "50",
            "--word_dim", "300",
            "--kg_dim", "100",
            "--kge_dim", "100",
            "--eval_every", "2",
            "--experiment_name", f"{pert_tag}_nsm",
            "--eps", "0.95",
            "--num_epoch", "30",
            "--use_self_loop",
            "--lr", "5e-4",
            "--q_type", "seq",
            "--word_emb_file", "word_emb_300d.npy",
            "--reason_kb",
            "--encode_type",
            "--loss_type", "kl",
            "--is_eval",
            "--load_experiment", ckpt_name,
        ],
        cwd=EPR_ROOT / "NSM_H",
    )

    # Extract result from NSM log (checkpoint_dir/experiment_name.log)
    nsm_log_file = EPR_ROOT / "NSM_H" / "checkpoint" / "pretrain" / f"{pert_tag}_nsm.log"
    result_lines = []
    if nsm_log_file.exists():
        with open(nsm_log_file) as f:
            for line in f:
                if "TEST F1" in line or "TEST H1" in line:
                    result_lines.append(line.strip())
    if result_lines:
        result_text = "\n".join(result_lines)
        with open(result_dir / "best_eval.txt", "w") as f:
            f.write(result_text + "\n")
        print(f"\n  NSM result ({dataset} {ptype}):\n  {result_text}")
    else:
        print(f"  [WARN] NSM result not found in log: {nsm_log_file}")


# ── Main ───────────────────────────────────────────────────────────────────────

STAGE_ORDER = ["prep", "ap", "ep", "subg", "nsm"]


def run_clean_baseline(dataset: str, stages: List[str]):
    """Run the clean (unperturbed) EPR-KGQA pipeline stages using the original config."""
    tag = DATASET_TAG[dataset]
    print(f"\n{'='*65}")
    print(f"EPR-KGQA Clean Baseline: {dataset.upper()}")
    print(f"  stages: {stages}")
    print(f"{'='*65}")

    config_path = EPR_ROOT / f"config_{tag}.yaml"
    if not config_path.exists():
        print(f"[ERROR] Clean config not found: {config_path}")
        sys.exit(1)

    test_file = EPR_ROOT / "data" / "dataset" / tag / (
        f"ComplexWebQuestions_test.json" if dataset == "cwq" else "test_simple.jsonl"
    )

    if "ap" in stages or "all" in stages:
        stage_ap(dataset, "clean", config_path, test_file)

    if "ep" in stages or "all" in stages:
        stage_ep(dataset, "clean", config_path)

    if "subg" in stages or "all" in stages:
        stage_subg(dataset, "clean", config_path)

    if "nsm" in stages or "all" in stages:
        stage_nsm(dataset, "clean", config_path)

    print(f"\n[DONE] {dataset} clean baseline")


def run_experiment(dataset: str, ptype: str, stages: List[str]):
    # Clean baseline uses the original EPR config directly (no prep needed)
    if ptype == "clean":
        run_clean_baseline(dataset, stages)
        return

    print(f"\n{'='*65}")
    print(f"EPR-KGQA Perturbation Experiment: {dataset.upper()} {ptype}")
    print(f"  stages: {stages}")
    print(f"{'='*65}")

    config_path = test_file = sr_file = None

    if "prep" in stages or "all" in stages:
        config_path, test_file, sr_file = stage_prep(dataset, ptype)
    else:
        # Reconstruct paths if skipping prep
        tag = DATASET_TAG[dataset]
        pert_tag = f"{tag}_{ptype}"
        config_path = EPR_ROOT / f"config_{pert_tag}.yaml"
        ext = "jsonl" if dataset == "webqsp" else "json"
        test_file = EPR_ROOT / "data" / "pert" / f"{pert_tag}_test.{ext}"
        sr_file   = EPR_ROOT / "data" / "pert" / f"{pert_tag}_sr_info.jsonl" if dataset == "cwq" else None
        if not config_path.exists():
            print(f"[ERROR] Config not found: {config_path} — run --stage prep first")
            sys.exit(1)

    if "ap" in stages or "all" in stages:
        stage_ap(dataset, ptype, config_path, test_file)

    if "ep" in stages or "all" in stages:
        stage_ep(dataset, ptype, config_path)

    if "subg" in stages or "all" in stages:
        stage_subg(dataset, ptype, config_path)

    if "nsm" in stages or "all" in stages:
        stage_nsm(dataset, ptype, config_path)

    print(f"\n[DONE] {dataset} {ptype}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", choices=["cwq", "webqsp"], default=["cwq"])
    parser.add_argument("--ptype",   nargs="+", choices=["p1", "p4", "p7", "clean"], default=["p4", "p7"])
    parser.add_argument("--stage",   nargs="+",
                        choices=STAGE_ORDER + ["all"], default=["all"])
    parser.add_argument("--gpu",     type=str, default="0")
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    for dataset in args.dataset:
        for ptype in args.ptype:
            run_experiment(dataset, ptype, args.stage)

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
