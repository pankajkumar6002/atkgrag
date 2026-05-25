#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
e2_webqsp_gnn_rag.py — Experiment E2: Stage-wise Attack on WebQSP using GNN-RAG

Mirror of e1_cwq_gnn_rag.py for the WebQSP dataset (1,639 test questions).

Run modes:
  full (default):  use gnn_input/webqsp_{p_type}/test_subgraph.json (ELQ-seeded subgraph)
  fast:            substitute question text only, reuse original GNN-RAG subgraph

Usage (new P1–P6 pipeline):
  python e2_webqsp_gnn_rag.py --all-pert-types --el-source elq
  python e2_webqsp_gnn_rag.py --pert-type p1 --mode fast --dry-run

Usage (legacy, backwards compat):
  CUDA_VISIBLE_DEVICES=0,1,3 python e2_webqsp_gnn_rag.py --variant greedy
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import numpy as np

GNN_RAG_GNN = REPO_ROOT / "external" / "GNN-RAG" / "gnn"
PERT_ROOT = REPO_ROOT / "data" / "perturbations_raw"
GNN_DATA = REPO_ROOT / "external" / "GNN-RAG" / "data"
DATASET_DIR = REPO_ROOT / "data" / "perturbations"
SUBGRAPH_VAR = DATASET_DIR / "subgraph_var" / "gnn_input"
CKPT_WEBQSP = GNN_DATA / "pretrain" / "ReaRev_webqsp.ckpt"
SPARQL_ENDPOINT = "http://localhost:8890/sparql"

PERT_TYPES = [
    "active_passive_flip",
    "adversarial_entity",
    "entity_alias_substitution",
    "question_reordering",
    "relation_synonym",
]
PERT_TO_CONDITION = {
    "entity_alias_substitution": "Lex",
    "relation_synonym": "Lex",
    "active_passive_flip": "Str",
    "question_reordering": "Str",
    "adversarial_entity": "Cmp",
}

NEW_PERT_TYPES = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
NEW_PERT_TO_CONDITION = {
    "p1": "Lex", "p2": "Str", "p3": "Sem",
    "p4": "Cmp", "p5": "Rel", "p6": "Noi", "p7": "RelSwap",
}

if str(GNN_RAG_GNN) not in sys.path:
    sys.path.insert(0, str(GNN_RAG_GNN))
SCRIPTS_ROOT = Path(__file__).parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from metrics import exact_match_kgqa, f1_kgqa, mean_ci, save_results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gnn_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    records = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["id"]] = r
    return records


def load_webqsp_perturbations(pert_file: Path) -> Dict[str, str]:
    """Returns {QuestionId: perturbed_ProcessedQuestion} — legacy format."""
    with pert_file.open() as f:
        data = json.load(f)
    questions = data["Questions"] if isinstance(data, dict) else data
    out = {}
    for r in questions:
        qid = r.get("QuestionId", "")
        pert = str(r.get("ProcessedQuestion") or r.get("RawQuestion") or "").strip()
        if qid and pert:
            out[qid] = pert
    return out


def load_new_pert_questions_webqsp(p_type: str) -> Dict[str, str]:
    """Returns {qid: perturbed_question} from data/perturbations/webqsp_{p_type}.json."""
    path = DATASET_DIR / f"webqsp_{p_type}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}. Run reorganize_pert.py first.")
    with open(path) as f:
        records = json.load(f)
    return {r["id"]: r["perturbed_question"] for r in records}


def load_webqsp_gold_answers(ori_path: Path) -> Dict[str, List[str]]:
    """Load gold answers keyed by question ID.

    Supports:
    - GNN-RAG JSONL: one JSON object per line, {"id":..., "answers":[{"kb_id":...}]}
    - Original WebQSP JSON: {"Questions":[{"QuestionId":..., "Parses":[...]}]}

    Returns {qid: [mid_string, ...]} for MID-based EM comparison.
    GNN-RAG predictions are Freebase MID strings (id2entity maps int → MID).
    """
    out: Dict[str, List[str]] = {}
    with ori_path.open() as f:
        first_line = f.readline().strip()

    if '"id"' in first_line and '"answers"' in first_line:
        # GNN-RAG JSONL format (one JSON object per line)
        with ori_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = r.get("id", "")
                mids = [a["kb_id"] for a in r.get("answers", [])
                        if isinstance(a, dict) and a.get("kb_id")]
                out[qid] = mids
    else:
        # Original WebQSP JSON: {"Questions": [...]}
        with ori_path.open() as f:
            data = json.load(f)
        questions = data["Questions"] if isinstance(data, dict) else data
        for r in questions:
            qid = r.get("QuestionId", "")
            mids: List[str] = []
            seen: set = set()
            for parse in r.get("Parses", []):
                for a in parse.get("Answers", []):
                    mid = a.get("AnswerArgument", "")  # Freebase MID
                    if mid and mid not in seen:
                        seen.add(mid)
                        mids.append(mid)
            out[qid] = mids
    return out


# ---------------------------------------------------------------------------
# GNN-RAG evaluation
# ---------------------------------------------------------------------------

def run_gnn_rag_on_records(
    records: List[Dict[str, Any]],
    ckpt_path: Path,
    data_dir: Path,
    dataset: str = "webqsp",
    word_emb_path=None,
    vocab_path=None,
    entity_path=None,   # override entities.txt (e.g. graftnet extended)
    relation_path=None, # override relations.txt
) -> Dict[str, str]:
    """Returns {id: prediction_string}, predictions read from test.info by position."""
    import uuid
    import argparse as _ap
    import logging

    try:
        from parsing import create_parser_rearev
        from train_model import Trainer_KBQA
    except ImportError as e:
        print(f"[ERROR] GNN-RAG import failed: {e}")
        sys.exit(1)

    exp_name = f"pert_{uuid.uuid4().hex[:10]}"

    # GNN-RAG dataset_load.py skips records with empty entities[] without writing to test.info.
    # Positional matching must use only the non-empty records to avoid ID→prediction mismatch.
    empty_entity_ids = {r["id"] for r in records if not r.get("entities")}
    filtered_records = [r for r in records if r.get("entities")]
    filtered_record_ids = [r["id"] for r in filtered_records]

    tmp_dir = Path(tempfile.mkdtemp(prefix="gnnrag_wqsp_"))
    info_path = ckpt_path.parent / f"{exp_name}_test.info"
    try:
        # Normalize answers: GNN-RAG expects [{"kb_id": "m.xxx", "text": ""}]
        # Load entity vocab to filter answers that would KeyError in dataset_load.py
        _ent_vocab_path = entity_path if entity_path and Path(entity_path).exists() else data_dir / dataset / "entities.txt"
        _known_entities: set = set()
        if Path(_ent_vocab_path).exists():
            with open(_ent_vocab_path) as _ef:
                for _line in _ef:
                    _mid = _line.strip()
                    if _mid: _known_entities.add(_mid)

        def _norm_answers(answers):
            out = []
            for a in answers:
                if isinstance(a, dict):
                    kb_id = a.get("kb_id", "")
                    if not _known_entities or kb_id in _known_entities:
                        out.append(a)
                elif isinstance(a, str):
                    if not _known_entities or a in _known_entities:
                        out.append({"kb_id": a, "text": a})
            return out

        # Baseline relation vocab size — remap out-of-baseline relation IDs to UNK_REL
        # Entities may exceed baseline; GNN-RAG re-indexes locally so this is safe
        _base_rel_path = data_dir / dataset / "relations.txt"
        import subprocess as _sp
        _n_baseline_rel = int(_sp.check_output(["wc", "-l", str(_base_rel_path)]).split()[0]) if _base_rel_path.exists() else 10**9
        _UNK_REL = _n_baseline_rel - 1

        def _adapt_subgraph(sg):
            if not sg: return sg
            adapted = []
            for t in sg.get("tuples", []):
                if len(t) < 3: continue
                s, r, o = t[0], t[1], t[2]
                adapted.append([s, r if r < _n_baseline_rel else _UNK_REL, o])
            return {"tuples": adapted, "entities": sg.get("entities", [])}

        tmp_test = tmp_dir / "test.json"
        with tmp_test.open("w") as f:
            for r in filtered_records:
                rec = dict(r)
                rec["answers"] = _norm_answers(rec.get("answers", []))
                rec["subgraph"] = _adapt_subgraph(rec.get("subgraph", {}))
                f.write(json.dumps(rec) + "\n")

        for fname in ["test.dep", "dev.json", "entity_emb_100d.npy", "relation_emb_100d.npy"]:
            src = data_dir / dataset / fname
            if src.exists():
                shutil.copy(src, tmp_dir / fname)
        # entities.txt: use extended version if provided; relations.txt: always baseline
        _ent_src = entity_path if entity_path and Path(entity_path).exists() else data_dir / dataset / "entities.txt"
        _rel_src = data_dir / dataset / "relations.txt"  # baseline only
        if Path(_ent_src).exists():
            shutil.copy(_ent_src, tmp_dir / "entities.txt")
        if Path(_rel_src).exists():
            shutil.copy(_rel_src, tmp_dir / "relations.txt")

        _vocab_src = vocab_path if vocab_path and Path(vocab_path).exists() else data_dir / dataset / "vocab.txt"
        _emb_src = word_emb_path if word_emb_path and Path(word_emb_path).exists() else data_dir / dataset / "word_emb.npy"
        if Path(_vocab_src).exists():
            shutil.copy(_vocab_src, tmp_dir / "vocab.txt")
        if Path(_emb_src).exists():
            shutil.copy(_emb_src, tmp_dir / "word_emb.npy")

        parser = _ap.ArgumentParser()
        create_parser_rearev(parser)
        args = parser.parse_args([
            "--name", dataset,
            "--data_folder", str(tmp_dir) + "/",
            "--is_eval",
            "--load_experiment", ckpt_path.name,
            "--checkpoint_dir", str(ckpt_path.parent),
            "--experiment_name", exp_name,
            "--lm", "sbert",
            "--word2id", "vocab.txt",
            "--entity2id", "entities.txt",
            "--relation2id", "relations.txt",
            "--num_ins", "2",   # WebQSP checkpoint trained with num_ins=2 (not default 3)
            "--num_gnn", "3",
        ])
        args.use_cuda = torch.cuda.is_available()

        logger = logging.getLogger(f"gnnrag_{exp_name}")
        logger.setLevel(logging.WARNING)

        trainer = Trainer_KBQA(args=vars(args), model_name="ReaRev", logger=logger)

        # Skip dev eval — call test evaluation directly
        trainer.load_ckpt(str(ckpt_path))
        trainer.evaluate(trainer.test_data, trainer.test_batch_size)

        # Read predictions from test.info positionally.
        # test.info has one line per NON-EMPTY-entity record (GNN-RAG skips empty ones).
        # Use filtered_record_ids for positional matching; pre-fill empty-entity records
        # with empty prediction (no seeds → subgraph retrieval failed → EM=0).
        pred_map: Dict[str, str] = {qid: "" for qid in empty_entity_ids}
        cands_map: Dict[str, list] = {qid: [] for qid in empty_entity_ids}
        gnn_metrics: Dict[str, dict] = {}
        if info_path.exists():
            with info_path.open() as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or i >= len(filtered_record_ids):
                        continue
                    obj = json.loads(line)
                    cands = obj.get("cand", [])
                    qid = filtered_record_ids[i]
                    if cands:
                        best = cands[0]
                        pred_map[qid] = best[0] if isinstance(best, (list, tuple)) else str(best)
                        cands_map[qid] = cands[:5]
                    else:
                        pred_map[qid] = ""
                        cands_map[qid] = []
                    gnn_metrics[qid] = {
                        "hit1": obj.get("hit", 0.0),
                        "em":   obj.get("em",  0.0),
                        "f1":   obj.get("f1",  0.0),
                        "precision": obj.get("precision", obj.get("precison", 0.0)),
                        "recall":    obj.get("recall",   0.0),
                        "answers":   obj.get("answers",  []),
                    }
        else:
            print(f"  [WARN] test.info not found: {info_path}")
        return pred_map, cands_map, gnn_metrics
    finally:
        # Explicitly free GPU memory before returning — trainer holds ~53 GB and the
        # caller may invoke this function twice (clean pass then pert pass).
        try:
            del trainer
        except NameError:
            pass
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        if info_path.exists():
            info_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Per-type evaluation
# ---------------------------------------------------------------------------

def evaluate_type(
    pert_type: str,
    variant: str,
    gnn_orig: Dict[str, Dict[str, Any]],
    gold_answers: Dict[str, List[str]],
    n_boot: int = 1000,
) -> Dict[str, Any]:
    pert_file = PERT_ROOT / "WebQSP" / variant / \
                f"WebQSP.test_pert_{pert_type}.json"
    if not pert_file.exists():
        return {"skipped": True, "reason": f"not found: {pert_file}"}

    pert_qs = load_webqsp_perturbations(pert_file)
    common_ids = sorted(set(gnn_orig) & set(pert_qs) & set(gold_answers))
    if not common_ids:
        return {"skipped": True, "reason": "no overlapping IDs"}
    print(f"    {len(common_ids)} aligned questions")

    orig_records = [gnn_orig[qid] for qid in common_ids]
    pert_records = []
    for qid in common_ids:
        r = copy.deepcopy(gnn_orig[qid])
        r["question"] = pert_qs[qid]
        pert_records.append(r)

    print(f"    Running GNN-RAG on clean questions...")
    orig_preds, _, _ = run_gnn_rag_on_records(orig_records, CKPT_WEBQSP, GNN_DATA, "webqsp")
    print(f"    Running GNN-RAG on perturbed questions ({pert_type})...")
    pert_preds, _, _ = run_gnn_rag_on_records(pert_records, CKPT_WEBQSP, GNN_DATA, "webqsp")

    per_q, em_c_all, em_p_all, f1_c_all, f1_p_all = [], [], [], [], []
    for qid in common_ids:
        gold = gold_answers.get(qid, [])
        o_pred = orig_preds.get(qid, "")
        p_pred = pert_preds.get(qid, "")
        em_c = exact_match_kgqa(o_pred, gold)
        em_p = exact_match_kgqa(p_pred, gold)
        f1_c = f1_kgqa(o_pred, gold)
        f1_p = f1_kgqa(p_pred, gold)
        em_c_all.append(em_c); em_p_all.append(em_p)
        f1_c_all.append(f1_c); f1_p_all.append(f1_p)
        per_q.append({"id": qid, "em_clean": em_c, "em_pert": em_p,
                      "f1_clean": f1_c, "f1_pert": f1_p,
                      "orig_pred": o_pred, "pert_pred": p_pred})

    dag_em = [c - p for c, p in zip(em_c_all, em_p_all)]
    dag_f1 = [c - p for c, p in zip(f1_c_all, f1_p_all)]
    return {
        "condition": PERT_TO_CONDITION.get(pert_type, "?"),
        "pert_type": pert_type, "n": len(common_ids),
        "EM_clean": mean_ci(em_c_all, n_boot),
        "EM_pert": mean_ci(em_p_all, n_boot),
        "delta_AG_EM": mean_ci(dag_em, n_boot),
        "F1_clean": mean_ci(f1_c_all, n_boot),
        "F1_pert": mean_ci(f1_p_all, n_boot),
        "delta_AG_F1": mean_ci(dag_f1, n_boot),
        "per_question": per_q,
    }


# ---------------------------------------------------------------------------
# New P1–P6 evaluation
# ---------------------------------------------------------------------------

def _metrics_for_ids(
    common_ids, orig_preds, pert_preds, gold_answers, p_type, el_source, n_boot
):
    from metrics import mean_ci
    per_q, em_c_all, em_p_all, f1_c_all, f1_p_all = [], [], [], [], []
    for qid in common_ids:
        gold = gold_answers.get(qid, [])
        o_pred, p_pred = orig_preds.get(qid, ""), pert_preds.get(qid, "")
        em_c, em_p = exact_match_kgqa(o_pred, gold), exact_match_kgqa(p_pred, gold)
        f1_c, f1_p = f1_kgqa(o_pred, gold), f1_kgqa(p_pred, gold)
        em_c_all.append(em_c); em_p_all.append(em_p)
        f1_c_all.append(f1_c); f1_p_all.append(f1_p)
        per_q.append({"id": qid, "em_clean": em_c, "em_pert": em_p,
                      "f1_clean": f1_c, "f1_pert": f1_p,
                      "orig_pred": o_pred, "pert_pred": p_pred})
    dag_em = [c - p for c, p in zip(em_c_all, em_p_all)]
    dag_f1 = [c - p for c, p in zip(f1_c_all, f1_p_all)]
    return {
        "condition": NEW_PERT_TO_CONDITION.get(p_type, "?"),
        "pert_type": p_type, "el_source": el_source, "n": len(common_ids),
        "EM_clean": mean_ci(em_c_all, n_boot), "EM_pert": mean_ci(em_p_all, n_boot),
        "delta_AG_EM": mean_ci(dag_em, n_boot),
        "F1_clean": mean_ci(f1_c_all, n_boot), "F1_pert": mean_ci(f1_p_all, n_boot),
        "delta_AG_F1": mean_ci(dag_f1, n_boot), "per_question": per_q,
    }


def load_gnn_jsonl_local(path):
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["id"]] = r
    return records


def evaluate_type_full(
    p_type: str,
    el_source: str,
    gnn_orig: Dict[str, Dict[str, Any]],
    gold_answers: Dict[str, List[str]],
    n_boot: int = 1000,
) -> Dict[str, Any]:
    """Full mode: load perturbed records from gnn_input (el_source-seeded subgraph)."""
    NEW_EXP = REPO_ROOT
    _ABLATION_SOURCES = {
        "nsm_gem", "nsm_elq", "graftnetv3_gem", "graftnetv3_elq",
    }
    if el_source in _ABLATION_SOURCES:
        subdir = f"webqsp_{p_type}_{el_source}"
        gnn_input_path = NEW_EXP / "gnn_input" / subdir / "test_subgraph.json"
    elif el_source == "elq":
        subdir = f"webqsp_{p_type}"
        gnn_input_path = DATASET_DIR / "gnn_input" / subdir / "test_subgraph.json"
    elif el_source == "graftnet":
        subdir = f"webqsp_{p_type}_graftnet"
        gnn_input_path = DATASET_DIR / "gnn_input" / subdir / "test_subgraph.json"
    elif el_source == "agent_fresh":
        subdir = f"webqsp_{p_type}_agent_fresh"
        gnn_input_path = DATASET_DIR / "gnn_input" / subdir / "test_subgraph.json"
    elif el_source == "agent_v1":
        subdir = f"webqsp_{p_type}_agent_v1"
        gnn_input_path = DATASET_DIR / "gnn_input" / subdir / "test_subgraph.json"
    elif el_source == "nsm":
        subdir = f"webqsp_{p_type}_nsm"
        gnn_input_path = DATASET_DIR / "gnn_input" / subdir / "test_subgraph.json"
    elif el_source == "graftnetv3":
        subdir = f"webqsp_{p_type}_graftnetv3"
        gnn_input_path = NEW_EXP / "gnn_input" / subdir / "test_subgraph.json"
    elif el_source == "graftnetv3_fixed":
        subdir = f"webqsp_{p_type}_graftnetv3"
        gnn_input_path = SUBGRAPH_VAR / subdir / "test_subgraph.json"
    else:
        subdir = f"webqsp_{p_type}_ela"
        gnn_input_path = DATASET_DIR / "gnn_input" / subdir / "test_subgraph.json"
    if not gnn_input_path.exists():
        return {"skipped": True, "reason": f"gnn_input not found: {gnn_input_path}"}

    gnn_pert = load_gnn_jsonl_local(gnn_input_path)
    common_ids = sorted(set(gnn_orig) & set(gnn_pert) & set(gold_answers))
    if not common_ids:
        return {"skipped": True, "reason": "no overlapping IDs"}
    print(f"    {len(common_ids)} aligned questions (gnn_input={len(gnn_pert)}, el={el_source})")

    emb_dir = DATASET_DIR / "embeddings"
    word_emb_path = emb_dir / "webqsp_pert_word_emb.npy"
    vocab_path = emb_dir / "webqsp_pert_vocab.txt"

    # Pass extended entities.txt for graftnet/nsm/graftnetv3/ablation; relations.txt always stays baseline
    _new_exp_sources = {"graftnetv3"} | _ABLATION_SOURCES
    if el_source == "graftnetv3_fixed":
        gnn_input_dir = SUBGRAPH_VAR / subdir
    elif el_source in _new_exp_sources:
        gnn_input_dir = NEW_EXP / "gnn_input" / subdir
    else:
        gnn_input_dir = DATASET_DIR / "gnn_input" / subdir
    _extended_vocab_sources = {"graftnet", "nsm", "graftnetv3", "graftnetv3_fixed"} | _ABLATION_SOURCES
    entity_path = gnn_input_dir / "entities.txt" if el_source in _extended_vocab_sources else None

    orig_records = [gnn_orig[qid] for qid in common_ids]
    pert_records = [gnn_pert[qid] for qid in common_ids]

    print(f"    Running GNN-RAG on clean questions (original subgraph)...")
    orig_preds, _, orig_gnn_metrics = run_gnn_rag_on_records(orig_records, CKPT_WEBQSP, GNN_DATA, "webqsp",
                                                              word_emb_path=word_emb_path, vocab_path=vocab_path)
    print(f"    Running GNN-RAG on perturbed questions ({p_type}, {el_source} subgraph)...")
    pert_preds, _, pert_gnn_metrics = run_gnn_rag_on_records(pert_records, CKPT_WEBQSP, GNN_DATA, "webqsp",
                                                              word_emb_path=word_emb_path, vocab_path=vocab_path,
                                                              entity_path=entity_path)

    res = _metrics_for_ids(common_ids, orig_preds, pert_preds, gold_answers,
                            p_type, el_source, n_boot)

    def _avg_gnn(metrics_map):
        if not metrics_map: return {}
        vals = list(metrics_map.values())
        keys = ["em", "hit1", "f1", "precision", "recall"]
        return {k: round(sum(v.get(k, 0) for v in vals) / len(vals), 4) for k in keys if any(k in v for v in vals)}

    res["gnn_internal_clean"] = _avg_gnn(orig_gnn_metrics)
    res["gnn_internal_pert"] = _avg_gnn(pert_gnn_metrics)
    return res


def evaluate_type_new(
    p_type: str,
    el_source: str,
    gnn_orig: Dict[str, Dict[str, Any]],
    gold_answers: Dict[str, List[str]],
    n_boot: int = 1000,
) -> Dict[str, Any]:
    """Fast mode: substitute question text only, reuse original GNN-RAG subgraph."""
    pert_qs = load_new_pert_questions_webqsp(p_type)
    common_ids = sorted(set(gnn_orig) & set(pert_qs) & set(gold_answers))
    if not common_ids:
        return {"skipped": True, "reason": "no overlapping IDs"}
    print(f"    {len(common_ids)} aligned questions")

    emb_dir = DATASET_DIR / "embeddings"
    word_emb_path = emb_dir / "webqsp_pert_word_emb.npy"
    vocab_path = emb_dir / "webqsp_pert_vocab.txt"

    orig_records = [gnn_orig[qid] for qid in common_ids]
    pert_records = []
    for qid in common_ids:
        r = copy.deepcopy(gnn_orig[qid])
        r["question"] = pert_qs[qid]
        pert_records.append(r)

    print(f"    Running GNN-RAG on clean questions...")
    orig_preds, _, _ = run_gnn_rag_on_records(orig_records, CKPT_WEBQSP, GNN_DATA, "webqsp",
                                               word_emb_path=word_emb_path, vocab_path=vocab_path)
    print(f"    Running GNN-RAG on perturbed questions ({p_type}, fast mode)...")
    pert_preds, _, _ = run_gnn_rag_on_records(pert_records, CKPT_WEBQSP, GNN_DATA, "webqsp",
                                               word_emb_path=word_emb_path, vocab_path=vocab_path)

    return _metrics_for_ids(common_ids, orig_preds, pert_preds, gold_answers,
                             p_type, el_source, n_boot)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2: stage-wise WebQSP attack — GNN-RAG")
    p.add_argument("--pert-type", dest="pert_types_new", nargs="+",
                   choices=NEW_PERT_TYPES, default=None)
    p.add_argument("--all-pert-types", action="store_true")
    p.add_argument("--el-source",
                   choices=["elq", "graftnet", "graftnetv3", "graftnetv3_fixed", "agent", "agent_fresh", "agent_v1", "nsm",
                            "nsm_gem", "nsm_elq", "graftnetv3_gem", "graftnetv3_elq"],
                   default="elq",
                   help="Ablation cells: nsm_gem, nsm_elq, graftnetv3_gem, graftnetv3_elq")
    p.add_argument("--mode", choices=["full", "fast"], default="full",
                   help="full=use gnn_input ELQ subgraph; fast=question text substitution only")
    p.add_argument("--variant", default=None)
    p.add_argument("--types", nargs="+", default=PERT_TYPES)
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / "e2_webqsp_gnn_rag")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-run p-types already present in summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Bug 3: graftnetv3 (no suffix) silently reads OLD subgraphs from new_exp/.
    # Use graftnetv3_fixed for the correct bug-fixed subgraphs in subgraph_var/.
    if args.el_source == "graftnetv3":
        import warnings
        warnings.warn(
            "\n[DEPRECATION] --el-source graftnetv3 reads OLD subgraphs from new_exp/gnn_input/.\n"
            "Use --el-source graftnetv3_fixed to read CORRECT subgraphs from subgraph_var/gnn_input/.\n"
            "Proceeding with old subgraphs — results will NOT match paper.",
            DeprecationWarning, stacklevel=2,
        )

    use_new = args.all_pert_types or args.pert_types_new
    run_label = args.el_source if use_new else (args.variant or "greedy")

    if use_new:
        mode_tag = f"{args.mode}_{args.el_source}"
        # Bug 2 prevention: detect if --out-dir was mistakenly passed with mode_tag already appended.
        if args.out_dir.name == mode_tag:
            raise SystemExit(
                f"[ERROR] --out-dir already ends with '{mode_tag}'.\n"
                f"  Pass the base directory (e.g. results/e2_webqsp_gnn_rag) and let the script append '{mode_tag}'.\n"
                f"  Got: {args.out_dir}"
            )

    (args.out_dir / run_label).mkdir(parents=True, exist_ok=True)

    print("Loading GNN-RAG WebQSP JSONL...")
    gnn_orig = load_gnn_jsonl(GNN_DATA / "webqsp" / "test.json")
    print(f"  {len(gnn_orig)} records")

    if args.dry_run:
        limited_ids = sorted(gnn_orig.keys())[:10]
        gnn_orig = {k: gnn_orig[k] for k in limited_ids}
        print(f"  [DRY RUN] limited to {len(gnn_orig)} questions")

    print("Loading WebQSP gold answers...")
    gold_answers = load_webqsp_gold_answers(
        GNN_DATA / "webqsp" / "test.json"
    )

    if use_new:
        p_types = NEW_PERT_TYPES if args.all_pert_types else args.pert_types_new
        mode_tag = f"{args.mode}_{args.el_source}"
        out_subdir = args.out_dir / mode_tag
        out_subdir.mkdir(parents=True, exist_ok=True)
        print(f"Mode: {args.mode} | EL: {args.el_source}")

        summary_path = out_subdir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as _sf:
                all_results: Dict[str, Any] = json.load(_sf)
        else:
            all_results: Dict[str, Any] = {
                "mode": args.mode, "el_source": args.el_source,
                "model": "GNN-RAG (ReaRev)", "dataset": "WebQSP"
            }
        eval_fn = evaluate_type_full if args.mode == "full" else evaluate_type_new

        for p_type in p_types:
            if p_type in all_results and not args.force:
                print(f"\n[{p_type}] already in summary — skipping (use --force to re-run)")
                continue
            print(f"\n[{p_type}]")
            res = eval_fn(p_type, args.el_source, gnn_orig, gold_answers, args.n_boot)
            all_results[p_type] = {k: v for k, v in res.items() if k != "per_question"}
            if not res.get("skipped"):
                save_results(out_subdir / f"{p_type}_per_question.json", res["per_question"])
                dag = res["delta_AG_EM"]
                print(f"  EM_clean={res['EM_clean']['mean']:.3f}  "
                      f"EM_pert={res['EM_pert']['mean']:.3f}  "
                      f"ΔAG={dag['mean']:.3f}")

        print("\n" + "=" * 75)
        print(f"EL: {args.el_source} | Model: GNN-RAG | Dataset: WebQSP")
        print(f"{'Type':<6} {'Cond':<5} {'EM_cl':>7} {'EM_pt':>7} {'ΔAG':>7}")
        print("-" * 75)
        _PTYPE_KEYS = set(NEW_PERT_TYPES)
        for p_type in sorted(k for k in all_results if k in _PTYPE_KEYS):
            r = all_results.get(p_type, {})
            if r.get("skipped"):
                print(f"  {p_type:<4} {'?':<5} skipped")
                continue
            dag = r["delta_AG_EM"]
            print(f"  {p_type:<4} {r['condition']:<5}"
                  f" {r['EM_clean']['mean']:>7.3f}"
                  f" {r['EM_pert']['mean']:>7.3f}"
                  f" {dag['mean']:>7.3f}")
        save_results(out_subdir / "summary.json", all_results)
        print(f"\nResults saved to {out_subdir}/")
        return

    # Legacy pipeline
    variant = args.variant or "greedy"
    all_ids = sorted(gnn_orig.keys())
    baseline_preds, _, _ = run_gnn_rag_on_records(
        [gnn_orig[qid] for qid in all_ids], CKPT_WEBQSP, GNN_DATA, "webqsp"
    )
    em_baseline = [exact_match_kgqa(baseline_preds.get(qid, ""), gold_answers.get(qid, []))
                   for qid in all_ids]
    print(f"  Baseline EM: {np.mean(em_baseline):.4f}")
    save_results(args.out_dir / variant / "baseline_preds.json",
                 {qid: baseline_preds.get(qid, "") for qid in all_ids})

    all_results = {
        "variant": variant, "model": "GNN-RAG (ReaRev)",
        "dataset": "WebQSP", "baseline_EM": float(np.mean(em_baseline)),
    }
    for pert_type in args.types:
        print(f"\n[{pert_type}]")
        res = evaluate_type(pert_type, variant, gnn_orig, gold_answers, args.n_boot)
        all_results[pert_type] = {k: v for k, v in res.items() if k != "per_question"}
        if not res.get("skipped"):
            save_results(args.out_dir / variant / f"{pert_type}_per_question.json",
                         res["per_question"])
            dag = res["delta_AG_EM"]
            print(f"  EM_clean={res['EM_clean']['mean']:.3f}  "
                  f"EM_pert={res['EM_pert']['mean']:.3f}  "
                  f"ΔAG={dag['mean']:.3f}")

    print("\n" + "=" * 75)
    print(f"Variant: {variant} | Model: GNN-RAG | Dataset: WebQSP")
    print(f"{'Type':<30} {'Cond':<5} {'EM_cl':>7} {'EM_pt':>7} {'ΔAG':>7}")
    print("-" * 75)
    for pert_type in args.types:
        r = all_results.get(pert_type, {})
        if r.get("skipped"):
            print(f"  {pert_type:<28} {'?':<5} {'—':>7} {'—':>7} {'—':>7}  skipped")
            continue
        dag = r["delta_AG_EM"]
        print(f"  {pert_type:<28} {r['condition']:<5}"
              f" {r['EM_clean']['mean']:>7.3f}"
              f" {r['EM_pert']['mean']:>7.3f}"
              f" {dag['mean']:>7.3f}")

    save_results(args.out_dir / args.variant / "summary.json", all_results)
    print(f"\nResults saved to {args.out_dir / args.variant}/")


if __name__ == "__main__":
    main()
