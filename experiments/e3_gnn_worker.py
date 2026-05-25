#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
e3_gnn_worker.py — Subprocess worker for GNN-RAG inference.

Called by e3_llm_reasoning.py to avoid utils module conflict between
GNN-RAG/gnn/utils.py and GNN-RAG/llm/src/utils/ package.

Usage:
  python e3_gnn_worker.py --dataset cwq --input records.jsonl --output cands.jsonl [--save-info path]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import torch

GNN_RAG_GNN = REPO_ROOT / "external" / "GNN-RAG" / "gnn"
GNN_DATA    = REPO_ROOT / "external" / "GNN-RAG" / "data"

if str(GNN_RAG_GNN) not in sys.path:
    sys.path.insert(0, str(GNN_RAG_GNN))

CKPT = {
    "cwq":    GNN_DATA / "pretrain" / "ReaRev_CWQ.ckpt",
    "webqsp": GNN_DATA / "pretrain" / "ReaRev_webqsp.ckpt",
}
# num_ins must match checkpoint architecture: CWQ ckpt=3 (default), webqsp ckpt=2
NUM_INS = {"cwq": 3, "webqsp": 2}
DATASET_SUBDIR = {"cwq": "CWQ", "webqsp": "webqsp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--input",     required=True, help="Input JSONL with subgraph records")
    parser.add_argument("--output",    required=True, help="Output JSONL with {id, cand} per record")
    parser.add_argument("--save-info", default=None,  help="Path to save raw test.info")
    args = parser.parse_args()

    try:
        from parsing import create_parser_rearev
        from train_model import Trainer_KBQA
    except ImportError as e:
        print(f"[GNN worker ERROR] import failed: {e}")
        sys.exit(1)

    dataset = args.dataset
    ckpt_path = CKPT[dataset]
    data_dir = GNN_DATA
    dataset_subdir = DATASET_SUBDIR[dataset]

    # Load records
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[GNN worker] {len(records)} records, dataset={dataset}")

    empty_entity_ids = {r["id"] for r in records if not r.get("entities")}
    filtered = [r for r in records if r.get("entities")]
    filtered_ids = [r["id"] for r in filtered]

    exp_name = f"e3w_{uuid.uuid4().hex[:10]}"
    tmp_dir = Path(tempfile.mkdtemp(prefix="e3gnn_"))
    info_path = ckpt_path.parent / f"{exp_name}_test.info"

    try:
        # Entity vocab
        ent_vocab_path = data_dir / dataset_subdir / "entities.txt"
        known_entities: set = set()
        if ent_vocab_path.exists():
            with open(ent_vocab_path) as ef:
                for ln in ef:
                    mid = ln.strip()
                    if mid:
                        known_entities.add(mid)

        def norm_answers(answers):
            out = []
            for a in answers:
                if isinstance(a, dict):
                    kb_id = a.get("kb_id", "")
                    if not known_entities or kb_id in known_entities:
                        out.append(a)
                elif isinstance(a, str):
                    if not known_entities or a in known_entities:
                        out.append({"kb_id": a, "text": a})
            return out

        # Relation remapping
        base_rel_path = data_dir / dataset_subdir / "relations.txt"
        import subprocess as _sp
        n_baseline_rel = int(_sp.check_output(["wc", "-l", str(base_rel_path)]).split()[0]) if base_rel_path.exists() else 10**9
        unk_rel = n_baseline_rel - 1

        def adapt_sg(sg):
            if not sg:
                return sg
            tuples = [[t[0], t[1] if t[1] < n_baseline_rel else unk_rel, t[2]]
                      for t in sg.get("tuples", []) if len(t) >= 3]
            return {"tuples": tuples, "entities": sg.get("entities", [])}

        tmp_test = tmp_dir / "test.json"
        with open(tmp_test, "w") as f:
            for r in filtered:
                rec = dict(r)
                rec["answers"] = norm_answers(rec.get("answers", []))
                rec["subgraph"] = adapt_sg(rec.get("subgraph", {}))
                f.write(json.dumps(rec) + "\n")

        for fname in ["test.dep", "dev.json"]:
            src = data_dir / dataset_subdir / fname
            if src.exists():
                shutil.copy(src, tmp_dir / fname)

        for fname in ["entities.txt", "relations.txt", "vocab.txt", "word_emb.npy"]:
            src = data_dir / dataset_subdir / fname
            if src.exists():
                shutil.copy(src, tmp_dir / fname)

        gnn_parser = argparse.ArgumentParser()
        create_parser_rearev(gnn_parser)
        gnn_args = gnn_parser.parse_args([
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
            "--num_ins", str(NUM_INS[dataset]),
        ])
        gnn_args.use_cuda = torch.cuda.is_available()

        logger = logging.getLogger(f"e3w_{exp_name}")
        logger.setLevel(logging.WARNING)

        trainer = Trainer_KBQA(args=vars(gnn_args), model_name="ReaRev", logger=logger)
        trainer.load_ckpt(str(ckpt_path))
        trainer.evaluate(trainer.test_data, trainer.test_batch_size)

        cands_map: dict = {qid: [] for qid in empty_entity_ids}
        if info_path.exists():
            if args.save_info:
                Path(args.save_info).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(info_path, args.save_info)
                print(f"[GNN worker] saved test.info → {args.save_info}")

            with open(info_path) as fi:
                for idx, line in enumerate(fi):
                    line = line.strip()
                    if not line or idx >= len(filtered_ids):
                        continue
                    obj = json.loads(line)
                    qid = filtered_ids[idx]
                    cands_map[qid] = obj.get("cand", [])[:10]
        else:
            print(f"[GNN worker WARN] test.info not found: {info_path}")

        # Write output
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for qid, cand in cands_map.items():
                f.write(json.dumps({"id": qid, "cand": cand}) + "\n")

        n_with_cands = sum(1 for v in cands_map.values() if v)
        print(f"[GNN worker] done — {n_with_cands}/{len(cands_map)} questions have candidates")

    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        if info_path.exists():
            info_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()