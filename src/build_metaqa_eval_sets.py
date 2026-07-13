#!/usr/bin/env python3
"""
build_metaqa_eval_sets.py : turn perturbation outputs into NSM-loader eval folders.

For each available pert file metaqa{hop}hop_{set}[ _full ].json, creates
  gnn/data/metaqa-{hop}hop-{set}[ -full ]/
    test.json      line-json in loader schema (id/question/entities/answers/subgraph)
    train.json,dev.json,entities.txt,relations.txt,vocab_new.txt,word_emb_300d.npy
                   -> symlinks to the original folder (loader requires them)
For LLM sets (p2..p7): question := perturbed text; entities/answers/subgraph from
the ORIGINAL test record (fixed-subgraph isolation).
For ES (p1): question/entities/answers/subgraph all from the ES record (rebuilt
subgraph around the swapped entity).

Usage: python build_metaqa_eval_sets.py --hops 3 2 --sets p1 p4 p7 [--full]
"""
import argparse
import json
from pathlib import Path

import os
_REPO_ROOT = Path(os.environ.get("CASCADEATTACK_ROOT",
                                 Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("METAQA_DATA_ROOT",
                                _REPO_ROOT / "data" / "metaqa"))
PERT_DIR = Path(os.environ.get("PERT_DATASET",
                               _REPO_ROOT / "data" / "perturbations"))
LINKS = ["train.json", "dev.json", "entities.txt", "relations.txt",
         "vocab_new.txt", "word_emb_300d.npy", "train_simple.json", "dev_simple.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", nargs="+", type=int, default=[3, 2])
    ap.add_argument("--sets", nargs="+", default=["p1", "p4", "p7"])
    ap.add_argument("--full", action="store_true", help="use *_full pert files")
    ap.add_argument("--control", action="store_true",
                    help="build clean-question control folder (metaqa-{hop}hop-ctrl3k) "
                         "over the same ids as the p4 sample file")
    args = ap.parse_args()

    if args.control:
        for hop in args.hops:
            src = DATA_ROOT / f"metaqa-{hop}hop"
            ids = {p["id"] for p in json.load(open(PERT_DIR / f"metaqa{hop}hop_p4.json"))}
            dst = DATA_ROOT / f"metaqa-{hop}hop-ctrl3k"
            dst.mkdir(exist_ok=True)
            for name in LINKS:
                link = dst / name
                if not link.exists():
                    link.symlink_to(src / name)
            n = 0
            with open(src / "test_simple.json") as f, open(dst / "test.json", "w") as out:
                for line in f:
                    if json.loads(line)["id"] in ids:
                        out.write(line)
                        n += 1
            print(f"built {dst.name}: {n} clean control Qs")
        return

    for hop in args.hops:
        src = DATA_ROOT / f"metaqa-{hop}hop"
        orig = {}
        with open(src / "test_simple.json") as f:
            for line in f:
                r = json.loads(line)
                orig[r["id"]] = r
        for set_id in args.sets:
            suffix = "_full" if args.full else ""
            pert_path = PERT_DIR / f"metaqa{hop}hop_{set_id}{suffix}.json"
            if not pert_path.exists():
                print(f"skip (missing): {pert_path}")
                continue
            pert = json.load(open(pert_path))
            dst = DATA_ROOT / f"metaqa-{hop}hop-{set_id}{'-full' if args.full else ''}"
            dst.mkdir(exist_ok=True)
            for name in LINKS:
                link = dst / name
                if not link.exists():
                    link.symlink_to(src / name)
            n_pert = 0
            with open(dst / "test.json", "w") as out:
                for p in pert:
                    o = orig[p["id"]]
                    if set_id == "p1":
                        rec = {"id": p["id"], "question": p["perturbed_question"],
                               "entities": p["entities"], "answers": p["answers"],
                               "subgraph": p.get("subgraph", o["subgraph"])}
                    else:
                        rec = {"id": p["id"], "question": p["perturbed_question"],
                               "entities": o["entities"], "answers": o["answers"],
                               "subgraph": o["subgraph"]}
                    n_pert += p["is_perturbed"]
                    out.write(json.dumps(rec) + "\n")
            print(f"built {dst.name}: {len(pert)} Qs ({n_pert} perturbed)")


if __name__ == "__main__":
    main()