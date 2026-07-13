#!/usr/bin/env python3
"""
build_metaqa_cascade_sets.py : FULL-PIPELINE (cascaded) eval sets for MetaQA.

Unlike the fast-mode folders (shipped NSM subgraph + perturbed text), these sets
re-run the whole front of the pipeline per question:

  stage 1  seed = link(question)          # automatic longest-match linker,
                                          # NO gold/manual seeding
  stage 2  subgraph = retrieve(seed)      # deterministic bidirectional BFS,
                                          # entity cap 500, union MetaQA KB
  stage 3  GNN-RAG inference              # ReaRev checkpoints in
                                          # checkpoint/pretrain/metaqa/

Two seeding conditions per variant:
  -casc-  seed = link(q')   full cascade: linker errors propagate downstream
  -gold-  seed = gold topic entity (for ES: the *intended swapped* entity X'),
          isolating retrieval+GNN from linking failures.

A "clean" variant (original questions) is built under both conditions : the
apples-to-apples cascade baseline, since the BFS retrieval operator differs
from the NSM-shipped subgraph construction and the GNN was trained on the
latter (absolute EM under the cascade is operator-specific; the clean-vs-
perturbed delta under the SAME operator is the measurement).

Questions whose linker returns nothing get entities=[] (the NSM loader skips
them); per-folder cascade_meta.json records n_total/n_linked/n_skipped so EM
can be renormalized: EM_true = EM_loader * n_kept / n_total.

Nothing is overwritten: all folders are new (metaqa-{hop}hop-casc-*/-gold-*).

Usage:
  python build_metaqa_cascade_sets.py --hops 3 2 --variants clean p4 p7 p1 [--full]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metaqa_es_p1 import load_union_kb
from metaqa_stage_isolation import link, retrieve

import os
_REPO_ROOT = Path(os.environ.get("CASCADEATTACK_ROOT",
                                 Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("METAQA_DATA_ROOT",
                                _REPO_ROOT / "data" / "metaqa"))
PERT_DIR = Path(os.environ.get("PERT_DATASET",
                               _REPO_ROOT / "data" / "perturbations"))
LINKS = ["train.json", "dev.json", "entities.txt", "relations.txt",
         "vocab_new.txt", "word_emb_300d.npy", "train_simple.json", "dev_simple.json"]


def write_folder(dst: Path, src: Path, rows, meta):
    dst.mkdir(exist_ok=True)
    for name in LINKS:
        lk = dst / name
        if not lk.exists():
            lk.symlink_to(src / name)
    with open(dst / "test.json", "w") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    json.dump(meta, open(dst / "cascade_meta.json", "w"), indent=1)
    print(f"built {dst.name}: total={meta['n_total']} linked={meta['n_linked']} "
          f"skipped={meta['n_skipped']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", nargs="+", type=int, default=[3, 2])
    ap.add_argument("--variants", nargs="+",
                    default=["clean", "p4", "p7", "p1", "p2", "p3", "p5", "p6"])
    ap.add_argument("--full", action="store_true", help="use *_full pert files")
    args = ap.parse_args()

    for hop in args.hops:
        src = DATA_ROOT / f"metaqa-{hop}hop"
        names = src.joinpath("entities.txt").read_text().splitlines()
        name2id = {}
        for i, nm in enumerate(names):
            name2id.setdefault(nm.lower(), i)
        max_len = max(len(nm.split()) for nm in names)
        print(f"[{hop}hop] loading union KB ...")
        fwd, bwd, out_rels, triples = load_union_kb(src)
        rel_ids = sorted({r for (_, r, _) in triples})
        rcache = {}

        def R(seed):
            if seed not in rcache:
                ents, tuples = retrieve(seed, hop, fwd, bwd, rel_ids)
                rcache[seed] = {"entities": sorted(ents),
                                "tuples": sorted(tuples)}
            return rcache[seed]

        orig = {}
        with open(src / "test_simple.json") as f:
            for line in f:
                r = json.loads(line)
                orig[r["id"]] = r

        for var in args.variants:
            if var == "clean":
                items = [(r["id"], r["question"], r["entities"][0], r["answers"])
                         for r in orig.values()]
                tagbase = "clean"
            else:
                suffix = "_full" if args.full else ""
                ppath = PERT_DIR / f"metaqa{hop}hop_{var}{suffix}.json"
                if not ppath.exists():
                    print(f"skip missing {ppath}")
                    continue
                pert = json.load(open(ppath))
                items = []
                for p in pert:
                    o = orig[p["id"]]
                    if var == "p1":  # ES: intended seed/answers are the swapped ones
                        items.append((p["id"], p["perturbed_question"],
                                      p["entities"][0], p["answers"]))
                    else:
                        items.append((p["id"], p["perturbed_question"],
                                      o["entities"][0], o["answers"]))
                tagbase = var + ("-full" if args.full else "")

            for cond in ["casc", "gold"]:
                rows, n_link, n_skip = [], 0, 0
                for qid, q, gold_seed, answers in items:
                    seed = link(q, name2id, max_len) if cond == "casc" else gold_seed
                    if seed is None:
                        n_skip += 1
                        rows.append({"id": qid, "question": q, "entities": [],
                                     "answers": answers,
                                     "subgraph": {"entities": [], "tuples": []}})
                        continue
                    n_link += 1
                    rows.append({"id": qid, "question": q, "entities": [seed],
                                 "answers": answers, "subgraph": R(seed)})
                meta = {"n_total": len(items), "n_linked": n_link,
                        "n_skipped": n_skip, "condition": cond, "variant": var}
                write_folder(DATA_ROOT / f"metaqa-{hop}hop-{cond}-{tagbase}",
                             src, rows, meta)


if __name__ == "__main__":
    main()