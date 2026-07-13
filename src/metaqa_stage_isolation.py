#!/usr/bin/env python3
"""
metaqa_stage_isolation.py : stage-resolved robustness metrics for MetaQA CR/RS,
mirroring the paper's protocol (Delta_EL / Delta_SR / alpha_pert; App. A):

Stage 1  Entity Linking. MetaQA has no learned linker; linking is entity-surface
         matching against the KB lexicon (standard for WikiMovies pipelines).
         SeedHit(q) = topic-entity name appears (case-insensitive) in the text.
         Delta_EL = SeedHit(clean) - SeedHit(pert).
Stage 2  Subgraph Retrieval. MetaQA retrieval is a deterministic seed
         neighborhood (no PPR, no question conditioning). Linked -> identical
         subgraph (Jaccard 1, Delta_SR_i = 0); unlinked -> no subgraph
         (Delta_SR_i = 1, answer absent). alpha_pert_i = gold answer present in
         the retrieved subgraph.
Stage 3  Composition with eval results (read from the eval logs separately):
         EM_full ~= EM_fast x linked_rate  (unlinked questions cannot be
         answered); delta_SG = EM_fast - EM_full; delta_DEC = EM_ctrl - EM_fast.

Usage: python metaqa_stage_isolation.py --hops 3 2 --sets p4 p7 [--full]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metaqa_es_p1 import load_union_kb  # union-KB adjacency (no Freebase needed)

import os
_REPO_ROOT = Path(os.environ.get("CASCADEATTACK_ROOT",
                                 Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("METAQA_DATA_ROOT",
                                _REPO_ROOT / "data" / "metaqa"))
PERT_DIR = Path(os.environ.get("PERT_DATASET",
                               _REPO_ROOT / "data" / "perturbations"))
RETRIEVAL_CAP = 500


def link(question: str, name2id: dict, max_len: int) -> int | None:
    """MetaQA surface linker: longest word-n-gram of the question that is an
    entity name in the lexicon (case-insensitive). Returns entity id or None."""
    toks = re.findall(r"\S+", question.lower())
    for n in range(min(max_len, len(toks)), 0, -1):
        for i in range(len(toks) - n + 1):
            cand = " ".join(toks[i:i + n]).strip(",.?!'\"")
            if cand in name2id:
                return name2id[cand]
    return None


def retrieve(seed: int, hop: int, fwd, bwd, rel_ids) -> tuple[set, set]:
    """Deterministic seed-only neighborhood retrieval (BFS, entity cap) : the
    question-independent MetaQA retrieval operator. Same function applied to the
    clean-linked and perturbed-linked seed, so DSR compares like with like."""
    ents, tuples, frontier = {seed}, set(), {seed}
    for _ in range(hop):
        nxt = set()
        for e in sorted(frontier):
            if len(ents) >= RETRIEVAL_CAP:
                break
            for r in rel_ids:
                for o in sorted(fwd.get((e, r), set())):
                    if o in ents:
                        tuples.add((e, r, o))
                    elif len(ents) < RETRIEVAL_CAP:
                        tuples.add((e, r, o)); ents.add(o); nxt.add(o)
                for s in sorted(bwd.get((e, r), set())):
                    if s in ents:
                        tuples.add((s, r, e))
                    elif len(ents) < RETRIEVAL_CAP:
                        tuples.add((s, r, e)); ents.add(s); nxt.add(s)
        frontier = nxt
        if len(ents) >= RETRIEVAL_CAP:
            break
    return ents, tuples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", nargs="+", type=int, default=[3, 2])
    ap.add_argument("--sets", nargs="+", default=["p4", "p7"])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--relink", action="store_true",
                    help="EMPIRICAL mode: re-link entities from the text with the "
                         "longest-match lexicon linker and re-run seed-neighborhood "
                         "retrieval from the union KB for clean AND perturbed "
                         "questions independently; measure Jaccard/alpha_pert.")
    ap.add_argument("--out", default=str(PERT_DIR / "metaqa_stage_isolation.json"))
    args = ap.parse_args()

    results = {}
    for hop in args.hops:
        src = DATA_ROOT / f"metaqa-{hop}hop"
        names = src.joinpath("entities.txt").read_text().splitlines()
        if args.relink:
            fwd, bwd, out_rels, triples = load_union_kb(src)
            rel_ids = sorted({r for (_, r, _) in triples})
            name2id = {}
            for i, nm in enumerate(names):
                name2id.setdefault(nm.lower(), i)
            max_name_len = max(len(nm.split()) for nm in names)
            rcache = {}

            def R(seed):
                if seed not in rcache:
                    rcache[seed] = retrieve(seed, hop, fwd, bwd, rel_ids)
                return rcache[seed]
        orig = {}
        with open(src / "test_simple.json") as f:
            for line in f:
                r = json.loads(line)
                orig[r["id"]] = r
        for set_id in args.sets:
            suffix = "_full" if args.full else ""
            path = PERT_DIR / f"metaqa{hop}hop_{set_id}{suffix}.json"
            if not path.exists():
                print(f"skip missing {path}")
                continue
            pert = json.load(open(path))
            n = len(pert)

            if args.relink:
                link_c = link_p = same_seed = 0
                jac_sum = 0.0
                a_clean = a_pert = 0
                for p in pert:
                    o = orig[p["id"]]
                    gold_seed = o["entities"][0]
                    gold = {a["kb_id"] for a in o["answers"]}
                    sc = link(o["question"], name2id, max_name_len)
                    sp = link(p["perturbed_question"], name2id, max_name_len)
                    link_c += (sc == gold_seed)
                    link_p += (sp == gold_seed)
                    same_seed += (sc == sp and sc is not None)
                    ec, tc = R(sc) if sc is not None else (set(), set())
                    ep, tp = R(sp) if sp is not None else (set(), set())
                    union = tc | tp
                    jac_sum += (len(tc & tp) / len(union)) if union else 0.0
                    a_clean += bool(gold & ec)
                    a_pert += bool(gold & ep)
                res = {
                    "n": n, "mode": "relink+re-retrieve (empirical)",
                    "LinkAcc_clean": round(100 * link_c / n, 2),
                    "LinkAcc_pert": round(100 * link_p / n, 2),
                    "Delta_EL_pp": round(100 * (link_c - link_p) / n, 2),
                    "SameSeed_pct": round(100 * same_seed / n, 2),
                    "Jaccard_mean": round(jac_sum / n, 4),
                    "Delta_SR": round(1 - jac_sum / n, 4),
                    "alpha_clean_pct": round(100 * a_clean / n, 2),
                    "alpha_pert_pct": round(100 * a_pert / n, 2),
                }
                results[f"{hop}hop_{set_id}{suffix}_relink"] = res
                print(f"[{hop}hop {set_id}{suffix} RELINK] n={n} "
                      f"LinkAcc {res['LinkAcc_clean']}->{res['LinkAcc_pert']} "
                      f"(dEL {res['Delta_EL_pp']}pp) sameSeed={res['SameSeed_pct']}% "
                      f"J={res['Jaccard_mean']} dSR={res['Delta_SR']} "
                      f"alpha {res['alpha_clean_pct']}->{res['alpha_pert_pct']}")
                continue

            seed_clean = seed_pert = 0
            jac_sum = 0.0
            alpha_clean = alpha_pert = 0
            linked_ids = []
            for p in pert:
                o = orig[p["id"]]
                ename = names[o["entities"][0]].lower()
                q0, q1 = o["question"].lower(), p["perturbed_question"].lower()
                s0, s1 = ename in q0, ename in q1
                seed_clean += s0
                seed_pert += s1
                ents = set(o["subgraph"]["entities"])
                gold = {a["kb_id"] for a in o["answers"]}
                present = bool(gold & ents)
                alpha_clean += present
                if s1:  # linked -> identical neighborhood subgraph
                    jac_sum += 1.0
                    alpha_pert += present
                    linked_ids.append(p["id"])
                # unlinked: Jaccard 0, answer absent
            res = {
                "n": n,
                "SeedHit_clean": round(100 * seed_clean / n, 2),
                "SeedHit_pert": round(100 * seed_pert / n, 2),
                "Delta_EL_pp": round(100 * (seed_clean - seed_pert) / n, 2),
                "Delta_SR": round(1 - jac_sum / n, 4),
                "alpha_clean_pct": round(100 * alpha_clean / n, 2),
                "alpha_pert_pct": round(100 * alpha_pert / n, 2),
                "linked_rate": round(seed_pert / n, 4),
            }
            results[f"{hop}hop_{set_id}{suffix}"] = res
            print(f"[{hop}hop {set_id}{suffix}] n={n} "
                  f"SeedHit {res['SeedHit_clean']}->{res['SeedHit_pert']} "
                  f"(dEL {res['Delta_EL_pp']}pp) dSR={res['Delta_SR']} "
                  f"alpha {res['alpha_clean_pct']}->{res['alpha_pert_pct']}")
    json.dump(results, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
