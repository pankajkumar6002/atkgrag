#!/usr/bin/env python3
"""
metaqa_es_p1.py : ES (p1, Entity Swap) for MetaQA 2-hop/3-hop. Deterministic (no LLM).

Paper semantics (App. E / Fig. 3): substitute the topic entity with a *different*
same-domain entity, producing a new gold answer by design. On MetaQA the entity
surface appears verbatim in the (template-derived) question text, so LLM rewriting
adds nothing: the swap is exact string substitution + KB recomputation.

Pipeline per question:
  1. Union-KB = all subgraph tuples across train/dev/test of the hop (MetaQA is a
     small closed KB; the union of NSM neighborhoods reconstructs it).
  2. Infer the question's relation-path signature sigma (sequence of
     (relation, direction), length <= hop) s.t. endpoints(X, sigma) == gold answers.
  3. Sample same-type X' (type = incident-relation profile) with non-empty
     endpoints(X', sigma); new answers A' = endpoints(X', sigma).
  4. Perturbed question = surface substitution X -> X'.
  5. Rebuild an NSM-style eval subgraph by BFS from X' (entity cap 500, like the
     original data), so the ES set is directly evaluable.
Questions with ambiguous/no path or no viable X' fall back (is_perturbed=False) or
are marked unanswerable : the answerable fraction is reported (paper reports the
CWQ analogue: 45.6% structurally unanswerable).

Usage:
  python metaqa_es_p1.py --hops 3 2 --limit 3000 --seed 13
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ES_SYSTEM = """You are a question rewriter for a KGQA robustness study.
Replace the topic entity with a different entity from the same domain.
Rules: (a) the replacement is given to you and is a valid KB entity with its own
gold answer to the question; (b) do not change the predicate or any other part
of the question; (c) output exactly one rewritten question and nothing else."""

import os
_REPO_ROOT = Path(os.environ.get("CASCADEATTACK_ROOT",
                                 Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("METAQA_DATA_ROOT",
                                _REPO_ROOT / "data" / "metaqa"))
OUT_DIR = Path(os.environ.get("PERT_DATASET",
                              _REPO_ROOT / "data" / "perturbations"))
ENTITY_CAP = 500


def load_union_kb(folder: Path):
    """Union of subgraph tuples over all splits. Returns adjacency + triple set."""
    fwd = defaultdict(set)   # (s, r) -> {o}
    bwd = defaultdict(set)   # (o, r) -> {s}
    out_rels = defaultdict(set)  # entity -> {(r, dir)} incident profile
    triples = set()
    for split in ["train_simple.json", "dev_simple.json", "test_simple.json"]:
        with open(folder / split) as f:
            for line in f:
                rec = json.loads(line)
                for s, r, o in rec["subgraph"]["tuples"]:
                    if (s, r, o) in triples:
                        continue
                    triples.add((s, r, o))
                    fwd[(s, r)].add(o)
                    bwd[(o, r)].add(s)
                    out_rels[s].add((r, "f"))
                    out_rels[o].add((r, "b"))
    return fwd, bwd, out_rels, triples


def endpoints(start, sig, fwd, bwd):
    """Follow signature sig = [(r, dir), ...] from start; return reachable set."""
    frontier = {start}
    for r, d in sig:
        nxt = set()
        for e in frontier:
            nxt |= fwd.get((e, r), set()) if d == "f" else bwd.get((e, r), set())
        if not nxt:
            return set()
        frontier = nxt
    frontier.discard(start)
    return frontier


def infer_signature(x, gold, hop, fwd, bwd, rel_ids):
    """Enumerate signature prefixes depth-first, carrying the frontier set so each
    extension is one adjacency union (no re-walk from scratch). Prefer exact match
    (endpoints == gold), else superset (gold subset of endpoints)."""
    exact, superset = [], []
    stack = [([], {x})]
    while stack:
        sig, frontier = stack.pop()
        if len(sig) == hop:
            continue
        for r in rel_ids:
            for d in ("f", "b"):
                nxt = set()
                for e in frontier:
                    nxt |= fwd.get((e, r), set()) if d == "f" else bwd.get((e, r), set())
                if not nxt:
                    continue
                sig2 = sig + [(r, d)]
                ep = nxt - {x}
                if ep == gold:
                    exact.append(sig2)
                elif gold and gold <= ep:
                    superset.append(sig2)
                stack.append((sig2, nxt))
    if exact:
        exact.sort(key=len)
        return exact[0], ("exact_unique" if len(exact) == 1 else "exact_multi")
    if superset:
        superset.sort(key=len)
        return superset[0], "superset"
    return None, "no_path"


def path_subgraph(start, sig, answers, fwd, bwd, rng):
    """ONE sampled walk per answer along signature sig (keeps the path footprint
    tiny : <= |answers| * len(sig) entities : so the 500 cap is never breached)."""
    levels = [{start}]
    for r, d in sig:
        nxt = set()
        for e in levels[-1]:
            nxt |= fwd.get((e, r), set()) if d == "f" else bwd.get((e, r), set())
        levels.append(nxt)
    ents, tuples = {start}, set()
    for a in answers:
        if a not in levels[-1]:
            continue
        node = a
        walk = [a]
        ok = True
        for i in range(len(sig) - 1, -1, -1):
            r, d = sig[i]
            # predecessors of `node` at level i under step (r, d)
            preds = (bwd.get((node, r), set()) if d == "f"
                     else fwd.get((node, r), set())) & levels[i]
            if not preds:
                ok = False
                break
            node = rng.choice(sorted(preds))
            walk.append(node)
        if not ok or walk[-1] != start:
            continue
        walk.reverse()  # start ... answer
        for i, (r, d) in enumerate(sig):
            s, o = walk[i], walk[i + 1]
            tuples.add((s, r, o) if d == "f" else (o, r, s))
            ents.update((s, o))
    return ents, tuples


def bfs_subgraph(start, hop, fwd, bwd, rng, rel_ids, sig=None, answers=None):
    """NSM-style subgraph: answer-path edges guaranteed first (matching the
    original data's 98.9-100% answer presence), then BFS fill to ENTITY_CAP."""
    ents, tuples = set(), set()
    if sig and answers:
        ents, tuples = path_subgraph(start, sig, set(answers), fwd, bwd, rng)
    ents |= {start}
    frontier = {start}
    for _ in range(hop):
        nxt = set()
        for e in sorted(frontier):
            if len(ents) >= ENTITY_CAP:
                break
            for r in rel_ids:
                for o in sorted(fwd.get((e, r), set())):
                    if o in ents:
                        tuples.add((e, r, o))
                    elif len(ents) < ENTITY_CAP:
                        tuples.add((e, r, o)); ents.add(o); nxt.add(o)
                for s in sorted(bwd.get((e, r), set())):
                    if s in ents:
                        tuples.add((s, r, e))
                    elif len(ents) < ENTITY_CAP:
                        tuples.add((s, r, e)); ents.add(s); nxt.add(s)
        frontier = nxt
        if len(ents) >= ENTITY_CAP:
            break
    return {"entities": sorted(ents), "tuples": sorted(tuples)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", nargs="+", type=int, default=[3, 2])
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--swap-tries", type=int, default=25)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--full-merge", action="store_true",
                    help="generate only ids NOT in metaqa{hop}hop_p1.json, merge, "
                         "write *_full.json (ignores --limit)")
    ap.add_argument("--llm", action="store_true",
                    help="rewrite swapped questions with Llama-3.3-70B (paper Fig. 3 "
                         "ES prompt, candidate provided); substitution is the fallback")
    ap.add_argument("--api-base", default="http://localhost:8010/v1")
    ap.add_argument("--model", default="llama-3.3-70b")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    llm = None
    if args.llm:
        from metaqa_perturb_crrs import Client
        llm = Client(args.api_base, args.model, args.workers)

    for hop in args.hops:
        folder = DATA_ROOT / f"metaqa-{hop}hop"
        names = folder.joinpath("entities.txt").read_text().splitlines()
        print(f"[metaqa-{hop}hop] building union KB ...")
        fwd, bwd, out_rels, triples = load_union_kb(folder)
        rel_ids = sorted({r for (_, r, _) in triples})
        print(f"  union KB: {len(triples)} triples, {len(out_rels)} entities, "
              f"{len(rel_ids)} relations")

        # type profile -> candidate pool
        profile2ents = defaultdict(list)
        for e, prof in out_rels.items():
            profile2ents[frozenset(r for r, _ in prof)].append(e)

        records = []
        with open(folder / "test_simple.json") as f:
            for line in f:
                records.append(json.loads(line))
        rng = random.Random(args.seed)
        prior = []
        if args.full_merge:
            base = OUT_DIR / f"metaqa{hop}hop_p1.json"
            if base.exists():
                prior = json.load(open(base))
            done_ids = {p["id"] for p in prior}
            records = [r for r in records if r["id"] not in done_ids]
            print(f"  full-merge: {len(prior)} prior, {len(records)} remaining")
        elif 0 < args.limit < len(records):
            records = rng.sample(records, args.limit)
            records.sort(key=lambda r: int(r["id"].split("_")[-1]))

        out, stats = [], defaultdict(int)
        for rec in records:
            x = rec["entities"][0]
            gold = {a["kb_id"] for a in rec["answers"]}
            sig, sig_status = infer_signature(x, gold, hop, fwd, bwd, rel_ids)
            entry = {"id": rec["id"],
                     "original_question": rec["question"],
                     "original_entities": rec["entities"],
                     "original_answers": rec["answers"],
                     "perturbation_type": "p1", "perturbation_name": "ES"}
            xname = names[x]
            done = False
            if sig and xname.lower() in rec["question"].lower():
                pool = profile2ents.get(frozenset(r for r, _ in out_rels.get(x, set())), [])
                for _ in range(args.swap_tries):
                    x2 = rng.choice(pool) if pool else None
                    if x2 is None or x2 == x:
                        continue
                    a2 = endpoints(x2, sig, fwd, bwd)
                    x2name = names[x2]
                    if not a2 or x2name.lower() in rec["question"].lower():
                        continue
                    idx = rec["question"].lower().index(xname.lower())
                    q2 = rec["question"][:idx] + x2name + rec["question"][idx + len(xname):]
                    entry.update({
                        "perturbed_question": q2,
                        "entities": [x2],
                        "entity_names": [x2name],
                        "answers": [{"text": names[a], "kb_id": a} for a in sorted(a2)],
                        "is_perturbed": True, "answerable": True,
                        "subgraph": bfs_subgraph(
                            x2, hop, fwd, bwd, rng, rel_ids, sig=sig,
                            answers=rng.sample(sorted(a2), min(20, len(a2)))),
                        "perturbation_meta": {"path_signature": sig,
                                              "sig_status": sig_status,
                                              "swapped_from": xname,
                                              "swapped_to": x2name,
                                              "reason": "pass"}})
                    stats["perturbed"] += 1
                    done = True
                    break
            if not done:
                reason = sig_status if sig is None else (
                    "entity_surface_not_in_question" if xname.lower() not in rec["question"].lower()
                    else "no_viable_swap")
                entry.update({"perturbed_question": rec["question"],
                              "entities": rec["entities"],
                              "entity_names": [xname],
                              "answers": rec["answers"],
                              "is_perturbed": False, "answerable": True,
                              "perturbation_meta": {"sig_status": sig_status,
                                                    "reason": reason}})
                stats[reason] += 1
            out.append(entry)

        if llm is not None:
            idxs = [i for i, e in enumerate(out) if e["is_perturbed"]]
            jobs = []
            for i in idxs:
                e = out[i]
                jobs.append((ES_SYSTEM,
                             f"Question: {e['original_question']}\n"
                             f"Topic entity: {e['perturbation_meta']['swapped_from']}\n"
                             f"Original answer: {'; '.join(a['text'] for a in e['original_answers'][:5])}\n"
                             f"Substitute the topic entity with: {e['perturbation_meta']['swapped_to']}"))
            texts = llm.gen_batch(jobs)
            n_llm = 0
            for k, i in enumerate(idxs):
                e, q2 = out[i], texts[k]
                x_old = e["perturbation_meta"]["swapped_from"].lower()
                x_new = e["perturbation_meta"]["swapped_to"].lower()
                if (q2 and x_new in q2.lower() and x_old not in q2.lower()
                        and q2.lower() != e["original_question"].lower()):
                    e["perturbed_question"] = q2
                    e["perturbation_meta"]["generator"] = "llama-3.3-70b"
                    n_llm += 1
                else:
                    e["perturbation_meta"]["generator"] = "substitution_fallback"
            print(f"  [p1 hop{hop}] LLM rewrite accepted: {n_llm}/{len(idxs)} "
                  f"(rest keep deterministic substitution)")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if args.full_merge:
            out = prior + out
            out.sort(key=lambda d: int(d["id"].split("_")[-1]))
            stats["perturbed"] = sum(d["is_perturbed"] for d in out)
            path = OUT_DIR / f"metaqa{hop}hop_p1_full{args.out_suffix}.json"
        else:
            path = OUT_DIR / f"metaqa{hop}hop_p1{args.out_suffix}.json"
        json.dump(out, open(path, "w"), indent=1)
        n_ok = stats["perturbed"]
        print(f"  [p1 hop{hop}] DONE: {n_ok}/{len(out)} swapped "
              f"({100*n_ok/len(out):.1f}%) -> {path}")
        print(f"  fallback reasons: {dict((k, v) for k, v in stats.items() if k != 'perturbed')}")


if __name__ == "__main__":
    main()