#!/usr/bin/env python3
"""
metaqa_perturb_crrs.py : CR (p4) + RS (p7) perturbation sets for MetaQA 2-hop/3-hop.

Generates via a local vLLM OpenAI server (Llama-3.3-70B-Instruct, GPUs 2+3, port 8010).
Prompts: tmlr/md/prompt.md "P5: Compositional Perturbation" (=CR/p4) and
"P6: Relation Substitution" (=RS/p7), aligned with the released CWQ/WebQSP sets
(final mapping: p4=CR, p7=RS).

Validity gate per attempt (mirrors the pipeline's pplr2p5-bert85 variant):
  single line · all topic-entity surfaces preserved · changed vs original ·
  BERTScore(q,q') >= 0.85 · PPL(q')/PPL(q) <= 2.5 (PPL from the 70B via echo logprobs).
attempts=4, then fallback to original (is_perturbed=False) : survivorship design
identical to the paper (reported severity = conservative lower bound).

MetaQA note: CR/RS preserve entities+answers by construction; the eval reuses the
per-question NSM subgraph (fixed-subgraph isolation, analogous to the RS-ELQ /
fast-mode settings on Freebase). No SPARQL/denotation stage exists or is needed.

Usage (after vLLM server is up):
  python metaqa_perturb_crrs.py \
      --hops 3 2 --sets p4 p7 --limit 3000 --scorer-gpu 0
Smoke:
  ... --limit 20 --out-suffix _smoke
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os
_REPO_ROOT = Path(os.environ.get("CASCADEATTACK_ROOT",
                                 Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("METAQA_DATA_ROOT",
                                _REPO_ROOT / "data" / "metaqa"))
OUT_DIR = Path(os.environ.get("PERT_DATASET",
                              _REPO_ROOT / "data" / "perturbations"))

CR_SYSTEM = """Apply EXACTLY ONE of the following operations:
1) Hop-order phrasing shift (reorder multi-hop reasoning steps linguistically)
2) Reorder the wording of the question ONLY.

STRICT RULES:
- Do NOT change predicates or relations
- Do NOT add or remove entities
- Do NOT introduce new constraints
- Do NOT add extra temporal constraints
- Do NOT paraphrase or substitute words
- Keep semantics and answer exactly the same

ALLOWED:
- Clause reordering
- WH-position shifts
- Minor syntactic rearrangement

Example:
Original Question: Which country is the birthplace of the CEO of Tesla?
Perturbed Question: The CEO of Tesla was born in which country?

Original Question: Which city is the birthplace of the founder of Microsoft?
Perturbed Question: The birthplace of the founder of Microsoft is which city?

Return one single-line question only. No explanation, no quotes."""

RS_SYSTEM = """You are an adversarial question rewriter for KG-QA.

Task:
Rewrite the question by substituting ONLY the relation (predicate)
with a synonymous or equivalent expression.

STRICT RULES:
- Preserve exact meaning, answer, and reasoning path
- Do NOT change entities
- Do NOT add or remove constraints
- Do NOT introduce temporal or contextual modifiers
- The rewritten relation must map to the SAME underlying KG predicate
- If no safe substitution exists, return the original question unchanged

Example:
Original Question: What does jamaican people speak?
Perturbed Question: Which language is spoken by Jamaican people?

Original Question: Where was Barack Obama born?
Perturbed Question: What is the birthplace of Barack Obama?

Return ONLY one single-line question. No explanation."""

S1_SYSTEM = """Rewrite the question using a synonym or paraphrase for the main \
predicate phrase. You may also rephrase entity descriptions but do not change \
any proper noun entity names or the correct answer. Output one rewritten question only."""

S2_SYSTEM = """Convert the question between active and passive voice, or vice versa, \
while keeping the meaning, all entities, and the correct answer exactly the same. \
Output one rewritten question only."""

S3_SYSTEM = """Insert one plausible but incorrect entity into the question as a \
distractor, without changing the correct entity or the answer. The inserted \
entity must be from the same domain as the topic entity. Output the rewritten \
question only."""

S4_SYSTEM = """Apply minor surface-form variation to the question: synonym \
substitution, word order change, or light paraphrase. Do not change entity \
names or the correct answer. Output one rewritten question only."""


def _answers_hint(r):
    return "; ".join(a["text"] for a in r["answers"][:5]) or "unknown"


# user-message builders per set (S1-S4 use the paper's Appendix H [USER] templates)
SETS = {
    "p2": ("S1", S1_SYSTEM,
           lambda r: f"Question: {r['question']}\nAnswer: {_answers_hint(r)}"),
    "p3": ("S2", S2_SYSTEM,
           lambda r: f"Question: {r['question']}\nAnswer: {_answers_hint(r)}"),
    "p4": ("CR", CR_SYSTEM,
           lambda r: f"Question: {r['question']}\nOperation: Apply one compositional-safe transformation."),
    "p5": ("S3", S3_SYSTEM,
           lambda r: f"Question: {r['question']}\nTopic entity: {', '.join(r['entity_names'])}\nAnswer: {_answers_hint(r)}"),
    "p6": ("S4", S4_SYSTEM,
           lambda r: f"Question: {r['question']}\nAnswer: {_answers_hint(r)}"),
    "p7": ("RS", RS_SYSTEM,
           lambda r: f"Question: {r['question']}\nOperation: relation substitution only."),
}


def clean_text(text: str) -> str:
    text = text.strip().split("\n")[0]
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(question|rewritten question|perturbed question|output)\s*:\s*",
                  "", text, flags=re.IGNORECASE)
    return text.strip(" \"'")


def load_metaqa(hop: int, limit: int, seed: int):
    folder = DATA_ROOT / f"metaqa-{hop}hop"
    ents = folder.joinpath("entities.txt").read_text().splitlines()
    records = []
    with open(folder / "test_simple.json") as f:
        for line in f:
            r = json.loads(line)
            records.append({
                "id": r["id"],
                "question": r["question"],
                "entities": r["entities"],
                "entity_names": [ents[e] for e in r["entities"]],
                "answers": r["answers"],
            })
    if 0 < limit < len(records):
        rng = random.Random(seed)
        records = rng.sample(records, limit)
        records.sort(key=lambda r: int(r["id"].split("_")[-1]))
    return records


class Client:
    def __init__(self, api_base: str, model: str, workers: int):
        from openai import OpenAI
        self.cli = OpenAI(base_url=api_base, api_key="EMPTY", timeout=180)
        self.model = model
        self.workers = workers

    def gen_batch(self, jobs):
        """jobs: list of (system, user). Returns list of cleaned strings."""
        def one(su):
            for _ in range(3):
                try:
                    r = self.cli.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": su[0]},
                                  {"role": "user", "content": su[1]}],
                        temperature=0.7, top_p=0.9, max_tokens=64)
                    return clean_text(r.choices[0].message.content or "")
                except Exception:
                    time.sleep(5)
            return ""
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(one, jobs))

    def ppl_batch(self, texts):
        """Perplexity via echo=True prompt logprobs on the completions API."""
        def one(t):
            if not t:
                return float("inf")
            for _ in range(3):
                try:
                    r = self.cli.completions.create(
                        model=self.model, prompt=t, max_tokens=1,
                        echo=True, logprobs=0, temperature=0.0)
                    lps = [x for x in (r.choices[0].logprobs.token_logprobs or [])
                           if x is not None]
                    if not lps:
                        return float("inf")
                    return math.exp(-sum(lps) / len(lps))
                except Exception:
                    time.sleep(5)
            return float("inf")
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(one, texts))


def bertscore_batch(cands, refs, device):
    from bert_score import score as bs
    if not cands:
        return []
    P, R, F1 = bs(cands, refs, lang="en", device=device,
                  batch_size=64, verbose=False)
    return [float(x) for x in F1]


def main():
    global DATA_ROOT, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", nargs="+", type=int, default=[3, 2])
    ap.add_argument("--sets", nargs="+", default=["p4", "p7"], choices=list(SETS))
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--api-base", default="http://localhost:8010/v1")
    ap.add_argument("--model", default="llama-3.3-70b")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--bertscore-threshold", type=float, default=0.85)
    ap.add_argument("--pplr-threshold", type=float, default=2.5)
    ap.add_argument("--scorer-gpu", default="0", help="physical GPU for BERTScore (cuda:N) or 'cpu'")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--full-merge", action="store_true",
                    help="Phase 2: generate only ids NOT already in metaqa{hop}hop_{set}.json, "
                         "merge with those results, write *_full.json (ignores --limit).")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    DATA_ROOT = Path(args.data_root)
    OUT_DIR = Path(args.out_dir)

    device = "cpu" if args.scorer_gpu == "cpu" else f"cuda:{args.scorer_gpu}"
    cli = Client(args.api_base, args.model, args.workers)

    for hop in args.hops:
        all_records = load_metaqa(hop, 0 if args.full_merge else args.limit, args.seed)
        print(f"[metaqa-{hop}hop] {len(all_records)} questions loaded"
              + ("" if args.full_merge else f" (sampled, seed {args.seed})"))
        ppl_cache = {}

        for set_id in args.sets:
            name, system, user_fn = SETS[set_id]
            prior = []
            records = all_records
            if args.full_merge:
                base_path = OUT_DIR / f"metaqa{hop}hop_{set_id}.json"
                if base_path.exists():
                    prior = json.load(open(base_path))
                done_ids = {p["id"] for p in prior}
                records = [r for r in all_records if r["id"] not in done_ids]
                print(f"  [{set_id} hop{hop}] full-merge: {len(prior)} prior, "
                      f"{len(records)} remaining to generate")
            missing = [r["question"] for r in records if r["question"] not in ppl_cache]
            for q, v in zip(missing, cli.ppl_batch(missing)):
                ppl_cache[q] = v
            ppl_orig = [ppl_cache[r["question"]] for r in records]
            out, pending = {}, list(range(len(records)))
            for attempt in range(1, args.attempts + 1):
                if not pending:
                    break
                jobs = [(system, user_fn(records[i])) for i in pending]
                cands = cli.gen_batch(jobs)
                f1s = bertscore_batch(cands,
                                      [records[i]["question"] for i in pending],
                                      device)
                ppls = cli.ppl_batch(cands)
                still = []
                for k, i in enumerate(pending):
                    r, q2 = records[i], cands[k]
                    ratio = (ppls[k] / ppl_orig[i]) if ppl_orig[i] > 0 else float("inf")
                    ok = (bool(q2)
                          and q2.lower() != r["question"].lower()
                          and all(en.lower() in q2.lower() for en in r["entity_names"])
                          and f1s[k] >= args.bertscore_threshold
                          and ratio <= args.pplr_threshold)
                    if ok:
                        out[i] = (q2, True, {"bertscore": round(f1s[k], 4),
                                             "ppl_orig": round(ppl_orig[i], 2),
                                             "ppl_pert": round(ppls[k], 2),
                                             "ppl_ratio": round(ratio, 3),
                                             "attempts_used": attempt,
                                             "reason": "pass"})
                    else:
                        still.append(i)
                pending = still
                print(f"  [{set_id} hop{hop}] attempt {attempt}: "
                      f"{len(out)}/{len(records)} passed, {len(pending)} pending")
            for i in pending:
                out[i] = (records[i]["question"], False,
                          {"attempts_used": args.attempts, "reason": "fallback_original"})

            data = []
            for i, r in enumerate(records):
                q2, pert, meta = out[i]
                meta["perturbation_type"] = set_id
                data.append({"id": r["id"],
                             "original_question": r["question"],
                             "perturbed_question": q2,
                             "entities": r["entities"],
                             "entity_names": r["entity_names"],
                             "answers": r["answers"],
                             "perturbation_type": set_id,
                             "perturbation_name": name,
                             "is_perturbed": pert,
                             "perturbation_meta": meta})
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            if args.full_merge:
                data = prior + data
                data.sort(key=lambda d: int(d["id"].split("_")[-1]))
                out_path = OUT_DIR / f"metaqa{hop}hop_{set_id}_full{args.out_suffix}.json"
            else:
                out_path = OUT_DIR / f"metaqa{hop}hop_{set_id}{args.out_suffix}.json"
            json.dump(data, open(out_path, "w"), indent=1)
            n_ok = sum(d["is_perturbed"] for d in data)
            print(f"  [{set_id} hop{hop}] DONE: {n_ok}/{len(data)} perturbed "
                  f"({100*n_ok/len(data):.1f}% pass) -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())