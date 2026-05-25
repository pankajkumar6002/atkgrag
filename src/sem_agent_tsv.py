#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
sem_agent_tsv.py — Semantic entity linker via 16M TSV trigram search.

Three mention-extraction methods:
  Method A (elq):  use ELQ v2 output mention spans → search 16M TSV
  Method B (ner):  BLINK Flair NER on raw question text → search 16M TSV
  Method C (llm):  Llama-3.3-70B prompted to extract entity → search 16M TSV
                   Requires 2× A100-SXM4-80GB (140 GB VRAM for fp16 weights)
                   Model: tmlr/models/tmlr/models/Llama-3.3-70B-Instruct/...

For each mention, trigram-cosine search over entities_sup_names.tsv (16M entries)
gives the closest name → assigns its MID → evaluate vs GT.

Usage:
  python sem_agent_tsv.py --all --method elq
  python sem_agent_tsv.py --all --method ner
  python sem_agent_tsv.py --all --method both   # default (elq+ner)
  python sem_agent_tsv.py --all --method llm    # Llama 70B mention extraction
  python sem_agent_tsv.py --all --method all    # elq + ner + llm
  python sem_agent_tsv.py --dataset cwq --p-type p1 --method both
  python sem_agent_tsv.py --dataset cwq --p-type p1 --method llm --cuda 0,1
"""
from __future__ import annotations

import array
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = REPO_ROOT
DATASET   = REPO_ROOT / "pert" / "dataset"
RESULTS   = REPO_ROOT / "results"
RESOURCES = REPO_ROOT / "resources"
GNN_DATA  = REPO_ROOT / "external" / "GNN-RAG" / "data"
NAMES_TSV = RESOURCES / "entities_sup_names.tsv"
RESULTS.mkdir(exist_ok=True)

_llm_cuda = "0,1"   # overridden by --cuda CLI argument

LLAMA70B_PATH = (
    REPO_ROOT / "models" / "Llama-3.3-70B-Instruct"
    / "models--meta-llama--Llama-3.3-70B-Instruct"
    / "snapshots"
    / "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
)

# ── LLM (Llama 70B) mention extraction ───────────────────────────────────────
_llm_model = None
_llm_tokenizer = None

def get_llm_model(cuda: str = "0,1"):
    """Load Llama-3.3-70B-Instruct for entity mention extraction."""
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return _llm_model, _llm_tokenizer
    import os, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = str(LLAMA70B_PATH)
    print(f"Loading Llama-3.3-70B from {model_path}…", flush=True)
    print(f"  CUDA devices: {cuda}", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

    _llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
    _llm_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _llm_model.eval()
    print("  Llama-70B loaded.", flush=True)
    return _llm_model, _llm_tokenizer


_LLM_PROMPT = """\
You are a named entity extractor. Given a question, output ONLY the main entity \
names separated by commas. Do not include any other text.

Question: {question}
Entities:"""


def extract_mentions_llm(text: str, cuda: str = "0,1") -> List[str]:
    """Use Llama-3.3-70B to extract entity mentions from a question."""
    import torch
    model, tokenizer = get_llm_model(cuda)
    prompt = _LLM_PROMPT.format(question=text.strip())
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Parse comma-separated entity names; stop at newline
    raw = raw.split("\n")[0].strip()
    mentions = [m.strip() for m in raw.split(",") if m.strip()]
    return mentions


# ── NER (BLINK / Flair) ───────────────────────────────────────────────────────
_ner_model = None

def get_ner_model():
    global _ner_model
    if _ner_model is not None:
        return _ner_model
    # Load Flair SequenceTagger directly (BLINK wrapper re-wraps Sentence, broken in Flair 0.15+)
    from flair.models import SequenceTagger
    print("Loading Flair NER model (conll03-english)…", flush=True)
    _ner_model = SequenceTagger.load("ner")   # uses cached dbmdz/bert-large-cased-finetuned-conll03
    print("  NER model loaded.", flush=True)
    return _ner_model


def extract_mentions_ner(text: str) -> List[str]:
    """Extract entity mentions using Flair SequenceTagger directly (bypasses BLINK wrapper)."""
    model = get_ner_model()
    from flair.data import Sentence as FlairSentence
    sentence = FlairSentence(text)
    model.predict(sentence)
    spans = []
    for entity in sentence.get_spans("ner"):
        t = entity.text.strip()
        if t:
            spans.append(t)
    return spans


# ── Trigram index ─────────────────────────────────────────────────────────────

def char_trigrams(text: str) -> Dict[str, int]:
    text = text.lower().strip()
    ng: Dict[str, int] = {}
    padded = f"  {text}  "
    for i in range(len(padded) - 2):
        k = padded[i:i+3]
        ng[k] = ng.get(k, 0) + 1
    return ng


def cosine_sim(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot   = sum(a[k] * b.get(k, 0) for k in a)
    na    = sum(v*v for v in a.values()) ** 0.5
    nb    = sum(v*v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class TSVTrigramIndex:
    def __init__(self) -> None:
        self.mids:     List[str] = []
        self.names:    List[str] = []
        self.inverted: Dict[str, array.array] = {}
        self._built = False

    def build(self, tsv_path: Path) -> None:
        t0 = time.time()
        print(f"Building trigram index from {tsv_path.name}…", flush=True)
        inv: Dict[str, array.array] = defaultdict(lambda: array.array("i"))
        mids: List[str] = []
        names: List[str] = []
        with open(tsv_path, encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(fh):
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                mid, name = parts
                name_lc = name.lower().strip()
                mids.append(mid)
                names.append(name_lc)
                padded = f"  {name_lc}  "
                seen: set = set()
                for j in range(len(padded) - 2):
                    ng = padded[j:j+3]
                    if ng not in seen:
                        seen.add(ng)
                        inv[ng].append(idx)
                if idx % 2_000_000 == 0 and idx > 0:
                    print(f"  {idx:,} entries… ({time.time()-t0:.0f}s)", flush=True)
        self.mids     = mids
        self.names    = names
        self.inverted = dict(inv)
        self._built   = True
        print(f"  Index ready: {len(mids):,} entries, {len(self.inverted):,} trigrams "
              f"({time.time()-t0:.1f}s)", flush=True)

    def search(self, mention: str, top_k: int = 1) -> List[Tuple[str, str, float]]:
        """Return [(mid, name, score)] top-k by trigram cosine."""
        if not self._built or not mention.strip():
            return []
        m_tg = char_trigrams(mention)
        if not m_tg:
            return []
        cand: Dict[int, int] = defaultdict(int)
        for ng in m_tg:
            for idx in self.inverted.get(ng, array.array("i")):
                cand[idx] += 1
        min_overlap = max(1, len(m_tg) // 4)
        top = sorted(((cnt, i) for i, cnt in cand.items() if cnt >= min_overlap),
                     reverse=True)[:200]
        scored = []
        for _, i in top:
            s = cosine_sim(m_tg, char_trigrams(self.names[i]))
            if s > 0:
                scored.append((s, self.mids[i], self.names[i]))
        scored.sort(reverse=True)
        return [(mid, nm, s) for s, mid, nm in scored[:top_k]]


# ── Data loading ──────────────────────────────────────────────────────────────

_id2mid_cache: Dict[str, Dict[int, str]] = {}

def load_id2mid(dataset: str) -> Dict[int, str]:
    if dataset not in _id2mid_cache:
        subdir = "CWQ" if dataset == "cwq" else "webqsp"
        with open(GNN_DATA / subdir / "entities.txt") as f:
            ents = [l.strip() for l in f if l.strip()]
        _id2mid_cache[dataset] = {i: e for i, e in enumerate(ents)}
    return _id2mid_cache[dataset]


def load_el(dataset: str, p_type: str) -> Optional[List[Dict]]:
    path = DATASET / "el_elq" / f"{dataset}_{p_type}_el.json"
    if not path.exists():
        return None
    return json.load(open(path))


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_one(
    dataset: str,
    p_type:  str,
    index:   TSVTrigramIndex,
    method:  str,   # "elq" | "ner" | "both"
    name2mid: Dict[str, str],
    sample: Optional[int] = None,
) -> Optional[Dict]:

    id2mid = load_id2mid(dataset)

    # GT from orig
    orig_records = json.load(open(DATASET / f"{dataset}_orig.json"))
    gt_map: Dict[str, Set[str]] = {
        r["id"]: {id2mid[i] for i in r.get("gnn_entities", []) if i in id2mid}
        for r in orig_records
    }

    # Dataset (question text)
    ds_records = json.load(open(DATASET / f"{dataset}_{p_type}.json"))
    q_by_id = {r["id"]: r for r in ds_records}

    # ELQ output (for method A spans and baseline exact metric)
    el_records = load_el(dataset, p_type)
    if el_records is None:
        print(f"  [SKIP] ELQ missing for {dataset} {p_type}")
        return None
    el_by_id = {r["id"]: r for r in el_records}

    if method == "both":
        methods = ["elq", "ner"]
    elif method == "all":
        methods = ["elq", "ner", "llm"]
    else:
        methods = [method]
    stats: Dict[str, Dict] = {
        m: {"n": 0, "elq_exact": 0, "tsv_exact": 0,
            "tsv_soft": 0, "tsv_partial": 0, "tsv_far": 0,
            "tsv_recov": 0, "sims": [], "no_mention": 0}
        for m in methods
    }
    THR = 0.6

    if sample is not None:
        ds_records = ds_records[:sample]

    for rec in ds_records:
        qid      = rec["id"]
        gt_mids  = gt_map.get(qid, set())
        if not gt_mids:
            continue
        el_rec   = el_by_id.get(qid, {})
        linked   = el_rec.get("linked_entities", [])
        pred_mids_elq = {e.get("mid", "") for e in linked if e.get("mid")}
        elq_exact = bool(pred_mids_elq & gt_mids)

        for m in methods:
            s = stats[m]

            # --- Extract mention spans ---
            if m == "elq":
                mentions = list({e.get("mention", "") for e in linked
                                 if e.get("mention")})
            elif m == "llm":
                q_text = rec.get("perturbed_question") or rec.get("original_question", "")
                mentions = extract_mentions_llm(q_text, cuda=_llm_cuda)
            else:  # ner
                q_text = rec.get("perturbed_question") or rec.get("original_question", "")
                mentions = extract_mentions_ner(q_text)

            if not mentions:
                s["no_mention"] += 1
                continue

            # --- TSV search for best MID ---
            best_sim  = 0.0
            best_mid  = ""
            for mention in mentions:
                hits = index.search(mention, top_k=1)
                if hits:
                    tmid, _, tsim = hits[0]
                    if tsim > best_sim:
                        best_sim, best_mid = tsim, tmid

            tsv_exact = best_mid in gt_mids

            s["n"]        += 1
            s["sims"].append(best_sim)
            if elq_exact:
                s["elq_exact"] += 1
            if tsv_exact:
                s["tsv_exact"] += 1
                s["tsv_recov"] += 1
            elif best_sim >= THR:
                s["tsv_soft"]  += 1
                s["tsv_recov"] += 1
            elif best_sim >= 0.3:
                s["tsv_partial"] += 1
            else:
                s["tsv_far"] += 1

    results = {}
    for m in methods:
        s = stats[m]
        n = max(1, s["n"])
        results[m] = {
            "dataset":      dataset,
            "p_type":       p_type,
            "method":       m,
            "n":            s["n"],
            "elq_exact_pct":  round(s["elq_exact"]  / n * 100, 2),
            "tsv_exact_pct":  round(s["tsv_exact"]  / n * 100, 2),
            "tsv_soft_pct":   round(s["tsv_soft"]   / n * 100, 2),
            "tsv_partial_pct":round(s["tsv_partial"] / n * 100, 2),
            "tsv_far_pct":    round(s["tsv_far"]    / n * 100, 2),
            "tsv_recov_pct":  round(s["tsv_recov"]  / n * 100, 2),
            "tsv_mean_sim":   round(float(np.mean(s["sims"])) if s["sims"] else 0.0, 4),
            "no_mention":  s["no_mention"],
        }
        r = results[m]
        print(f"    [{m}] n={r['n']}  ELQ_exact={r['elq_exact_pct']}%  "
              f"TSV_exact={r['tsv_exact_pct']}%  recov={r['tsv_recov_pct']}%  "
              f"mean_sim={r['tsv_mean_sim']:.3f}", flush=True)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    global _llm_cuda
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  choices=["cwq", "webqsp"])
    p.add_argument("--p-type",   choices=["orig","p1","p2","p3","p4","p5","p6"])
    p.add_argument("--all",      action="store_true")
    p.add_argument("--method",   default="both",
                   choices=["elq", "ner", "both", "llm", "all"])
    p.add_argument("--sample",    type=int, default=None,
                   help="Limit evaluation to first N questions per combo (for quick checks)")
    p.add_argument("--cuda",      default="0,1",
                   help="CUDA device IDs for Llama-70B (default: 0,1). Only used with --method llm/all")
    p.add_argument("--out-file",  default=None,
                   help="Override output JSON path (default: tmlr/results/sem_agent_tsv_results.json). "
                        "Use separate files per parallel session to avoid write races; merge with --merge.")
    p.add_argument("--merge",     nargs="+", metavar="JSON",
                   help="Merge listed JSON result files into the main results file and exit.")
    args = p.parse_args()
    _llm_cuda = args.cuda

    # ── Merge mode ────────────────────────────────────────────────────────────
    if args.merge:
        main_out = RESULTS / "sem_agent_tsv_results.json"
        merged: Dict = {}
        if main_out.exists():
            merged = json.load(open(main_out))
        for src in args.merge:
            src_path = Path(src)
            if src_path.exists():
                merged.update(json.load(open(src_path)))
                print(f"  merged {src_path.name}")
            else:
                print(f"  [SKIP] {src_path} not found")
        json.dump(merged, open(main_out, "w"), indent=2)
        print(f"Merged {len(args.merge)} files → {main_out}  ({len(merged)} total entries)")
        return

    # Build index
    index = TSVTrigramIndex()
    index.build(NAMES_TSV)

    # Load name2mid (for reference, not used directly in evaluation)
    name2mid: Dict[str, str] = {}

    # Determine combos
    combos: List[Tuple[str, str]] = []
    if args.all:
        for ds in ["cwq", "webqsp"]:
            for pt in ["orig", "p1", "p2", "p3", "p4", "p5", "p6"]:
                combos.append((ds, pt))
    else:
        if not args.dataset or not args.p_type:
            p.error("Specify --dataset + --p-type, or --all")
        combos = [(args.dataset, args.p_type)]

    all_results: Dict[str, Dict] = {}
    for dataset, p_type in combos:
        print(f"\n── {dataset} {p_type} ──", flush=True)
        r = evaluate_one(dataset, p_type, index, args.method, name2mid, sample=args.sample)
        if r:
            for method_key, res in r.items():
                all_results[f"{dataset}_{p_type}_{method_key}"] = res

    # Save
    out = Path(args.out_file) if args.out_file else (RESULTS / "sem_agent_tsv_results.json")
    # Merge with existing if present
    existing: Dict = {}
    if out.exists():
        existing = json.load(open(out))
    existing.update(all_results)
    json.dump(existing, open(out, "w"), indent=2)
    print(f"\nSaved {len(all_results)} entries → {out}", flush=True)

    # Print table
    print(f"\n{'='*90}")
    print(f"  sem_agent TSV results  (method={args.method})")
    print(f"{'='*90}")
    print(f"  {'DS':<8} {'PT':<5} {'M':<4} {'n':>5}  {'ELQ%':>6}  {'TSV%':>6}  "
          f"{'soft%':>6}  {'recov%':>7}  {'sim':>6}")
    print("  " + "-"*80)
    for key in sorted(all_results):
        r = all_results[key]
        print(f"  {r['dataset']:<8} {r['p_type']:<5} {r['method']:<4} {r['n']:>5}  "
              f"{r['elq_exact_pct']:>5.1f}%  {r['tsv_exact_pct']:>5.1f}%  "
              f"{r['tsv_soft_pct']:>5.1f}%  {r['tsv_recov_pct']:>6.1f}%  "
              f"{r['tsv_mean_sim']:>6.3f}")


if __name__ == "__main__":
    main()