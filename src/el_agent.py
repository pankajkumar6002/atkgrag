#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
el_agent.py — ELA (Entity Linking Agent) using Llama-3-70B-Instruct.

3-step agentic workflow per question:
  Step 1 — Candidate generation:
    Prompt Llama to extract entity mentions + surface-form candidates
  Step 2 — CoT disambiguation:
    Rank candidates using chain-of-thought with Wikipedia label context (id2text.json)
    and Freebase name lookup from entities_sup_names.tsv
  Step 3 — Verification:
    (a) Check if resolved MID is in entities_sup.txt (23.6M entity superset) — fast, no network
    (b) If not in set: SPARQL ASK on localhost:3001 (Freebase is loaded here) as fallback
    Also attempt name-based MID lookup: surface form → entities_sup_names reverse index → MID

entities_sup.txt and entities_sup_names.tsv are built by:
  python tmlr/scripts/build_entities_sup.py

NOTE: Port 3001 hosts the live Freebase Virtuoso instance.
      Port 8890 is an empty Virtuoso shell — do NOT use for entity verification.

Output: data/perturbations/el_agent/ELA_v1/{dataset}_{p_type}_el.json
  Schema: {id, perturbed_question, linked_entities: [{mention, title, mid, score, verified}]}
  Legacy results archived in: data/perturbations/el_agent/ELA_old/

Model path (loaded in cache-style storage after planning):
  tmlr/models/Llama-3-70B-Instruct/   OR  HuggingFace cache (auto-detected)
  Override with --model-path

Usage:
  python el_agent.py --dataset cwq --p-type p1
  python el_agent.py --dataset cwq --p-type p1 --dry-run   # first 5 questions
  python el_agent.py --all --batch-size 4                   # all 12 combinations
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import torch
from SPARQLWrapper import JSON as SPARQL_JSON, SPARQLWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_DIR = REPO_ROOT / "data" / "perturbations"
EL_OUT_DIR  = DATASET_DIR / "el_agent" / "ELA_v1"
ELQ_OUT_DIR = DATASET_DIR / "el_elq"          # ELQ results feed ELA in elq-mode
BLINK_MODELS = REPO_ROOT / "external" / "BLINK" / "models"
RESOURCES = REPO_ROOT / "data" / "resources"

# Port 3001 has Freebase loaded; port 8890 is an empty Virtuoso shell
SPARQL_ENDPOINT = "http://localhost:3001/sparql"

# Broad entity superset (23.6 M unique MIDs — union of GNN-RAG CWQ, WebQSP,
# UniKGQA, and full freebase_2hops/all_entities)
ENTITIES_SUP_TXT   = RESOURCES / "entities_sup.txt"
ENTITIES_SUP_NAMES = RESOURCES / "entities_sup_names.tsv"

# Llama-3.3-70B-Instruct loaded in HF cache style under tmlr/models/
_LLAMA70B_SNAPSHOT = (
    REPO_ROOT / "models" / "Llama-3.3-70B-Instruct"
    / "models--meta-llama--Llama-3.3-70B-Instruct"
    / "snapshots"
    / "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
)
MODEL_CANDIDATES = [
    _LLAMA70B_SNAPSHOT,
    REPO_ROOT / "models" / "Llama-3.3-70B-Instruct",
    REPO_ROOT / "models" / "Llama-3.1-70B-Instruct",
]
# Fallback to 8B if 70B not yet available
FALLBACK_MODEL = REPO_ROOT / "models" / "Llama-3.1-8B-Instruct"

FB_NS = "http://rdf.freebase.com/ns/"

# ─── Entity superset helpers ──────────────────────────────────────────────────

def load_entities_sup() -> FrozenSet[str]:
    """Load entities_sup.txt → frozenset of MIDs (m.XXXX).

    Returns an empty frozenset (with a warning) if the file doesn't exist yet
    so that callers fall back gracefully to SPARQL verification.
    """
    if not ENTITIES_SUP_TXT.exists():
        print(
            f"[WARN] {ENTITIES_SUP_TXT} not found. "
            "Run `python tmlr/scripts/build_entities_sup.py --ids-only` first. "
            "Falling back to SPARQL-only verification.",
            flush=True,
        )
        return frozenset()
    with open(ENTITIES_SUP_TXT) as fh:
        s = frozenset(line.strip() for line in fh if line.strip())
    print(f"Loaded entities_sup: {len(s):,} MIDs from {ENTITIES_SUP_TXT.name}", flush=True)
    return s


def load_entities_sup_names() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load entities_sup_names.tsv → (mid2name, name2mid).

    name2mid maps lowercased name → first MID encountered.
    Returns empty dicts if file not yet built (still scanning fb_en.txt).
    """
    mid2name: Dict[str, str] = {}
    name2mid: Dict[str, str] = {}
    if not ENTITIES_SUP_NAMES.exists():
        print(
            f"[WARN] {ENTITIES_SUP_NAMES} not found yet (fb_en.txt scan still running?). "
            "Name-based disambiguation will be unavailable.",
            flush=True,
        )
        return mid2name, name2mid
    with open(ENTITIES_SUP_NAMES, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            mid, name = parts
            if mid not in mid2name:
                mid2name[mid] = name
            key = name.lower()
            if key not in name2mid:
                name2mid[key] = mid
    print(
        f"Loaded entities_sup_names: {len(mid2name):,} entries "
        f"({len(name2mid):,} unique names) from {ENTITIES_SUP_NAMES.name}",
        flush=True,
    )
    return mid2name, name2mid


# ─── Prompts ──────────────────────────────────────────────────────────────────

STEP1_SYSTEM = (
    "You are an expert entity extractor for knowledge-base question answering. "
    "Extract all named entities (people, places, organizations, events, works) "
    "from the question. For each entity, provide its exact mention span and "
    "2-3 possible Wikipedia title candidates. "
    "Output format: JSON array [{\"mention\": \"...\", \"candidates\": [\"...\", ...]}]. "
    "Return only valid JSON, nothing else."
)

STEP2_SYSTEM = (
    "You are an expert entity disambiguation system. "
    "Given a question, an entity mention, and candidate Wikipedia titles with descriptions, "
    "select the SINGLE best matching title. "
    "Reason step by step, then output ONLY the best title on the last line after 'ANSWER:'."
)

STEP2_USER_TMPL = (
    "Question: {question}\n"
    "Entity mention: \"{mention}\"\n"
    "Candidates:\n{candidates_block}\n"
    "Select the best matching Wikipedia title. Reason briefly, then output ANSWER: <title>"
)


def find_model_path(override: Optional[str]) -> Path:
    if override:
        p = Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(f"Model path not found: {override}")
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    if FALLBACK_MODEL.exists():
        print(f"WARNING: 70B model not found. Falling back to {FALLBACK_MODEL}")
        return FALLBACK_MODEL
    raise FileNotFoundError(
        "No Llama model found. Set --model-path or load Llama-3-70B into tmlr/models/."
    )


def load_model(model_path: Path) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batch generation

    # Spread model across all visible GPUs (80 GB A100s, keep 4 GB headroom for activations)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        # Use string-based memory specs to avoid CUDA init ordering issues
        max_memory = {i: "76GB" for i in range(n_gpus)}
        print(f"  Spreading model across {n_gpus} GPUs (76 GB each max)")
    else:
        max_memory = None

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    )
    model.eval()
    print("Model loaded.")
    return tokenizer, model


def _first_device(model: AutoModelForCausalLM) -> torch.device:
    """Return the device of the first parameter (for input tensors)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chat_generate_batch(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    conversations: List[Tuple[str, str]],  # [(system, user), ...]
    max_new_tokens: int = 256,
) -> List[str]:
    """Batch-generate responses for a list of (system, user) pairs."""
    prompts = []
    for system, user in conversations:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt = f"[SYSTEM]{system}[/SYSTEM]\n{user}\n"
        prompts.append(prompt)

    device = _first_device(model)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=512).to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i in range(len(prompts)):
        new_tokens = out[i][input_len:]
        results.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return results


def chat_generate(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    system: str,
    user: str,
    max_new_tokens: int = 256,
) -> str:
    return chat_generate_batch(tokenizer, model, [(system, user)], max_new_tokens)[0]


def step1_extract_candidates(
    tokenizer, model, question: str
) -> List[Dict[str, Any]]:
    """Return [{mention, candidates: [title, ...]}]"""
    raw = chat_generate(tokenizer, model, STEP1_SYSTEM, f"Question: {question}", max_new_tokens=256)
    # Parse JSON from response
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: extract any quoted strings as candidates
    mentions = re.findall(r'"mention"\s*:\s*"([^"]+)"', raw)
    return [{"mention": m, "candidates": []} for m in mentions]


def step2_disambiguate(
    tokenizer, model,
    question: str,
    mention: str,
    candidates: List[str],
    id2text: Dict[str, str],
    title2id: Dict[str, str],
) -> str:
    """Return best Wikipedia title for this mention."""
    if not candidates:
        return mention

    # Build candidates block with descriptions
    lines = []
    for i, cand in enumerate(candidates[:5], 1):
        eid = title2id.get(cand, "")
        desc = id2text.get(str(eid), "")[:120] if eid else ""
        lines.append(f"{i}. {cand}" + (f" — {desc}" if desc else ""))

    user = STEP2_USER_TMPL.format(
        question=question,
        mention=mention,
        candidates_block="\n".join(lines),
    )
    raw = chat_generate(tokenizer, model, STEP2_SYSTEM, user, max_new_tokens=128)

    # Extract answer after "ANSWER:"
    ans_match = re.search(r"ANSWER:\s*(.+)", raw, re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).strip().strip('"')
    # Fallback: first candidate
    return candidates[0] if candidates else mention


def step3_verify_mid(
    mid: str,
    entities_sup: FrozenSet[str],
    sparql_client: Optional["SPARQLWrapper"],
) -> bool:
    """Return True if MID is a known Freebase entity.

    Verification order:
      1. O(1) membership check against entities_sup (23.6 M MID superset).
      2. SPARQL ASK on localhost:3001 (live Freebase) for any MID outside local set.
         Port 8890 must NOT be used — it is an empty Virtuoso shell.
    """
    if not mid:
        return False
    if entities_sup and mid in entities_sup:
        return True
    if sparql_client is None:
        return False
    query = f"ASK {{ <{FB_NS}{mid}> ?p ?o }}"
    try:
        # Each call needs its own SPARQLWrapper to be thread-safe
        from SPARQLWrapper import JSON as _J, SPARQLWrapper as _SW
        c = _SW(sparql_client.endpoint)
        c.setTimeout(10)
        c.setQuery(query)
        c.setReturnFormat(_J)
        return c.query().convert().get("boolean", False)
    except Exception:
        return False


def step3_verify_batch(
    items: List[Tuple[str, str]],     # [(mid, mention), ...]
    entities_sup: FrozenSet[str],
    endpoint: str,
    n_threads: int,
) -> Dict[str, bool]:
    """Verify a batch of MIDs in parallel using a thread pool.

    Returns {mid: verified_bool}.  Uses entities_sup for O(1) fast-path; only
    fires SPARQL for the small fraction of MIDs not in the local set.
    n_threads is bounded by the number of MIDs that actually need SPARQL.
    """
    results: Dict[str, bool] = {}

    # Fast-path: entities_sup lookup (no network, no threads needed)
    need_sparql: List[str] = []
    for mid, _ in items:
        if not mid:
            results[mid] = False
        elif entities_sup and mid in entities_sup:
            results[mid] = True
        else:
            results[mid] = False   # default; overwritten if SPARQL confirms
            need_sparql.append(mid)

    if not need_sparql:
        return results

    # Thread-pool for SPARQL ASK on port 3001
    actual_threads = min(n_threads, len(need_sparql))

    def _ask(mid: str) -> Tuple[str, bool]:
        from SPARQLWrapper import JSON as _J, SPARQLWrapper as _SW
        query = f"ASK {{ <{FB_NS}{mid}> ?p ?o }}"
        try:
            c = _SW(endpoint)
            c.setTimeout(10)
            c.setQuery(query)
            c.setReturnFormat(_J)
            ok = c.query().convert().get("boolean", False)
        except Exception:
            ok = False
        return mid, ok

    with ThreadPoolExecutor(max_workers=actual_threads) as ex:
        for mid, ok in ex.map(_ask, need_sparql):
            results[mid] = ok

    return results


def lookup_mid_by_name(
    surface: str,
    name2mid: Dict[str, str],
    entities_sup: FrozenSet[str],
) -> str:
    """Try to find a verified MID for a surface-form string via the name index.

    Tries exact match (case-insensitive), then a simple prefix/contains heuristic.
    Returns "" if no match found.
    """
    if not surface or not name2mid:
        return ""
    key = surface.lower().strip()
    mid = name2mid.get(key, "")
    if mid and (not entities_sup or mid in entities_sup):
        return mid
    # Prefix heuristic: e.g. "United States of America" → "united states"
    words = key.split()
    if len(words) >= 2:
        prefix = " ".join(words[:2])
        mid = name2mid.get(prefix, "")
        if mid and (not entities_sup or mid in entities_sup):
            return mid
    return ""


def load_elq_spans(dataset: str, p_type: str) -> Dict[str, List[Dict]]:
    """Load ELQ-linked entities as {question_id → [linked_entity_dicts]}.

    Each linked entity dict from el_elq.py has:
      {mention, title, entity_id, mid, score, candidates: [{title,entity_id,mid,score},...]}

    Returns empty dict if ELQ output not found (falls back to LLM Step 1).
    """
    path = ELQ_OUT_DIR / f"{dataset}_{p_type}_el.json"
    if not path.exists():
        print(f"[WARN] ELQ output not found: {path} — will use LLM Step 1 for extraction",
              flush=True)
        return {}
    with open(path) as f:
        records: List[Dict] = json.load(f)
    span_map: Dict[str, List[Dict]] = {}
    n_missing_cands = 0
    for rec in records:
        ents = rec.get("linked_entities", [])
        # Backward compat: old ELQ files lack 'candidates' — synthesise single-entry list
        for e in ents:
            if "candidates" not in e:
                e["candidates"] = [{
                    "title": e.get("title", ""),
                    "entity_id": e.get("entity_id", ""),
                    "mid": e.get("mid", ""),
                    "score": e.get("score", 0.0),
                }]
                n_missing_cands += 1
        span_map[rec["id"]] = ents
    if n_missing_cands:
        print(f"  [NOTE] {n_missing_cands} entities had no 'candidates' field "
              "(old ELQ output) — re-run el_elq.py with --top-k-candidates 5 "
              "to get full candidate lists for better ELA disambiguation.", flush=True)
    print(f"Loaded ELQ spans for {len(span_map)} questions from {path.name}", flush=True)
    return span_map


def link_question(
    tokenizer, model,
    question: str,
    id2text: Dict[str, str],
    title2id: Dict[str, str],
    title2mid: Dict[str, str],
    name2mid: Dict[str, str],
    entities_sup: FrozenSet[str],
    sparql_client: Optional[SPARQLWrapper],
    verify: bool,
) -> List[Dict[str, Any]]:
    """Run 3-step ELA and return linked entities."""
    # Step 1: candidate extraction
    entity_candidates = step1_extract_candidates(tokenizer, model, question)

    linked = []
    for ec in entity_candidates:
        mention = ec.get("mention", "")
        candidates = ec.get("candidates", [])

        # Step 2: disambiguation
        best_title = step2_disambiguate(
            tokenizer, model, question, mention, candidates, id2text, title2id
        )

        # MID resolution: title2freebase_mid.json keys are all-lowercase
        mid = title2mid.get(best_title.lower(), "") or title2mid.get(best_title, "")

        # Supplement: if title lookup failed, try Freebase name index
        if not mid:
            mid = lookup_mid_by_name(best_title, name2mid, entities_sup)
        if not mid:
            mid = lookup_mid_by_name(mention, name2mid, entities_sup)

        # Step 3: verification against superset + SPARQL fallback
        verified = False
        if verify and mid:
            verified = step3_verify_mid(mid, entities_sup, sparql_client)
            if sparql_client:
                time.sleep(0.05)  # polite rate-limit

        linked.append({
            "mention": mention,
            "title": best_title,
            "mid": mid,
            "score": 1.0,  # ELA doesn't produce numeric scores
            "verified": verified,
        })

    return linked


def load_supporting_maps() -> Tuple[Dict, Dict, Dict, Dict[str, str], FrozenSet[str]]:
    """Load id2text, title2id, title2mid, name2mid, and entities_sup."""
    print("Loading id2text.json ...")
    with open(BLINK_MODELS / "id2text.json") as f:
        id2text: Dict[str, str] = json.load(f)

    print("Loading id2title.json ...")
    with open(BLINK_MODELS / "id2title.json") as f:
        id2title: Dict[str, str] = json.load(f)
    # Invert: title → entity_id
    title2id = {v: k for k, v in id2title.items()}

    print("Loading title2freebase_mid.json ...")
    with open(BLINK_MODELS / "title2freebase_mid.json") as f:
        title2mid: Dict[str, str] = json.load(f)

    # Broad entity superset (23.6 M MIDs — replaces GNN-RAG-only vocabulary check)
    print("Loading entities_sup.txt ...")
    entities_sup = load_entities_sup()

    # Freebase name → MID index (built from fb_en.txt scan; may be absent if scan in progress)
    print("Loading entities_sup_names.tsv ...")
    _mid2name, name2mid = load_entities_sup_names()

    return id2text, title2id, title2mid, name2mid, entities_sup


def link_batch(
    tokenizer, model,
    questions: List[str],
    id2text: Dict, title2id: Dict, title2mid: Dict,
    name2mid: Dict[str, str],
    entities_sup: FrozenSet[str],
    sparql_client: Optional[SPARQLWrapper],
    verify: bool,
    n_verify_threads: int = 32,
    endpoint: str = SPARQL_ENDPOINT,
) -> List[List[Dict[str, Any]]]:
    """Batch-link a list of questions. Returns list of linked_entities per question."""
    # Step 1 batch: extract candidates for all questions at once
    convs = [(STEP1_SYSTEM, f"Question: {q}") for q in questions]
    step1_outs = chat_generate_batch(tokenizer, model, convs, max_new_tokens=256)

    entity_candidates_per_q: List[List[Dict]] = []
    for raw in step1_outs:
        try:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            ecs = json.loads(match.group()) if match else []
        except (json.JSONDecodeError, AttributeError):
            mentions = re.findall(r'"mention"\s*:\s*"([^"]+)"', raw)
            ecs = [{"mention": m, "candidates": []} for m in mentions]
        entity_candidates_per_q.append(ecs)

    # Step 2 batch: disambiguation — batch across all (question, mention) pairs
    step2_convs: List[Tuple[str, str]] = []
    step2_index: List[Tuple[int, int]] = []  # (q_idx, mention_idx)
    for q_idx, (question, ecs) in enumerate(zip(questions, entity_candidates_per_q)):
        for m_idx, ec in enumerate(ecs):
            mention = ec.get("mention", "")
            candidates = ec.get("candidates", [])
            lines = []
            for i, cand in enumerate(candidates[:5], 1):
                eid = title2id.get(cand, "")
                desc = id2text.get(str(eid), "")[:120] if eid else ""
                lines.append(f"{i}. {cand}" + (f" — {desc}" if desc else ""))
            user = STEP2_USER_TMPL.format(
                question=question, mention=mention,
                candidates_block="\n".join(lines) if lines else "(no candidates)"
            )
            step2_convs.append((STEP2_SYSTEM, user))
            step2_index.append((q_idx, m_idx))

    step2_outs: List[str] = []
    if step2_convs:
        step2_outs = chat_generate_batch(tokenizer, model, step2_convs, max_new_tokens=128)

    # ── Assemble pre-verification records ────────────────────────────────────
    pre_results: List[Tuple[int, str, str, str]] = []  # (q_idx, mention, best_title, mid)
    for conv_idx, (q_idx, m_idx) in enumerate(step2_index):
        ec = entity_candidates_per_q[q_idx][m_idx]
        mention = ec.get("mention", "")
        candidates = ec.get("candidates", [])
        raw2 = step2_outs[conv_idx] if conv_idx < len(step2_outs) else ""
        ans_match = re.search(r"ANSWER:\s*(.+)", raw2, re.IGNORECASE)
        best_title = (ans_match.group(1).strip().strip('"')
                      if ans_match else (candidates[0] if candidates else mention))

        # MID resolution: title2freebase_mid.json keys are all-lowercase
        mid = title2mid.get(best_title.lower(), "") or title2mid.get(best_title, "")
        # Supplement: Freebase name index (no network cost)
        if not mid:
            mid = lookup_mid_by_name(best_title, name2mid, entities_sup)
        if not mid:
            mid = lookup_mid_by_name(mention, name2mid, entities_sup)

        pre_results.append((q_idx, mention, best_title, mid))

    # ── Step 3: batch-parallel verification (thread pool for SPARQL fallback) ─
    verified_map: Dict[str, bool] = {}
    if verify:
        items_to_verify = [(mid, mention) for (_, mention, _, mid) in pre_results if mid]
        verified_map = step3_verify_batch(
            items_to_verify, entities_sup, endpoint, n_verify_threads
        )

    # ── Build final per-question results ──────────────────────────────────────
    results_per_q: List[List[Dict]] = [[] for _ in questions]
    for q_idx, mention, best_title, mid in pre_results:
        verified = verified_map.get(mid, False) if verify else False
        results_per_q[q_idx].append({
            "mention": mention,
            "title": best_title,
            "mid": mid,
            "score": 1.0,
            "verified": verified,
        })

    return results_per_q


def link_batch_elq_mode(
    tokenizer, model,
    questions: List[str],
    question_ids: List[str],
    elq_span_map: Dict[str, List[Dict]],
    id2text: Dict, title2id: Dict, title2mid: Dict,
    name2mid: Dict[str, str],
    entities_sup: FrozenSet[str],
    verify: bool,
    n_verify_threads: int = 32,
    endpoint: str = SPARQL_ENDPOINT,
) -> List[List[Dict[str, Any]]]:
    """ELQ-NER mode: use ELQ span detection + candidates; LLM only runs Step 2.

    For each question:
      - ELQ has already detected mention spans and retrieved top-K Wikipedia candidates
      - LLM Step 2 re-ranks those candidates with CoT + description context
      - Step 3 verifies the chosen MID against entities_sup / SPARQL

    Questions where ELQ found no spans fall back to LLM Step 1 (full pipeline).
    """
    # Partition: ELQ spans available vs. need LLM Step 1 fallback
    has_elq  = [(i, qid, q) for i, (qid, q) in enumerate(zip(question_ids, questions))
                if elq_span_map.get(qid)]
    no_elq   = [(i, qid, q) for i, (qid, q) in enumerate(zip(question_ids, questions))
                if not elq_span_map.get(qid)]

    # --- ELQ-mode Step 2: batch disambiguation for ELQ spans -----------------
    step2_convs:  List[Tuple[str, str]] = []
    step2_index:  List[Tuple[int, int]] = []  # (batch_idx, span_idx)

    for batch_i, qid, question in has_elq:
        spans = elq_span_map[qid]
        for span_i, span in enumerate(spans):
            mention = span.get("mention", "")
            candidates = span.get("candidates", [])  # [{title, entity_id, mid, score}]

            # Build candidates block using ELQ top-K (with Wikipedia descriptions)
            lines = []
            for rank, cand in enumerate(candidates[:5], 1):
                cand_title = cand.get("title", "")
                eid = title2id.get(cand_title, cand.get("entity_id", ""))
                desc = id2text.get(str(eid), "")[:120] if eid else ""
                lines.append(f"{rank}. {cand_title}" + (f" — {desc}" if desc else ""))

            user = STEP2_USER_TMPL.format(
                question=question,
                mention=mention,
                candidates_block="\n".join(lines) if lines else "(no candidates from ELQ)",
            )
            step2_convs.append((STEP2_SYSTEM, user))
            step2_index.append((batch_i, span_i))

    # Batch LLM call for all ELQ-span disambiguations
    step2_outs: List[str] = []
    if step2_convs:
        step2_outs = chat_generate_batch(tokenizer, model, step2_convs, max_new_tokens=128)

    # Assemble pre-verification records from ELQ spans
    pre_results: List[Tuple[int, str, str, str]] = []  # (batch_idx, mention, title, mid)
    for conv_idx, (batch_i, span_i) in enumerate(step2_index):
        qid = question_ids[batch_i]
        span = elq_span_map[qid][span_i]
        mention = span.get("mention", "")
        candidates = span.get("candidates", [])
        raw2 = step2_outs[conv_idx] if conv_idx < len(step2_outs) else ""

        ans_match = re.search(r"ANSWER:\s*(.+)", raw2, re.IGNORECASE)
        if ans_match:
            chosen_title = ans_match.group(1).strip().strip('"')
        else:
            # LLM gave no ANSWER: — use ELQ top-1 directly
            chosen_title = candidates[0]["title"] if candidates else span.get("title", "")

        # MID resolution: title2freebase_mid.json keys are all-lowercase
        mid = title2mid.get(chosen_title.lower(), "") or title2mid.get(chosen_title, "")
        if not mid and candidates:
            mid = candidates[0].get("mid", "")  # ELQ top-1 MID as fallback
        if not mid:
            mid = lookup_mid_by_name(chosen_title, name2mid, entities_sup)
        if not mid:
            mid = lookup_mid_by_name(mention, name2mid, entities_sup)

        pre_results.append((batch_i, mention, chosen_title, mid))

    # --- Fallback: full LLM pipeline for questions ELQ gave no spans ----------
    fallback_pre: List[Tuple[int, str, str, str]] = []
    if no_elq:
        fb_qs    = [q for (_, _, q) in no_elq]
        fb_idxs  = [i for (i, _, _) in no_elq]
        fb_convs = [(STEP1_SYSTEM, f"Question: {q}") for q in fb_qs]
        fb_step1 = chat_generate_batch(tokenizer, model, fb_convs, max_new_tokens=256)

        fb_step2_convs: List[Tuple[str, str]] = []
        fb_step2_idx:   List[Tuple[int, int]] = []
        fb_ecs_per_q:   List[List[Dict]] = []
        for fb_i, (raw, q) in enumerate(zip(fb_step1, fb_qs)):
            try:
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                ecs = json.loads(match.group()) if match else []
            except (json.JSONDecodeError, AttributeError):
                mentions = re.findall(r'"mention"\s*:\s*"([^"]+)"', raw)
                ecs = [{"mention": m, "candidates": []} for m in mentions]
            fb_ecs_per_q.append(ecs)
            for m_i, ec in enumerate(ecs):
                mention = ec.get("mention", "")
                cands_t = ec.get("candidates", [])
                lines   = []
                for k, cand in enumerate(cands_t[:5], 1):
                    eid  = title2id.get(cand, "")
                    desc = id2text.get(str(eid), "")[:120] if eid else ""
                    lines.append(f"{k}. {cand}" + (f" — {desc}" if desc else ""))
                user = STEP2_USER_TMPL.format(
                    question=q, mention=mention,
                    candidates_block="\n".join(lines) if lines else "(no candidates)"
                )
                fb_step2_convs.append((STEP2_SYSTEM, user))
                fb_step2_idx.append((fb_i, m_i))

        fb_step2_outs: List[str] = []
        if fb_step2_convs:
            fb_step2_outs = chat_generate_batch(tokenizer, model, fb_step2_convs,
                                                max_new_tokens=128)

        for conv_i, (fb_i, m_i) in enumerate(fb_step2_idx):
            batch_i = fb_idxs[fb_i]
            ec      = fb_ecs_per_q[fb_i][m_i]
            mention = ec.get("mention", "")
            cands_t = ec.get("candidates", [])
            raw2    = fb_step2_outs[conv_i] if conv_i < len(fb_step2_outs) else ""
            ans_m   = re.search(r"ANSWER:\s*(.+)", raw2, re.IGNORECASE)
            best_t  = (ans_m.group(1).strip().strip('"')
                       if ans_m else (cands_t[0] if cands_t else mention))
            mid = title2mid.get(best_t.lower(), "") or title2mid.get(best_t, "")
            if not mid:
                mid = lookup_mid_by_name(best_t, name2mid, entities_sup)
            if not mid:
                mid = lookup_mid_by_name(mention, name2mid, entities_sup)
            fallback_pre.append((batch_i, mention, best_t, mid))

    all_pre = pre_results + fallback_pre

    # --- Step 3: batch-parallel verification ----------------------------------
    verified_map: Dict[str, bool] = {}
    if verify:
        items_to_verify = [(mid, mention) for (_, mention, _, mid) in all_pre if mid]
        verified_map = step3_verify_batch(
            items_to_verify, entities_sup, endpoint, n_verify_threads
        )

    # --- Build final per-question output --------------------------------------
    results_per_q: List[List[Dict]] = [[] for _ in questions]
    for batch_i, mention, best_title, mid in all_pre:
        verified = verified_map.get(mid, False) if verify else False
        results_per_q[batch_i].append({
            "mention": mention,
            "title": best_title,
            "mid": mid,
            "score": 1.0,
            "verified": verified,
            "ela_source": "elq_span",
        })

    return results_per_q


def link_dataset(
    dataset: str,
    p_type: str,
    tokenizer, model,
    id2text: Dict, title2id: Dict, title2mid: Dict,
    name2mid: Dict[str, str],
    entities_sup: FrozenSet[str],
    sparql_client: Optional[SPARQLWrapper],
    dry_run: bool,
    verify: bool,
    batch_size: int = 32,
    overwrite: bool = False,
    n_verify_threads: int = 32,
    endpoint: str = SPARQL_ENDPOINT,
    mode: str = "elq",              # "elq" = BLINK+ELQ NER→LLM Step2; "llm" = full LLM pipeline
    out_dir: Optional[Path] = None,  # override EL_OUT_DIR (for versioned runs)
) -> None:
    in_path = DATASET_DIR / f"{dataset}_{p_type}.json"
    _out_dir = Path(out_dir) if out_dir else EL_OUT_DIR
    _out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _out_dir / f"{dataset}_{p_type}_el.json"

    if not in_path.exists():
        print(f"  [SKIP] {in_path} not found")
        return

    if out_path.exists() and not dry_run and not overwrite:
        print(f"  [SKIP] {out_path} already exists ({out_path.stat().st_size // 1024} KB)  (use --overwrite to redo)")
        return

    with open(in_path) as f:
        records: List[Dict] = json.load(f)

    if dry_run:
        records = records[:5]

    # Load ELQ spans for elq-mode (BLINK+ELQ as NER feeding Step 2)
    elq_span_map: Dict[str, List[Dict]] = {}
    if mode == "elq":
        elq_span_map = load_elq_spans(dataset, p_type)
        if not elq_span_map:
            print(f"  [WARN] ELQ spans unavailable — falling back to full LLM pipeline",
                  flush=True)
            mode = "llm"

    print(f"  Linking {len(records)} records (batch_size={batch_size}, mode={mode}) ...",
          flush=True)
    all_results = []

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        questions   = [r.get("perturbed_question", "") for r in batch]
        question_ids = [r["id"] for r in batch]

        if mode == "elq":
            linked_list = link_batch_elq_mode(
                tokenizer, model, questions, question_ids,
                elq_span_map,
                id2text, title2id, title2mid,
                name2mid, entities_sup,
                verify,
                n_verify_threads=n_verify_threads,
                endpoint=endpoint,
            )
        else:
            linked_list = link_batch(
                tokenizer, model, questions,
                id2text, title2id, title2mid,
                name2mid, entities_sup,
                sparql_client, verify,
                n_verify_threads=n_verify_threads,
                endpoint=endpoint,
            )

        for rec, linked in zip(batch, linked_list):
            all_results.append({
                "id": rec["id"],
                "original_question": rec.get("original_question", ""),
                "perturbed_question": rec.get("perturbed_question", ""),
                "sparql": rec.get("sparql", ""),
                "perturbation_type": p_type,
                "linked_entities": linked,
            })

        if (i // batch_size + 1) % 20 == 0:
            print(f"    {i + len(batch)}/{len(records)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    linked_count = sum(1 for r in all_results if r["linked_entities"])
    print(f"  Saved → {out_path}  (linked={linked_count}/{len(all_results)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="ELA: LLM-agentic entity linking")
    parser.add_argument("--dataset", choices=["cwq", "webqsp"], default=None)
    parser.add_argument("--p-type", choices=["p1","p2","p3","p4","p5","p6","orig"], default=None)
    parser.add_argument("--all", action="store_true", help="Run all 12 dataset×p_type combos")
    parser.add_argument("--all-p-types", action="store_true",
                        help="Run all 6 P-types for the specified --dataset")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="LLM generation batch size (default 32; tune for GPU VRAM/throughput)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip entity verification (Step 3)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if output file already exists")
    parser.add_argument("--endpoint", default=SPARQL_ENDPOINT,
                        help=f"SPARQL endpoint (default: {SPARQL_ENDPOINT})")
    parser.add_argument(
        "--verify-threads", type=int,
        default=min(64, cpu_count()),
        help="Threads for parallel SPARQL verification (default min(64, nproc))",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Override output directory (default: el_agent/ELA_v1/). "
             "Pass e.g. el_agent/ELA_v2 for a new versioned run.",
    )
    parser.add_argument(
        "--mode", choices=["elq", "llm"], default="elq",
        help=(
            "ELA pipeline mode (default: elq). "
            "'elq' = BLINK+ELQ provides NER spans + top-K candidates; LLM only runs "
            "Step 2 (disambiguation) and Step 3 (verification). "
            "Falls back to 'llm' for any question where ELQ produced no spans. "
            "'llm' = original full pipeline: LLM Step 1 (extraction) + Step 2 + Step 3."
        ),
    )
    args = parser.parse_args()

    if args.all:
        datasets = ["cwq", "webqsp"]
        p_types = ["p1","p2","p3","p4","p5","p6"]
    elif args.all_p_types:
        if not args.dataset:
            parser.error("--all-p-types requires --dataset cwq|webqsp")
        datasets = [args.dataset]
        p_types = ["p1","p2","p3","p4","p5","p6"]
    else:
        if not args.dataset or not args.p_type:
            parser.error("Specify --dataset and --p-type, --all-p-types, or --all")
        datasets = [args.dataset]
        p_types = [args.p_type]

    model_path = find_model_path(args.model_path)
    tokenizer, model = load_model(model_path)
    id2text, title2id, title2mid, name2mid, entities_sup = load_supporting_maps()

    # Port 3001 = live Freebase Virtuoso; port 8890 = empty shell (do not use)
    sparql_client: Optional[SPARQLWrapper] = None
    if not args.no_verify:
        endpoint = args.endpoint  # default is already port 3001
        try:
            sparql_client = SPARQLWrapper(endpoint)
            sparql_client.setTimeout(15)
            print(f"SPARQL client → {endpoint} (Freebase fallback for MIDs not in entities_sup)")
        except Exception as e:
            print(f"WARNING: SPARQL client init failed ({e}). Using entities_sup only for verification.")

    n_verify_threads = args.verify_threads
    print(
        f"Config: mode={args.mode}  batch_size={args.batch_size}  "
        f"verify_threads={n_verify_threads}  "
        f"verify={'yes' if not args.no_verify else 'no'}  overwrite={args.overwrite}",
        flush=True,
    )

    for dataset in datasets:
        print(f"\n=== Dataset: {dataset} ===", flush=True)
        for p_type in p_types:
            print(f"\n--- {dataset} {p_type} ---", flush=True)
            t0 = time.time()
            link_dataset(
                dataset, p_type,
                tokenizer, model,
                id2text, title2id, title2mid,
                name2mid, entities_sup,
                sparql_client,
                dry_run=args.dry_run,
                verify=not args.no_verify,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                n_verify_threads=n_verify_threads,
                endpoint=args.endpoint,
                mode=args.mode,
                out_dir=Path(args.out_dir) if args.out_dir else None,
            )
            print(f"  Elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()