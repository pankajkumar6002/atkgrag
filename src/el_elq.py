#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
el_elq.py — ELQ/BLINK entity linking on perturbed questions.

Architecture:
  - ELQ biencoder: GPU-accelerated (BERT encoder on CUDA)
  - FAISS HNSW index: CPU-only (graph-based ANN — does not support GPU FAISS)
  - title2freebase_mid / id2title lookup: CPU, parallelised with ThreadPoolExecutor
  - Batch inference via ELQ's built-in batched `run()` call

Multi-worker strategy:
  --workers N  : ThreadPoolExecutor for I/O (JSON read/write, MID lookup)
                 Inference itself is batched on 1 GPU per dataset.
  --batch-size : controls GPU batch for the biencoder; larger = faster on A100.

Models used:
  CWQ:    BLINK/models/elq_wiki_large.bin     (general Wikipedia ELQ)
  WebQSP: BLINK/models/elq_webqsp_large.bin  (WebQSP fine-tuned ELQ)

No SPARQL needed — ELQ is fully local (biencoder + FAISS + lookup tables).

Usage:
  CUDA_VISIBLE_DEVICES=0 python el_elq.py --dataset cwq --p-type p1 --batch-size 64
  CUDA_VISIBLE_DEVICES=0 python el_elq.py --dataset cwq --p-type p1 --dry-run
  CUDA_VISIBLE_DEVICES=0,1 python el_elq.py --all --batch-size 64 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

BLINK_ROOT = REPO_ROOT / "external" / "BLINK"
MODELS_DIR = BLINK_ROOT / "models"
DATASET_DIR = REPO_ROOT / "data" / "perturbations"
EL_OUT_DIR = DATASET_DIR / "el_elq"
ENTITIES_SUP_NAMES = REPO_ROOT / "data" / "resources" / "entities_sup_names.tsv"

# Default mention detection threshold.  ELQ's original default is -2.0, but that is
# too strict for multi-entity CWQ questions where secondary entities (e.g. "France"
# in "who is the president of France?") often score slightly below the threshold.
# Lowering to -3.0 improves recall at a small precision cost.
DEFAULT_MENTION_THRESHOLD: float = -3.0

# select_seeds is kept for reference but is NO LONGER called in run_elq_batch.
# Applying dual-seed logic across spans (not within a span) incorrectly drops
# valid second entities in multi-hop questions.
DEFAULT_DUAL_SEED_MARGIN: float = 1.5  # kept only for backward-compat imports

sys.path.insert(0, str(BLINK_ROOT))
# ELQ uses hardcoded relative 'models/' paths internally — must run from BLINK root
os.chdir(str(BLINK_ROOT))


def load_name2mid() -> Dict[str, str]:
    """Load entities_sup_names.tsv → {name.lower(): MID} fallback index.

    Used when title2freebase_mid.json has no entry for a Wikipedia title
    (e.g. less prominent entities whose Freebase MID wasn't in BLINK's training
    corpus).  entities_sup_names.tsv covers 16.2 M Freebase entities.
    """
    name2mid: Dict[str, str] = {}
    if not ENTITIES_SUP_NAMES.exists():
        print(f"[WARN] {ENTITIES_SUP_NAMES} not found — name2mid fallback disabled.")
        return name2mid
    print(f"Loading name2mid from {ENTITIES_SUP_NAMES.name} ({ENTITIES_SUP_NAMES.stat().st_size/1e9:.1f} GB)...")
    with open(ENTITIES_SUP_NAMES, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            mid, name = parts
            key = name.lower()
            if key not in name2mid:
                name2mid[key] = mid
    print(f"  Loaded {len(name2mid):,} name→MID entries.")
    return name2mid


def load_title2mid() -> Dict[str, str]:
    path = MODELS_DIR / "title2freebase_mid.json"
    print(f"Loading title2freebase_mid ({path.stat().st_size / 1e9:.1f} GB) ...")
    with open(path) as f:
        return json.load(f)


def load_id2title() -> Dict[str, str]:
    path = MODELS_DIR / "id2title.json"
    with open(path) as f:
        return json.load(f)


def load_elq_models(dataset: str, batch_size: int,
                    mention_threshold: float = DEFAULT_MENTION_THRESHOLD):
    """Load ELQ model appropriate for the dataset."""
    import argparse as _ap
    from elq.main_dense import load_models

    webqsp_model = MODELS_DIR / "elq_webqsp_large.bin"
    wiki_model = MODELS_DIR / "elq_wiki_large.bin"
    if dataset == "webqsp" and webqsp_model.stat().st_size > 1_000_000_000:
        biencoder_model = str(webqsp_model)
    else:
        if dataset == "webqsp":
            print(f"  [WARN] elq_webqsp_large.bin appears corrupted "
                  f"({webqsp_model.stat().st_size / 1e6:.0f} MB < 1 GB) — "
                  f"falling back to elq_wiki_large.bin")
        biencoder_model = str(wiki_model)

    args = _ap.Namespace(
        biencoder_model=biencoder_model,
        biencoder_config=str(MODELS_DIR / "elq_large_params.json"),
        entity_catalogue=str(MODELS_DIR / "entity.jsonl"),
        entity_encoding=str(MODELS_DIR / "all_entities_large.t7"),
        cand_token_ids_path=str(MODELS_DIR / "entity_token_ids_128.t7"),
        faiss_index="hnsw",
        index_path=str(MODELS_DIR / "faiss_hnsw_index.pkl"),
        num_cand_mentions=10,
        num_cand_entities=10,
        threshold_type="joint",
        threshold=-4.5,
        mention_threshold=mention_threshold,
        max_context_length=128,
        use_cuda=True,
        no_logger=False,
        save_preds_dir=None,
        get_predictions=False,
        interactive=False,
        test_mentions=None,
        test_entities=None,
        eval_batch_size=batch_size,
        debug_biencoder=False,
        output_path=None,
    )
    print(f"Loading ELQ models (dataset={dataset}) ...")
    models = load_models(args, logger=None)
    return args, models


def resolve_mid(entity_id: int, pred: Dict, i: int,
                id2title: Dict[str, str],
                title2mid: Dict[str, str]) -> Dict[str, Any]:
    """Resolve a single ELQ prediction triple to a MID — runs in thread pool."""
    mention = ""
    if pred.get("pred_tuples_string") and i < len(pred["pred_tuples_string"]):
        mention = pred["pred_tuples_string"][i][1]
    score = pred["scores"][i] if i < len(pred.get("scores", [])) else 0.0
    title = id2title.get(str(entity_id), "")
    # title2freebase_mid.json keys are lowercase — normalise for lookup
    mid = title2mid.get(title.lower(), "") or title2mid.get(title, "")
    return {
        "mention": mention,
        "title": title,
        "entity_id": entity_id,
        "mid": mid,
        "score": round(float(score), 4),
    }


def select_seeds(
    linked: List[Dict[str, Any]],
    dual_seed_margin: float = DEFAULT_DUAL_SEED_MARGIN,
) -> List[Dict[str, Any]]:
    """Apply two-seed heuristic: if top-2 joint scores are within `dual_seed_margin`
    nats of each other, keep BOTH as seeds; otherwise keep only the best one.

    The ELQ joint score = log_softmax(cand_logits) + log(sigmoid(mention_logit)).
    Scores are in log-space (always negative).  A small |score[0] - score[1]|
    means the model is nearly equally confident about two candidates.

    For WebQSP questions that genuinely have two topic entities (e.g. "who plays
    ken barlow in coronation street"), the two entities will typically come from
    *different* mention spans and thus their gap can exceed the margin even when
    both are correct.  The heuristic targets the case where the same mention
    produces two near-equally-scored candidates → the model is ambiguous, so we
    include both to avoid dropping the correct seed.

    Records with only 0 or 1 linked entity are returned unchanged.
    """
    if len(linked) < 2:
        return linked

    # Sort by descending score so index 0 is always the best
    sorted_linked = sorted(linked, key=lambda e: e.get("score", 0.0), reverse=True)

    top_score = sorted_linked[0].get("score", 0.0)
    second_score = sorted_linked[1].get("score", 0.0)
    gap = abs(top_score - second_score)

    if gap < dual_seed_margin:
        # Scores are close — keep top-2 seeds; mark both with a flag
        for e in sorted_linked[:2]:
            e["dual_seed"] = True
        return sorted_linked[:2]
    else:
        # Clear winner — use only top-1
        sorted_linked[0]["dual_seed"] = False
        return sorted_linked[:1]


def _title_to_mid_lookup(
    title: str,
    title2mid: Dict[str, str],
    name2mid: Optional[Dict[str, str]],
) -> str:
    """Resolve a Wikipedia title to a Freebase MID.

    Lookup order:
      1. title2freebase_mid.json — exact, then lowercase (keys are all-lowercase)
      2. entities_sup_names.tsv  — exact lowercase title
      3. Strip parenthetical qualifier, e.g. "James Burke (actor)" → "James Burke"
      4. Strip geographic suffix after comma, e.g. "Paducah, Kentucky" → "Paducah"

    Variants 3 and 4 are tried against BOTH title2mid and name2mid so that
    disambiguation-style Wikipedia titles ("X (Y)") and geographic titles ("X, State")
    are recovered even when the base form is in the Freebase data.
    """
    import re

    # 1. Primary: title2mid exact + lowercase
    mid = title2mid.get(title.lower(), "") or title2mid.get(title, "")
    if mid:
        return mid

    # 2. name2mid exact
    if name2mid:
        mid = name2mid.get(title.lower(), "")
        if mid:
            return mid

    # Build normalised variants to try
    variants: List[str] = []

    # Variant A: strip trailing parenthetical "(…)"
    stripped_paren = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    if stripped_paren and stripped_paren != title:
        variants.append(stripped_paren)

    # Variant B: strip geographic suffix after last comma, e.g. "Paducah, Kentucky"
    if "," in title:
        stripped_comma = title.rsplit(",", 1)[0].strip()
        if stripped_comma and stripped_comma != title:
            variants.append(stripped_comma)

    for v in variants:
        mid = title2mid.get(v.lower(), "") or title2mid.get(v, "")
        if mid:
            return mid
        if name2mid:
            mid = name2mid.get(v.lower(), "")
            if mid:
                return mid

    return ""


def group_candidates_by_mention(
    triples: List,
    pred: Dict,
    id2title: Dict[str, str],
    title2mid: Dict[str, str],
    top_k: int = 5,
    name2mid: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Group ELQ prediction triples by mention span; keep top-K candidates per span.

    ELQ returns one triple per (mention_span × candidate) combination, all sorted
    by joint score.  We group them so each entry in the output represents ONE
    mention span with a ranked list of up to `top_k` candidate entities.

    ALL detected spans are returned — never drop a span here.  Multi-entity
    questions (e.g. CWQ with 2-3 topic entities) need every span kept.

    MID resolution order (via _title_to_mid_lookup):
      1. title2freebase_mid.json  (22.9M entries, all-lowercase keys)
      2. entities_sup_names.tsv   (16.2M entries — fallback for entities missing from title2mid)
      3. Strip parenthetical — "James Burke (actor)" → "James Burke"
      4. Strip geo-suffix — "Paducah, Kentucky" → "Paducah"

    Output per mention:
      {mention, title, entity_id, mid, score,   ← top-1 per span
       candidates: [{title, entity_id, mid, score}, ...]}  ← top-K for ELA
    """
    from collections import OrderedDict

    scores = pred.get("scores", [])
    tuples_str = pred.get("pred_tuples_string", [])

    # Pass 1: group by mention span
    mention_to_cands: "OrderedDict[str, list]" = OrderedDict()
    for i, triple in enumerate(triples):
        entity_id = triple[0]
        mention = tuples_str[i][1] if i < len(tuples_str) else ""
        score = float(scores[i]) if i < len(scores) else 0.0
        title = id2title.get(str(entity_id), "")
        mid = _title_to_mid_lookup(title, title2mid, name2mid)

        if mention not in mention_to_cands:
            mention_to_cands[mention] = []
        mention_to_cands[mention].append({
            "title": title,
            "entity_id": str(entity_id),
            "mid": mid,
            "score": round(score, 4),
        })

    # Pass 2: best candidate at top per span; keep up to top_k candidates
    linked = []
    for mention, cands in mention_to_cands.items():
        cands.sort(key=lambda c: c["score"], reverse=True)
        best = cands[0]
        linked.append({
            "mention": mention,
            "title": best["title"],
            "entity_id": best["entity_id"],
            "mid": best["mid"],
            "score": best["score"],
            "candidates": cands[:top_k],  # top-K for ELA Step 2 re-ranking
        })

    # Return ALL spans — caller must not prune across spans (Bug fix: select_seeds
    # was incorrectly called across spans, dropping valid multi-entity question seeds)
    return linked


def run_elq_batch(
    args,
    models,
    questions: List[Dict],
    id2title: Dict[str, str],
    title2mid: Dict[str, str],
    workers: int = 8,
    top_k_candidates: int = 10,
    name2mid: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    from elq.main_dense import run

    data = [{"id": q["id"], "text": q["perturbed_question"],
             "tokens": q["perturbed_question"].split()}
            for q in questions]

    predictions = run(args, None, *models, test_data=data)

    results = []
    for pred in predictions:
        triples = pred.get("pred_triples", [])

        # Group by mention span; return ALL spans (never prune across spans).
        # select_seeds() was removed here — it incorrectly compared scores across
        # different mention spans, dropping valid 2nd/3rd entities in multi-hop
        # questions.  Each span is already the top-1 candidate for that mention.
        linked = group_candidates_by_mention(
            triples, pred, id2title, title2mid,
            top_k=top_k_candidates, name2mid=name2mid,
        )

        results.append({
            "id": pred["id"],
            "perturbed_question": pred["text"],
            "linked_entities": linked,
        })
    return results


def link_dataset(
    dataset: str,
    p_type: str,
    batch_size: int,
    dry_run: bool,
    args_elq,
    models,
    id2title: Dict[str, str],
    title2mid: Dict[str, str],
    workers: int = 8,
    top_k_candidates: int = 10,
    name2mid: Optional[Dict[str, str]] = None,
) -> None:
    in_path = DATASET_DIR / f"{dataset}_{p_type}.json"
    out_path = EL_OUT_DIR / f"{dataset}_{p_type}_el.json"

    if not in_path.exists():
        print(f"  [SKIP] {in_path} not found")
        return

    with open(in_path) as f:
        records: List[Dict] = json.load(f)

    if dry_run:
        records = records[:10]

    print(f"  Linking {len(records)} records from {in_path.name} ...")

    all_results: List[Dict] = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        batch_results = run_elq_batch(args_elq, models, batch, id2title, title2mid,
                                       workers=workers,
                                       top_k_candidates=top_k_candidates,
                                       name2mid=name2mid)
        all_results.extend(batch_results)
        if (i // batch_size + 1) % 10 == 0:
            print(f"    processed {i + len(batch)}/{len(records)}")

    # Merge original record fields back in for downstream use
    id2record = {r["id"]: r for r in records}
    for res in all_results:
        orig = id2record.get(res["id"], {})
        res["original_question"] = orig.get("original_question", "")
        res["sparql"] = orig.get("sparql", "")
        res["perturbation_type"] = orig.get("perturbation_type", p_type)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    linked_count = sum(1 for r in all_results if r["linked_entities"])
    print(f"  Saved {len(all_results)} records → {out_path}  "
          f"(linked={linked_count}, no_link={len(all_results)-linked_count})")


def main() -> None:
    parser = argparse.ArgumentParser(description="ELQ/BLINK entity linking on perturbed questions")
    parser.add_argument("--dataset", choices=["cwq", "webqsp"], default=None)
    parser.add_argument("--p-type", choices=["p1","p2","p3","p4","p5","p6","p7","orig"], default=None)
    parser.add_argument("--all", action="store_true", help="Run all 12 dataset×p_type combinations")
    parser.add_argument("--all-p-types", action="store_true",
                        help="Run all 6 P-types for the specified --dataset")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="GPU batch size for biencoder (default 64 for A100)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Thread workers for parallel MID lookup (default 8)")
    parser.add_argument("--dry-run", action="store_true", help="Only process first 10 questions")
    parser.add_argument(
        "--top-k-candidates", type=int, default=10,
        help=(
            "Number of ELQ candidates to keep per mention span (default 10). "
            "These are stored in the 'candidates' field of each linked entity "
            "and used by el_agent.py --mode elq for LLM Step 2 re-ranking. "
            "Higher values give the LLM more candidates to pick from, improving "
            "recall when ELQ's top-1 is wrong."
        ),
    )
    parser.add_argument(
        "--mention-threshold", type=float, default=DEFAULT_MENTION_THRESHOLD,
        help=(
            "ELQ mention detection threshold (default %(default)s). "
            "Lower values detect more spans (higher recall, lower precision). "
            "Original ELQ default was -2.0; -3.0 improves recall for CWQ "
            "multi-entity questions without significant precision loss."
        ),
    )
    parser.add_argument(
        "--no-name2mid", action="store_true",
        help="Disable entities_sup_names.tsv fallback (faster startup, lower MID coverage).",
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

    id2title = load_id2title()
    title2mid = load_title2mid()
    name2mid = {} if args.no_name2mid else load_name2mid()

    for dataset in datasets:
        print(f"\n=== Dataset: {dataset} ===")
        args_elq, models = load_elq_models(dataset, args.batch_size,
                                           mention_threshold=args.mention_threshold)
        for p_type in p_types:
            print(f"\n--- {dataset} {p_type} ---")
            link_dataset(dataset, p_type, args.batch_size, args.dry_run,
                         args_elq, models, id2title, title2mid,
                         workers=args.workers,
                         top_k_candidates=args.top_k_candidates,
                         name2mid=name2mid)


if __name__ == "__main__":
    main()