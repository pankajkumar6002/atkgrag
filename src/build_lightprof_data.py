#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_lightprof_data.py — Build LightPROF data pipeline for perturbed CWQ/WebQSP.

Outputs (in tmlr/Repo/LightPROF/data/):
  1. freebase_triples_full.tsv   — KG triples (head_name\trelation\ttail_name)
  2. webqsp_{split}.jsonl        — QA JSONL per split/perturbation
  3. cwq_{split}.jsonl           — QA JSONL per split/perturbation
  4. hop_labels_webqsp.json      — hop labels for HopPredictor training (WebQSP=2)
  5. hop_labels_cwq.json         — hop labels for HopPredictor training (CWQ=1-4)

Usage:
  python build_lightprof_data.py --step kg        # Build KG triples TSV
  python build_lightprof_data.py --step qa        # Build QA JSONL files
  python build_lightprof_data.py --step hops      # Build hop label files
  python build_lightprof_data.py --step all       # All steps
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT     = REPO_ROOT
TMLR          = REPO_ROOT / "tmlr"
LIGHTPROF_DIR = TMLR / "Repo/LightPROF"
OUT_DIR       = LIGHTPROF_DIR / "data"
PERT_DIR      = TMLR / "pert/dataset"
MID_NAME_TSV  = TMLR / "resources/entities_sup_names.tsv"
PKL_WEBQSP    = REPO_ROOT / "data" / "mid_cache" / "neighborhoods_webqsp.pkl"
PKL_CWQ       = REPO_ROOT / "data" / "mid_cache" / "neighborhoods_cwq.pkl"
GNN_DATA      = TMLR / "Repo/GNN-RAG/data"

# Perturbation types for each dataset
PERT_TYPES = ["p1", "p4", "p7"]


# ---------------------------------------------------------------------------
# Step 1: Build KG triples TSV (MID tuples → name-based triples)
# ---------------------------------------------------------------------------

def load_mid_name_map(tsv_path: Path, mids_needed: set[str]) -> dict[str, str]:
    """Load MID→name mapping for only the MIDs we need (memory-efficient scan)."""
    print(f"  Loading MID→name map from {tsv_path} for {len(mids_needed)} MIDs ...")
    mid2name: dict[str, str] = {}
    with open(tsv_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 2_000_000 == 0:
                print(f"    scanned {i:,} lines, found {len(mid2name):,} / {len(mids_needed):,}")
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[0] in mids_needed:
                mid2name[parts[0]] = parts[1]
            if len(mid2name) == len(mids_needed):
                break
    print(f"  Found {len(mid2name):,} / {len(mids_needed):,} MIDs")
    return mid2name


def sanitize_name(name: str) -> str:
    """Replace spaces/special chars with underscores for TSV entity names."""
    # Keep alphanumeric, dots (for relation names), underscores, hyphens
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    return name


def build_kg(out_dir: Path) -> None:
    """Extract all triples from PKL caches and write name-based TSV."""
    out_path = out_dir / "freebase_triples_full.tsv"
    if out_path.exists():
        print(f"  KG already exists: {out_path} ({out_path.stat().st_size // 1_048_576} MB)")
        return

    print("Step 1: Building freebase_triples_full.tsv ...")
    print("  Loading PKL caches ...")
    with open(PKL_WEBQSP, "rb") as f:
        pkl_w = pickle.load(f)
    with open(PKL_CWQ, "rb") as f:
        pkl_c = pickle.load(f)

    # Collect all unique triples and entity MIDs
    all_triples: set[tuple[str, str, str]] = set()
    entity_mids: set[str] = set()
    for nhood in list(pkl_w.values()) + list(pkl_c.values()):
        for h, r, t in nhood["tuples"]:
            all_triples.add((h, r, t))
            if h.startswith("m.") or h.startswith("g."):
                entity_mids.add(h)
            if t.startswith("m.") or t.startswith("g."):
                entity_mids.add(t)

    print(f"  Total triples: {len(all_triples):,}, entity MIDs to resolve: {len(entity_mids):,}")

    # Load name map for all entity MIDs
    mid2name = load_mid_name_map(MID_NAME_TSV, entity_mids)

    # Write TSV: skip triples where entity names are missing
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for h, r, t in all_triples:
            # Map entity MIDs to names; keep relation as-is (it's already text)
            h_name = mid2name.get(h, h) if (h.startswith("m.") or h.startswith("g.")) else h
            t_name = mid2name.get(t, t) if (t.startswith("m.") or t.startswith("g.")) else t
            # Skip triples where entity MID couldn't be resolved (still looks like MID)
            if (h_name.startswith("m.") and len(h_name) < 20) or (t_name.startswith("m.") and len(t_name) < 20):
                skipped += 1
                continue
            h_san = sanitize_name(h_name)
            t_san = sanitize_name(t_name)
            r_san = sanitize_name(r)
            f.write(f"{h_san}\t{r_san}\t{t_san}\n")
            written += 1

    print(f"  Written: {written:,} triples, skipped (unresolved MID): {skipped:,}")
    print(f"  Output: {out_path} ({out_path.stat().st_size // 1_048_576} MB)")


# ---------------------------------------------------------------------------
# Step 2: Build QA JSONL files from perturbed datasets
# ---------------------------------------------------------------------------

def load_mid2name_from_perturbed(records: list[dict]) -> dict[str, str]:
    """Extract MID→name from gold_entity_map across all records."""
    mid2name: dict[str, str] = {}
    for r in records:
        gem = r.get("gold_entity_map", {})
        if isinstance(gem, dict):
            for mid, name in gem.items():
                if mid and name:
                    mid2name[mid] = name
    return mid2name


def build_qa_jsonl(dataset: str, out_dir: Path, mid2name_global: dict[str, str]) -> None:
    """Build LightPROF-format JSONL files for a dataset (orig + all pert types)."""
    print(f"\nStep 2: Building QA JSONL for {dataset} ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build MID→name cache from all available perturbed files (accumulate names)
    for pt in PERT_TYPES:
        pfile = PERT_DIR / f"{dataset}_{pt}.json"
        if pfile.exists():
            records = json.load(open(pfile))
            mid2name_global.update(load_mid2name_from_perturbed(records))

    # Build id→gem fallback from p4/p7 (p1 has no GEM; use these for orig and p1)
    id2gem: dict[str, dict[str, str]] = {}  # question_id → gold_entity_map
    for fallback_pt in ["p4", "p7", "p2", "p3", "p5", "p6"]:
        fb_file = PERT_DIR / f"{dataset}_{fallback_pt}.json"
        if fb_file.exists():
            fb_records = json.load(open(fb_file))
            for r in fb_records:
                qid = r.get("id", "")
                gem = r.get("gold_entity_map", {}) or {}
                if qid and gem and qid not in id2gem:
                    id2gem[qid] = gem
            if id2gem:
                break  # Use first available fallback pt

    # Load original GNN-RAG test.json: build id→answers map
    # Some perturbed files (e.g. p1) don't carry answers/gem — join from here
    gnn_test = GNN_DATA / ("CWQ" if dataset == "cwq" else "webqsp") / "test.json"
    orig_answers: dict[str, list[str]] = {}   # id → [answer_text, ...]
    if gnn_test.exists():
        with open(gnn_test) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    qid = d.get("id", "")
                    ans_list = []
                    for ans in d.get("answers", []):
                        if isinstance(ans, dict):
                            text = ans.get("text", "")
                            mid = ans.get("kb_id", "")
                            if text:
                                ans_list.append(sanitize_name(text))
                            if mid and text:
                                mid2name_global[mid] = text
                    orig_answers[qid] = ans_list
                except Exception:
                    pass

    def resolve_answer(mid: str) -> str | None:
        """Resolve a MID to text name. Returns None if unresolvable."""
        if not mid.startswith("m.") and not mid.startswith("g."):
            return mid  # Already a text name
        return mid2name_global.get(mid)

    def gem_to_anchor_entities(gem: dict[str, str]) -> list[dict[str, str]]:
        """Convert gold_entity_map to anchor_entities dicts for FreebaseKG lookup.

        Output format: [{"id": sanitized_name, "name": original_name}]
        The "id" field uses sanitized name since FreebaseKG indexes by name (not MID).
        """
        result = []
        for mid, name in gem.items():
            if mid and name:
                result.append({"id": sanitize_name(name), "name": name})
        return result

    for pt in PERT_TYPES:
        pfile = PERT_DIR / f"{dataset}_{pt}.json"
        if not pfile.exists():
            print(f"  Skipping {dataset} {pt}: file not found")
            continue

        records = json.load(open(pfile))
        out_file = out_dir / f"{dataset}_{pt}_test.jsonl"

        written = skipped_ans = 0
        with open(out_file, "w", encoding="utf-8") as fout:
            for rec in records:
                qid = rec.get("id", "")
                question = rec.get("perturbed_question", rec.get("original_question", ""))
                gem = rec.get("gold_entity_map", {}) or {}
                raw_answers = rec.get("answers", [])

                # Resolve answers from MIDs to text names
                answers_text = []
                for ans in raw_answers:
                    if isinstance(ans, dict):
                        text = ans.get("text") or resolve_answer(ans.get("kb_id", ""))
                    else:
                        text = resolve_answer(str(ans))
                    if text:
                        answers_text.append(sanitize_name(text))

                # Fallback: use original test.json answers (for p1 which has no answer field)
                if not answers_text and qid in orig_answers:
                    answers_text = orig_answers[qid]

                if not answers_text:
                    skipped_ans += 1

                # Build anchor entities; fall back to p4/p7 GEM if this pt has none
                if not gem:
                    gem = id2gem.get(qid, {})
                anchor_entities = gem_to_anchor_entities(gem)

                sample = {
                    "question_id": qid,
                    "question": question,
                    "answers": answers_text,
                    "anchor_entities": anchor_entities,
                    "perturbation_type": pt,
                }
                fout.write(json.dumps(sample) + "\n")
                written += 1

        print(f"  {dataset} {pt}: {written} records → {out_file.name} (empty_ans={skipped_ans})")

    # Also build original (unperturbed) test set for baseline
    orig_out = out_dir / f"{dataset}_orig_test.jsonl"
    if gnn_test.exists():
        written = 0
        with open(gnn_test) as fin, open(orig_out, "w") as fout:
            for line in fin:
                try:
                    d = json.loads(line)
                    qid = d.get("id", "")
                    question = d.get("question", "")
                    answers_text = [
                        sanitize_name(a["text"])
                        for a in d.get("answers", [])
                        if isinstance(a, dict) and a.get("text")
                    ]
                    # Use GEM from p4/p7 as anchor entities (integer IDs in GNN-RAG are unusable)
                    gem = id2gem.get(qid, {})
                    anchor_entities = gem_to_anchor_entities(gem)
                    sample = {
                        "question_id": qid,
                        "question": question,
                        "answers": answers_text,
                        "anchor_entities": anchor_entities,
                        "perturbation_type": "orig",
                    }
                    fout.write(json.dumps(sample) + "\n")
                    written += 1
                except Exception:
                    pass
        print(f"  {dataset} orig: {written} records → {orig_out.name}")


# ---------------------------------------------------------------------------
# Step 3: Build hop label files for HopPredictor training
# ---------------------------------------------------------------------------

def build_hop_labels(dataset: str, out_dir: Path) -> None:
    """Build hop_labels JSON for HopPredictor training."""
    print(f"\nStep 3: Building hop labels for {dataset} ...")

    # WebQSP: all questions are 1-2 hop (label all as 2 — conservative)
    # CWQ: complexity field maps to hop count
    CWQ_HOP_MAP = {
        "conjunction": 2,
        "composition": 3,
        "comparative": 3,
        "superlative": 4,
    }

    out_path = out_dir / f"hop_labels_{dataset}.json"
    if out_path.exists():
        print(f"  Already exists: {out_path}")
        return

    hop_labels = []

    if dataset == "webqsp":
        # All WebQSP questions are 1-2 hop — use 2 as label
        gnn_splits = ["train", "dev", "test"]
        for split in gnn_splits:
            p = GNN_DATA / "webqsp" / f"{split}.json"
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        hop_labels.append({"question": d.get("question", ""), "hops": 2})
                    except Exception:
                        pass
        # Add perturbed questions (same hop count as original)
        for pt in PERT_TYPES:
            pfile = PERT_DIR / f"webqsp_{pt}.json"
            if pfile.exists():
                for rec in json.load(open(pfile)):
                    hop_labels.append({
                        "question": rec.get("perturbed_question", ""),
                        "hops": 2,
                    })

    elif dataset == "cwq":
        # CWQ has complexity field
        gnn_splits = ["train", "dev", "test"]
        for split in gnn_splits:
            p = GNN_DATA / "CWQ" / f"{split}.json"
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        # GNN-RAG CWQ doesn't have complexity directly — default to 3
                        hop_labels.append({"question": d.get("question", ""), "hops": 3})
                    except Exception:
                        pass
        # Add perturbed questions
        for pt in PERT_TYPES:
            pfile = PERT_DIR / f"cwq_{pt}.json"
            if pfile.exists():
                for rec in json.load(open(pfile)):
                    hop_labels.append({
                        "question": rec.get("perturbed_question", ""),
                        "hops": 3,
                    })

    # Deduplicate by question text
    seen = set()
    deduped = []
    for h in hop_labels:
        q = h["question"].strip()
        if q and q not in seen:
            seen.add(q)
            deduped.append({"question": q, "hops": h["hops"]})

    with open(out_path, "w") as f:
        json.dump(deduped, f, indent=2)
    print(f"  {dataset}: {len(deduped)} hop labels → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build LightPROF data pipeline")
    parser.add_argument(
        "--step",
        choices=["kg", "qa", "hops", "all"],
        default="all",
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--dataset",
        choices=["webqsp", "cwq", "both"],
        default="both",
        help="Dataset to process (for qa/hops steps)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = ["webqsp", "cwq"] if args.dataset == "both" else [args.dataset]

    if args.step in ("kg", "all"):
        build_kg(OUT_DIR)

    if args.step in ("qa", "all"):
        # Shared MID→name cache populated across both datasets
        mid2name_global: dict[str, str] = {}
        for ds in datasets:
            build_qa_jsonl(ds, OUT_DIR, mid2name_global)

    if args.step in ("hops", "all"):
        for ds in datasets:
            build_hop_labels(ds, OUT_DIR)

    print("\nDone. Output files in:", OUT_DIR)
    for f in sorted(OUT_DIR.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name}: {size:,} bytes")


if __name__ == "__main__":
    main()