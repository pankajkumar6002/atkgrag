#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_entities_sup.py — Build a superset entity vocabulary from four sources, then
extract Freebase English name strings for every entity in that set.

Sources (union, deduplicated):
  1. tmlr/Repo/UniKGQA/Preprocess/data/test/webqsp_test_2hop/entities.txt  (m.XXXX, one/line)
  2. tmlr/Repo/GNN-RAG/data/webqsp/entities.txt                             (m.XXXX)
  3. tmlr/Repo/GNN-RAG/data/CWQ/entities.txt                               (m.XXXX)
  4. freebase/data/dataset/freebase_2hops/all_entities                     (<fb:m.XXXX>)

Output:
  tmlr/resources/entities_sup.txt         — one normalised MID per line (m.XXXX), sorted
  tmlr/resources/entities_sup_names.tsv   — mid<TAB>name  (first name per MID)

Name sources (checked in order):
  1. tmlr/resources/fb_en.txt             — TSV triples (mid  predicate  value), 920M lines
  2. data/resources/manual_fb_filter.txt  — same format (fallback), 688M lines
  3. SPARQL batch queries on localhost:3001 for any still-missing MIDs (English filter)

Parallel scan:
  Both TSV files are scanned in parallel using multiprocessing.  Each worker is given a
  byte-range chunk of the file; it aligns to the next newline boundary then scans until
  its end offset.  Results are merged in the main process (first name wins per MID).
  Default workers = min(32, cpu_count()) to balance I/O bandwidth and CPU.

Usage:
  python build_entities_sup.py               # full run (all steps)
  python build_entities_sup.py --ids-only    # only build entities_sup.txt
  python build_entities_sup.py --names-only  # load existing IDs, scan for names
  python build_entities_sup.py --sparql-fill # fill still-missing names via SPARQL
  python build_entities_sup.py --workers N   # override parallel worker count
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from multiprocessing import Manager, Pool, cpu_count, shared_memory
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = REPO_ROOT
RESOURCES = REPO_ROOT / "tmlr" / "resources"

ENTITY_SOURCES = [
    REPO_ROOT / "tmlr/Repo/UniKGQA/Preprocess/data/test/webqsp_test_2hop/entities.txt",
    REPO_ROOT / "tmlr/Repo/GNN-RAG/data/webqsp/entities.txt",
    REPO_ROOT / "tmlr/Repo/GNN-RAG/data/CWQ/entities.txt",
    REPO_ROOT / "freebase/data/dataset/freebase_2hops/all_entities",
]
FB_EN_TXT       = RESOURCES / "fb_en.txt"
MANUAL_FB       = RESOURCES / "manual_fb_filter.txt"
SPARQL_ENDPOINT = "http://localhost:3001/sparql"

OUT_IDS   = RESOURCES / "entities_sup.txt"
OUT_NAMES = RESOURCES / "entities_sup_names.tsv"

NAME_PRED = b"type.object.name"   # bytes for fast binary scan

_MID_RE = re.compile(r"m\.[0-9a-zA-Z_]+")


# ── Step 1: Build entities_sup.txt ────────────────────────────────────────────

def build_entity_set() -> Set[str]:
    entity_set: Set[str] = set()
    for src in ENTITY_SOURCES:
        if not src.exists():
            print(f"  [WARN] Not found: {src}", flush=True)
            continue
        before = len(entity_set)
        with open(src, "rb") as fh:
            for raw in fh:
                line = raw.decode("utf-8", errors="replace").strip()
                m = _MID_RE.search(line)
                if m:
                    entity_set.add(m.group())
        print(f"  {src.name}: +{len(entity_set)-before:,}  (total {len(entity_set):,})", flush=True)
    return entity_set


def save_entity_set(entity_set: Set[str]) -> None:
    OUT_IDS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_IDS, "w") as fh:
        for mid in sorted(entity_set):
            fh.write(mid + "\n")
    print(f"Saved {len(entity_set):,} unique MIDs → {OUT_IDS}", flush=True)


def load_entity_set() -> Set[str]:
    with open(OUT_IDS) as fh:
        return set(line.strip() for line in fh if line.strip())


# ── Step 2: Parallel name extraction ─────────────────────────────────────────

def _chunk_offsets(file_size: int, n_workers: int) -> List[Tuple[int, int]]:
    """Divide file_size into n_workers roughly equal byte ranges."""
    chunk = file_size // n_workers
    offsets = []
    start = 0
    for i in range(n_workers):
        end = file_size if i == n_workers - 1 else start + chunk
        offsets.append((start, end))
        start = end
    return offsets


def _scan_chunk(args: Tuple) -> Dict[str, str]:
    """Worker: scan [start_off, end_off) of filepath for type.object.name triples.

    Returns {mid: name} for every matching triple whose MID is in entity_set_file.
    entity_set is passed as a path to a sorted text file to avoid pickling a huge set.
    """
    filepath, start_off, end_off, entity_set_file, worker_id, total_workers = args

    # Load entity set in each worker (workers share the kernel page cache after
    # the first load, so this is effectively free after the first worker reads it)
    entity_set: Set[str] = set()
    with open(entity_set_file) as fh:
        for line in fh:
            s = line.strip()
            if s:
                entity_set.add(s)

    results: Dict[str, str] = {}
    file_size = os.path.getsize(filepath)

    with open(filepath, "rb") as fh:
        # Align to next line boundary
        if start_off > 0:
            fh.seek(start_off)
            fh.readline()   # skip partial first line
        else:
            fh.seek(0)

        processed = 0
        report_every = 10_000_000

        while True:
            pos = fh.tell()
            if pos >= end_off:
                break
            raw = fh.readline()
            if not raw:
                break
            processed += 1
            if processed % report_every == 0:
                pct = 100.0 * pos / file_size
                print(
                    f"  [worker {worker_id}/{total_workers}] {processed//1_000_000}M lines  "
                    f"pos={pos//(1<<30)}GB/{file_size//(1<<30)}GB ({pct:.1f}%)  "
                    f"found={len(results):,}",
                    flush=True,
                )

            if NAME_PRED not in raw:
                continue

            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                continue

            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            mid = parts[0].rstrip()
            pred = parts[1].rstrip()
            val  = parts[2].rstrip()

            if pred != "type.object.name":
                continue
            if mid not in entity_set:
                continue
            if mid not in results:
                results[mid] = val

    print(
        f"  [worker {worker_id}/{total_workers}] done — {len(results):,} names in chunk",
        flush=True,
    )
    return results


def _scan_file_parallel(
    path: Path,
    n_workers: int,
    label: str,
    already_named: Set[str],
) -> Dict[str, str]:
    """Scan path in parallel using n_workers processes. Returns {mid: name}."""
    file_size = path.stat().st_size
    gb = file_size / (1 << 30)
    print(f"\nScanning {label} ({gb:.1f} GB) with {n_workers} workers …", flush=True)

    offsets = _chunk_offsets(file_size, n_workers)
    tasks = [
        (str(path), start, end, str(OUT_IDS), i+1, n_workers)
        for i, (start, end) in enumerate(offsets)
    ]

    merged: Dict[str, str] = {}
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        for chunk_result in pool.imap_unordered(_scan_chunk, tasks):
            for mid, name in chunk_result.items():
                if mid not in merged and mid not in already_named:
                    merged[mid] = name

    elapsed = time.time() - t0
    print(
        f"  {label}: {len(merged):,} names extracted in {elapsed:.0f}s "
        f"({gb/elapsed:.2f} GB/s throughput)",
        flush=True,
    )
    return merged


def extract_names_parallel(
    entity_set: Set[str],
    n_workers: int,
    existing_names: Dict[str, str],
) -> Dict[str, str]:
    names = dict(existing_names)  # resume-safe: start from already known

    # Pass 1: fb_en.txt
    if FB_EN_TXT.exists():
        chunk = _scan_file_parallel(FB_EN_TXT, n_workers, "fb_en.txt", set(names.keys()))
        names.update(chunk)
        missing = len(entity_set) - len(names)
        print(f"After fb_en.txt: {len(names):,} named, {missing:,} still missing", flush=True)
    else:
        print(f"[WARN] {FB_EN_TXT} not found", flush=True)

    # Pass 2: manual_fb_filter.txt (for anything still missing)
    missing_set = entity_set - names.keys()
    if missing_set and MANUAL_FB.exists():
        print(f"Fallback scan of manual_fb_filter.txt for {len(missing_set):,} missing …", flush=True)
        chunk2 = _scan_file_parallel(MANUAL_FB, n_workers, "manual_fb_filter.txt", set(names.keys()))
        names.update(chunk2)
        print(f"After manual_fb_filter.txt: {len(names):,} named", flush=True)
    elif missing_set:
        print(f"[WARN] {MANUAL_FB} not found — skipping fallback", flush=True)

    return names


def save_names(names: Dict[str, str]) -> None:
    OUT_NAMES.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_NAMES, "w", encoding="utf-8") as fh:
        for mid in sorted(names):
            fh.write(f"{mid}\t{names[mid]}\n")
    print(f"Saved {len(names):,} (mid, name) pairs → {OUT_NAMES}", flush=True)


# ── Step 3: SPARQL fill for remaining ────────────────────────────────────────

def _sparql_batch_worker(args: Tuple) -> Dict[str, str]:
    """Query a single batch of MIDs from port-3001 Freebase. Returns {mid: name}."""
    from SPARQLWrapper import JSON as SPARQL_JSON, SPARQLWrapper  # type: ignore
    chunk, endpoint, timeout = args
    FB_NS = "http://rdf.freebase.com/ns/"
    NAME_PRED_FULL = f"{FB_NS}type.object.name"
    values = " ".join(f"<{FB_NS}{m}>" for m in chunk)
    query = (
        "SELECT ?mid ?name WHERE {"
        f" VALUES ?mid {{ {values} }}"
        f" ?mid <{NAME_PRED_FULL}> ?name ."
        " FILTER(langMatches(lang(?name), 'en') || lang(?name) = '')"
        f"}} LIMIT {len(chunk) * 3}"
    )
    results: Dict[str, str] = {}
    try:
        client = SPARQLWrapper(endpoint)
        client.setTimeout(timeout)
        client.setQuery(query)
        client.setReturnFormat(SPARQL_JSON)
        data = client.query().convert()
        for binding in data["results"]["bindings"]:
            uri = binding["mid"]["value"]
            mid = uri.replace(FB_NS, "")
            name = binding["name"]["value"]
            if mid not in results:
                results[mid] = name
    except Exception:
        pass
    return results


def sparql_fill_parallel(
    entity_set: Set[str],
    names: Dict[str, str],
    n_workers: int = 32,
    batch_size: int = 500,
) -> int:
    missing = sorted(entity_set - names.keys())
    if not missing:
        print("No missing MIDs for SPARQL fill.", flush=True)
        return 0

    print(
        f"SPARQL fill: {len(missing):,} MIDs in batches of {batch_size} "
        f"using {n_workers} parallel workers …",
        flush=True,
    )
    chunks = [missing[i:i+batch_size] for i in range(0, len(missing), batch_size)]
    tasks  = [(chunk, SPARQL_ENDPOINT, 30) for chunk in chunks]

    added = 0
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_sparql_batch_worker, tasks), 1):
            for mid, name in result.items():
                if mid not in names:
                    names[mid] = name
                    added += 1
            if i % 200 == 0:
                elapsed = time.time() - t0
                print(
                    f"  {i}/{len(chunks)} batches done  "
                    f"({i*batch_size:,} queried)  {added:,} new names  "
                    f"({elapsed:.0f}s elapsed)",
                    flush=True,
                )

    print(f"SPARQL fill: added {added:,} names in {time.time()-t0:.0f}s", flush=True)
    return added


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build entities_sup.txt + entities_sup_names.tsv")
    parser.add_argument("--ids-only",    action="store_true")
    parser.add_argument("--names-only",  action="store_true", help="Load existing IDs; scan for names")
    parser.add_argument("--sparql-fill", action="store_true", help="Fill missing names via SPARQL")
    parser.add_argument("--no-sparql",   action="store_true", help="Skip SPARQL fill step")
    parser.add_argument(
        "--workers", type=int,
        default=min(32, cpu_count()),
        help=f"Parallel workers (default min(32, nproc)={min(32, cpu_count())})",
    )
    parser.add_argument("--sparql-workers", type=int, default=32,
                        help="Workers for SPARQL fill (default 32)")
    parser.add_argument("--sparql-batch",   type=int, default=500,
                        help="Batch size for SPARQL (default 500)")
    args = parser.parse_args()

    print(f"Workers: {args.workers}  |  CPUs available: {cpu_count()}", flush=True)

    # ── SPARQL-fill-only mode ─────────────────────────────────────────────────
    if args.sparql_fill:
        entity_set = load_entity_set()
        names: Dict[str, str] = {}
        if OUT_NAMES.exists():
            with open(OUT_NAMES) as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t", 1)
                    if len(parts) == 2:
                        names[parts[0]] = parts[1]
            print(f"Loaded {len(names):,} existing names", flush=True)
        sparql_fill_parallel(entity_set, names, n_workers=args.sparql_workers, batch_size=args.sparql_batch)
        save_names(names)
        return

    # ── Step 1: Entity IDs ────────────────────────────────────────────────────
    if args.names_only:
        print("Loading existing entities_sup.txt …", flush=True)
        entity_set = load_entity_set()
        print(f"  Loaded {len(entity_set):,} MIDs", flush=True)
    else:
        print("=== Step 1: Building entity superset ===", flush=True)
        entity_set = build_entity_set()
        save_entity_set(entity_set)

    if args.ids_only:
        return

    # ── Step 2: Parallel name extraction ─────────────────────────────────────
    existing_names: Dict[str, str] = {}
    if OUT_NAMES.exists():
        print(f"Resuming: loading existing {OUT_NAMES.name} …", flush=True)
        with open(OUT_NAMES) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    existing_names[parts[0]] = parts[1]
        print(f"  {len(existing_names):,} names already known", flush=True)

    print("=== Step 2: Parallel name extraction ===", flush=True)
    names = extract_names_parallel(entity_set, args.workers, existing_names)
    save_names(names)

    # ── Step 3: SPARQL fill ───────────────────────────────────────────────────
    if not args.no_sparql:
        missing_count = len(entity_set) - len(names)
        print(f"\n=== Step 3: SPARQL fill for {missing_count:,} missing names ===", flush=True)
        if missing_count > 0:
            sparql_fill_parallel(
                entity_set, names,
                n_workers=args.sparql_workers,
                batch_size=args.sparql_batch,
            )
            save_names(names)

    named = len(names)
    total = len(entity_set)
    print(f"\n=== Done ===", flush=True)
    print(f"  Entity set : {total:,}", flush=True)
    print(f"  Named      : {named:,}  ({100*named/total:.1f}%)", flush=True)
    print(f"  Unnamed    : {total - named:,}", flush=True)
    print(f"  → {OUT_IDS}", flush=True)
    print(f"  → {OUT_NAMES}", flush=True)


if __name__ == "__main__":
    main()