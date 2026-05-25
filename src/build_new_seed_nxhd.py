#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
Build .nxhd neighborhood files for new ELQ seed MIDs not in original GNN-RAG seeds.

Source priority:
  Tier 1 — neighborhoods_cwq.pkl  (keyed by MID → {entities, tuples})
  Tier 2 — neighborhoods_webqsp.pkl
  Tier 3 — data/resources/manual_fb_filter.txt  (one 39 GB pass for all missing MIDs)

Output: data/new_seed_nxhd/{mid}.nxhd
        Tab-separated: <fb:subject>\t<fb:predicate>\tobject
        Subject is always the seed MID (and triples where seed appears as object are also included).

Usage:
  python3 src/build_new_seed_nxhd.py [--mids-file PATH] [--out-dir PATH] [--dry-run]
"""

import argparse
import gzip
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT =  Path(__file__).resolve().parents[2]  # 

MIDS_FILE   = REPO_ROOT / "data" / "new_seed_nxhd" / "new_mids.txt"
OUT_DIR     = REPO_ROOT / "data" / "new_seed_nxhd"
CWQ_PKL     = REPO_ROOT / "data" / "mid_cache" / "neighborhoods_cwq.pkl"
WQ_PKL      = REPO_ROOT / "data" / "mid_cache" / "neighborhoods_webqsp.pkl"
FB_FILTER   = REPO_ROOT / "data" / "resources" / "manual_fb_filter.txt"

# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------

def _wrap(token: str) -> str:
    """Wrap a Freebase MID or predicate in <fb:...> if it looks like one."""
    if token.startswith("m.") or token.startswith("g.") or "." in token:
        return f"<fb:{token}>"
    return token


def tuples_to_nxhd_lines(triples: List[Tuple[str, str, str]]) -> List[str]:
    """Convert (s, p, o) triples to nxhd tab-separated lines."""
    lines = []
    for s, p, o in triples:
        s_str = f"<fb:{s}>" if (s.startswith("m.") or s.startswith("g.")) else s
        p_str = f"<fb:{p}>"
        # Objects: MID → wrap, else keep as literal
        if o.startswith("m.") or o.startswith("g."):
            o_str = f"<fb:{o}>"
        else:
            o_str = o
        lines.append(f"{s_str}\t{p_str}\t{o_str}")
    return lines


def load_pkl_index(pkl_path: Path) -> Dict[str, List[Tuple]]:
    """Load PKL and return {mid: list_of_(s,p,o)_tuples}."""
    print(f"  Loading PKL: {pkl_path.name} ...", flush=True)
    t0 = time.time()
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"    {len(data):,} entries in {time.time()-t0:.1f}s")
    # Normalise to {mid: [(s,p,o), ...]}
    result: Dict[str, List[Tuple]] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            triples = v.get("tuples", [])
        elif isinstance(v, (list, tuple)):
            triples = list(v)
        else:
            triples = []
        result[k] = triples
    return result


def write_nxhd(mid: str, triples: List[Tuple], out_dir: Path) -> int:
    """Write triples to {mid}.nxhd; returns number of triples written."""
    lines = tuples_to_nxhd_lines(triples)
    if not lines:
        return 0
    out_path = out_dir / f"{mid}.nxhd"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mids-file", default=str(MIDS_FILE))
    parser.add_argument("--out-dir",   default=str(OUT_DIR))
    parser.add_argument("--dry-run",   action="store_true",
                        help="Process only first 20 MIDs; skip manual_fb_filter.txt tier")
    args = parser.parse_args()

    mids_file = Path(args.mids_file)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load target MID list
    mids: List[str] = [l.strip() for l in mids_file.read_text().splitlines() if l.strip()]
    print(f"Target MIDs: {len(mids):,}")
    if args.dry_run:
        mids = mids[:20]
        print(f"  [dry-run] Limiting to {len(mids)} MIDs")

    # Skip MIDs that already have output files
    pending = [m for m in mids if not (out_dir / f"{m}.nxhd").exists()]
    print(f"Already done: {len(mids) - len(pending):,} | Pending: {len(pending):,}")
    if not pending:
        print("Nothing to do.")
        return

    # ---- Tier 1: CWQ PKL ----
    cwq_idx = load_pkl_index(CWQ_PKL)
    found_cwq: Set[str] = set()
    for mid in list(pending):
        if mid in cwq_idx and cwq_idx[mid]:
            write_nxhd(mid, cwq_idx[mid], out_dir)
            found_cwq.add(mid)
    pending = [m for m in pending if m not in found_cwq]
    print(f"Tier 1 (CWQ PKL): found {len(found_cwq):,} | remaining {len(pending):,}")
    del cwq_idx

    # ---- Tier 2: WebQSP PKL ----
    wq_idx = load_pkl_index(WQ_PKL)
    found_wq: Set[str] = set()
    for mid in list(pending):
        if mid in wq_idx and wq_idx[mid]:
            write_nxhd(mid, wq_idx[mid], out_dir)
            found_wq.add(mid)
    pending = [m for m in pending if m not in found_wq]
    print(f"Tier 2 (WebQSP PKL): found {len(found_wq):,} | remaining {len(pending):,}")
    del wq_idx

    if not pending:
        print("All MIDs resolved via PKL tiers. Done.")
        return

    if args.dry_run:
        print(f"[dry-run] Skipping manual_fb_filter.txt (tier 3). {len(pending)} MIDs unresolved:")
        for m in pending:
            print(f"  {m}")
        return

    # ---- Tier 3: manual_fb_filter.txt (single pass) ----
    if not FB_FILTER.exists():
        print(f"ERROR: {FB_FILTER} not found. {len(pending)} MIDs remain unresolved.")
        return

    pending_set = set(pending)
    # Accumulate triples per MID
    mid_triples: Dict[str, List[Tuple]] = {m: [] for m in pending_set}
    print(f"Tier 3: scanning {FB_FILTER} ({FB_FILTER.stat().st_size / 1e9:.1f} GB) "
          f"for {len(pending_set):,} MIDs ...", flush=True)

    t0 = time.time()
    lines_read = 0
    matched = 0
    report_every = 10_000_000

    with open(FB_FILTER, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines_read += 1
            if lines_read % report_every == 0:
                elapsed = time.time() - t0
                print(f"  {lines_read/1e6:.0f}M lines in {elapsed:.0f}s  matched={matched:,}",
                      flush=True)
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            s, p, o = parts[0], parts[1], parts[2]
            if s in pending_set:
                mid_triples[s].append((s, p, o))
                matched += 1
            # Also capture triples where seed appears as object
            if o in pending_set:
                mid_triples[o].append((s, p, o))
                matched += 1

    elapsed = time.time() - t0
    print(f"Scan complete: {lines_read/1e6:.1f}M lines in {elapsed:.0f}s  matched={matched:,}")

    found_fb = 0
    empty_fb = 0
    for mid, triples in mid_triples.items():
        if triples:
            write_nxhd(mid, triples, out_dir)
            found_fb += 1
        else:
            # Write empty file as marker so we know it was processed
            (out_dir / f"{mid}.nxhd").touch()
            empty_fb += 1

    print(f"Tier 3 (manual_fb_filter.txt): found={found_fb:,} empty={empty_fb:,}")

    # Summary
    total_done = len(mids) - len([m for m in mids if not (out_dir / f"{m}.nxhd").exists()])
    print(f"\nDone. {total_done:,}/{len(mids):,} MIDs have .nxhd files in {out_dir}")


if __name__ == "__main__":
    main()