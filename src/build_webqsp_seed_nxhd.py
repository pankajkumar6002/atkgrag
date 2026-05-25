#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_webqsp_seed_nxhd.py — Build per-entity nxhd files for WebQSP ELQ seeds.

Background:
  The existing WebQSP nxhd files (stagg.neighborhoods/WebQTest-N.nxhd) are indexed
  by question ID, not by entity MID.  When ELQ predicts a DIFFERENT seed entity on
  a perturbed question, the subgraph analysis falls back to the same question-indexed
  file → Jaccard = 1.0 (artificially identical subgraphs).

  This script builds per-entity nxhd files analogous to CWQ's new_seed_nxhd/{mid}.nxhd,
  sourced from neighborhoods_webqsp.pkl (1.4 GB, keyed by MID).

Inputs:
  • data/perturbations/el_elq/webqsp_p{1-6}_el.json  — ELQ-predicted MIDs
  • Pre-process/mid_cache/neighborhoods_webqsp.pkl   — MID → subgraph dict
  • (fallback) Pre-process/webqsp_2hop_nxhd/dataset/freebase_2hops/stagg.neighborhoods/
    — per-question nxhd files (when MID not in PKL)

Output:
  • Pre-process/webqsp_seed_nxhd/{mid}.nxhd   — one file per unique ELQ seed MID

Usage:
  python build_webqsp_seed_nxhd.py                 # build all 3082 MIDs
  python build_webqsp_seed_nxhd.py --dry-run       # print stats, no write
  python build_webqsp_seed_nxhd.py --check-pkl     # check PKL coverage only
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = REPO_ROOT
TMLR      = REPO_ROOT / "tmlr"
PREPROC   = REPO_ROOT / "Pre-process"

EL_DIR        = TMLR / "pert" / "dataset" / "el_elq"
PKL_PATH      = PREPROC / "mid_cache" / "neighborhoods_webqsp.pkl"
PKL_CWQ_PATH  = PREPROC / "mid_cache" / "neighborhoods_cwq.pkl"
STAGG_DIR     = PREPROC / "webqsp_2hop_nxhd" / "dataset" / "freebase_2hops" / "stagg.neighborhoods"
FB_EN_PATH    = TMLR / "resources" / "fb_en.txt"   # 920M lines, tab-sep: mid\trel\tobj
OUT_DIR       = PREPROC / "webqsp_seed_nxhd"


# ── Collect unique ELQ-predicted MIDs from all WebQSP p1-p6 files ─────────────

def collect_elq_mids() -> Set[str]:
    mids: Set[str] = set()
    for pt in ["orig", "p1", "p2", "p3", "p4", "p5", "p6"]:
        f = EL_DIR / f"webqsp_{pt}_el.json"
        if not f.exists():
            print(f"  [SKIP] {f.name} not found")
            continue
        recs = json.load(open(f))
        for r in recs:
            for e in r.get("linked_entities", []):
                m = e.get("mid", "").strip()
                if m:
                    mids.add(m)
    return mids


# ── PKL subgraph format → nxhd lines ─────────────────────────────────────────

def subgraph_dict_to_nxhd_lines(subgraph: Dict) -> List[str]:
    """Convert a neighborhoods_webqsp.pkl subgraph dict to nxhd-format lines.

    PKL format: { 'entities': [mid, ...], 'tuples': [(subj, rel, obj), ...] }
    nxhd output format: <fb:subj>\\t<fb:rel>\\t<fb:obj>
    """
    lines: List[str] = []
    if not subgraph:
        return lines

    # Primary format: dict with 'tuples' key (webqsp PKL)
    if "tuples" in subgraph:
        for triple in subgraph["tuples"]:
            if len(triple) == 3:
                s, r, o = triple
                lines.append(f"<fb:{s}>\t<fb:{r}>\t<fb:{o}>")
        return lines

    # Fallback: nested dict keyed by subject MID (cwq PKL format)
    for subj, data in subgraph.items():
        if not isinstance(data, dict):
            continue
        edges = data.get("e") or data.get("edges") or []
        for entry in edges:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                rel, obj = entry
                lines.append(f"<fb:{subj}>\t<fb:{rel}>\t<fb:{obj}>")
            elif isinstance(entry, dict):
                rel = entry.get("r", entry.get("rel", ""))
                obj = entry.get("e", entry.get("obj", ""))
                if rel and obj:
                    lines.append(f"<fb:{subj}>\t<fb:{rel}>\t<fb:{obj}>")

    return lines


# ── Main extraction logic ─────────────────────────────────────────────────────

def build_nxhd_files(dry_run: bool = False, check_pkl: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: collect all ELQ MIDs
    print("Collecting ELQ-predicted MIDs from webqsp el_elq files...")
    all_mids = collect_elq_mids()
    print(f"  Total unique MIDs: {len(all_mids):,}")

    # Step 2: skip already-built files
    already_done = {f.stem for f in OUT_DIR.glob("*.nxhd")}
    pending = sorted(all_mids - already_done)
    print(f"  Already done: {len(already_done)}  |  Pending: {len(pending)}")

    if check_pkl:
        print("\nLoading PKL to check coverage...")
        t0 = time.time()
        with open(PKL_PATH, "rb") as f:
            pkl = pickle.load(f)
        print(f"  PKL loaded: {len(pkl):,} entries ({time.time()-t0:.1f}s)")
        in_pkl = sum(1 for m in pending if m in pkl)
        print(f"  Pending MIDs in PKL: {in_pkl}/{len(pending)} ({100*in_pkl/max(1,len(pending)):.1f}%)")
        return

    if dry_run:
        print(f"\nDRY RUN — would build {len(pending)} nxhd files → {OUT_DIR}")
        print("Sample pending MIDs:", pending[:10])
        return

    if not pending:
        print("Nothing to do — all MIDs already built.")
        return

    # Step 3: load WebQSP PKL
    print(f"\nLoading neighborhoods_webqsp.pkl ({PKL_PATH.stat().st_size/1e9:.2f} GB)...")
    t0 = time.time()
    with open(PKL_PATH, "rb") as f:
        pkl: Dict = pickle.load(f)
    print(f"  Loaded: {len(pkl):,} entries ({time.time()-t0:.1f}s)")

    # Step 3b: load CWQ PKL as fallback for MIDs missing from WebQSP PKL
    pkl_cwq: Optional[Dict] = None
    if PKL_CWQ_PATH.exists():
        print(f"Loading neighborhoods_cwq.pkl as fallback ({PKL_CWQ_PATH.stat().st_size/1e9:.2f} GB)...")
        t0b = time.time()
        with open(PKL_CWQ_PATH, "rb") as f:
            pkl_cwq = pickle.load(f)
        print(f"  Loaded: {len(pkl_cwq):,} entries ({time.time()-t0b:.1f}s)")

    # Step 4: build nxhd files
    n_from_pkl = 0
    n_from_cwq = 0
    n_missing = 0
    missing_mids: List[str] = []

    t0 = time.time()
    for i, mid in enumerate(pending):
        out_path = OUT_DIR / f"{mid}.nxhd"

        # Try WebQSP PKL first
        subgraph = pkl.get(mid)
        source = "webqsp_pkl"
        if subgraph is None and pkl_cwq is not None:
            subgraph = pkl_cwq.get(mid)
            source = "cwq_pkl"

        if subgraph is not None:
            lines = subgraph_dict_to_nxhd_lines(subgraph)
            with open(out_path, "w", encoding="utf-8") as fh:
                if lines:
                    fh.write("\n".join(lines) + "\n")
            if source == "webqsp_pkl":
                n_from_pkl += 1
            else:
                n_from_cwq += 1
        else:
            missing_mids.append(mid)
            n_missing += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(pending)}] webqsp_pkl={n_from_pkl}  cwq_pkl={n_from_cwq}  "
                  f"missing={n_missing}  ({elapsed:.0f}s, {(i+1)/elapsed:.0f}/s)")

    # Step 5: report
    elapsed = time.time() - t0
    total_written = n_from_pkl + n_from_cwq
    print(f"\nDone: {total_written} written (webqsp_pkl={n_from_pkl}, cwq_pkl={n_from_cwq}), "
          f"{n_missing} missing from both PKLs ({elapsed:.1f}s)")
    if missing_mids:
        missing_out = TMLR / "logs" / "webqsp_seed_missing_mids.txt"
        missing_out.parent.mkdir(exist_ok=True)
        with open(missing_out, "w") as f:
            f.write("\n".join(missing_mids))
        print(f"Missing MIDs saved → {missing_out}")
        print("Sample missing:", missing_mids[:10])

        # Optional: try fb_en.txt grep for missing MIDs (1-hop only; 2-hop too slow
        # without a pre-built sorted index on the 920M-line file).
        if FB_EN_PATH.exists():
            print(f"\nAttempting fb_en.txt 1-hop fallback for {len(missing_mids)} missing MIDs...")
            print("  (Note: 2-hop requires sorted index; providing 1-hop outgoing triples only)")
            n_fb = 0
            for mid in missing_mids:
                out_path = OUT_DIR / f"{mid}.nxhd"
                if out_path.exists():
                    continue
                try:
                    import subprocess
                    result = subprocess.run(
                        ["grep", "-a", f"^{mid}\t", str(FB_EN_PATH)],
                        capture_output=True, text=True, timeout=120
                    )
                    if result.stdout:
                        lines = []
                        for line in result.stdout.strip().split("\n"):
                            parts = line.split("\t", 2)
                            if len(parts) == 3:
                                s, r, o = parts
                                lines.append(f"<fb:{s}>\t<fb:{r}>\t<fb:{o}>")
                        if lines:
                            with open(out_path, "w") as fh:
                                fh.write("\n".join(lines) + "\n")
                            n_fb += 1
                except Exception as e:
                    pass
            print(f"  fb_en.txt fallback: {n_fb}/{len(missing_mids)} MIDs resolved (1-hop only)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run",   action="store_true", help="Print stats, don't write files")
    ap.add_argument("--check-pkl", action="store_true", help="Check PKL coverage only")
    args = ap.parse_args()

    build_nxhd_files(dry_run=args.dry_run, check_pkl=args.check_pkl)


if __name__ == "__main__":
    main()