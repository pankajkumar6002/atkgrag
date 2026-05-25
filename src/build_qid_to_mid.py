#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_qid_to_mid.py — Build QID→MID and MID→QID mapping from fb2w.nt.gz.

fb2w.nt.gz contains N-triples of the form:
  <http://rdf.freebase.com/ns/m.XXXX> owl:sameAs <http://www.wikidata.org/entity/QYYY>

Outputs:
  Pre-process/mid_cache/qid_to_mid.pkl   {QID_str → MID_str}  e.g. "Q123" → "m.0abc"
  Pre-process/mid_cache/mid_to_qid.pkl   {MID_str → QID_str}  (inverse)

Usage:
  python build_qid_to_mid.py
  python build_qid_to_mid.py --fb2w /path/to/fb2w.nt.gz --out-dir /path/to/cache
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import re
from pathlib import Path

FB2W_PATH = REPO_ROOT / "data" / "resources" / "fb2w.nt.gz"
OUT_DIR = REPO_ROOT / "data" / "mid_cache"

# Regex patterns
_MID_RE = re.compile(r"<http://rdf\.freebase\.com/ns/(m\.\w+)>")
_QID_RE = re.compile(r"<http://www\.wikidata\.org/entity/(Q\d+)>")


def build_mappings(fb2w_path: Path) -> tuple[dict, dict]:
    qid_to_mid: dict[str, str] = {}
    mid_to_qid: dict[str, str] = {}
    n_lines = 0
    n_matched = 0

    print(f"Parsing {fb2w_path} ...")
    with gzip.open(fb2w_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            mid_m = _MID_RE.search(line)
            qid_m = _QID_RE.search(line)
            if mid_m and qid_m:
                mid = mid_m.group(1)   # e.g. "m.0abc"
                qid = qid_m.group(1)   # e.g. "Q123"
                qid_to_mid[qid] = mid
                mid_to_qid[mid] = qid
                n_matched += 1

            if n_lines % 1_000_000 == 0:
                print(f"  lines={n_lines:,}  matched={n_matched:,}")

    print(f"Done. lines={n_lines:,}  matched={n_matched:,}")
    return qid_to_mid, mid_to_qid


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QID↔MID mapping from fb2w.nt.gz")
    parser.add_argument("--fb2w", type=Path, default=FB2W_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    qid_to_mid, mid_to_qid = build_mappings(args.fb2w)

    qid_path = args.out_dir / "qid_to_mid.pkl"
    mid_path = args.out_dir / "mid_to_qid.pkl"

    with open(qid_path, "wb") as f:
        pickle.dump(qid_to_mid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(qid_to_mid):,} entries → {qid_path}")

    with open(mid_path, "wb") as f:
        pickle.dump(mid_to_qid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(mid_to_qid):,} entries → {mid_path}")


if __name__ == "__main__":
    main()