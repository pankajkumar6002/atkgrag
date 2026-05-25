#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
build_entity_db.py — Build the SQLite entity database used by seed_entity_agent_llama.py
from LOCAL resources only (no HuggingFace download, no internet required).

Replaces prepare_data_llama.py entirely.

Sources:
  1. Pre-process/mid_cache/qid_to_mid.pkl  →  freebase_wikidata table (QID→MID, 2.1M rows)
  2. tmlr/resources/entities_sup_names.tsv  →  entity_names table (name→MID, 16.2M rows)
     Keys both by exact name AND by lowercased name for fast offline search_entity() calls.

Output:
  tmlr/resources/entity_mappings.db   (SQLite, ~2-4 GB)

Usage:
  python build_entity_db.py
  python build_entity_db.py --db-path /custom/path/entity_mappings.db
  python build_entity_db.py --skip-names   # only load freebase_wikidata table
"""

from __future__ import annotations

import argparse
import pickle
import sqlite3
from pathlib import Path

REPO_ROOT = REPO_ROOT
RESOURCES = REPO_ROOT / "tmlr" / "resources"

QID_TO_MID_PKL     = REPO_ROOT / "Pre-process/mid_cache/qid_to_mid.pkl"
ENTITIES_SUP_NAMES = RESOURCES / "entities_sup_names.tsv"
DEFAULT_DB_PATH    = RESOURCES / "entity_mappings.db"


def build_db(db_path: Path, skip_names: bool) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()

    # ── Schema ────────────────────────────────────────────────────────────────
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS freebase_wikidata (
            freebase_mid TEXT PRIMARY KEY,
            wikidata_qid TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_fb_qid ON freebase_wikidata(wikidata_qid);

        CREATE TABLE IF NOT EXISTS entity_names (
            mid          TEXT NOT NULL,
            entity_name  TEXT NOT NULL,
            name_lower   TEXT NOT NULL,
            PRIMARY KEY (mid, entity_name)
        );
        CREATE INDEX IF NOT EXISTS idx_en_lower ON entity_names(name_lower);
        CREATE INDEX IF NOT EXISTS idx_en_mid   ON entity_names(mid);
    """)
    conn.commit()

    # ── Table 1: freebase_wikidata from qid_to_mid.pkl ────────────────────────
    print(f"Loading {QID_TO_MID_PKL} …", flush=True)
    with open(QID_TO_MID_PKL, "rb") as fh:
        qid_to_mid: dict = pickle.load(fh)
    print(f"  {len(qid_to_mid):,} QID→MID entries", flush=True)

    batch, n = [], 0
    for qid, mid in qid_to_mid.items():
        # Normalise: store mid as m.XXXX (GNN-RAG format), not /m/XXXX
        mid_norm = mid.lstrip("/").replace("/", ".")  # /m/0695j → m.0695j
        # Normalise QID: ensure no URI prefix
        qid_norm = qid.split("/")[-1] if "/" in qid else qid
        batch.append((mid_norm, qid_norm))
        n += 1
        if len(batch) >= 50_000:
            cur.executemany(
                "INSERT OR REPLACE INTO freebase_wikidata VALUES (?,?)", batch
            )
            conn.commit()
            batch = []
            if n % 500_000 == 0:
                print(f"  {n:,} rows inserted …", flush=True)

    if batch:
        cur.executemany("INSERT OR REPLACE INTO freebase_wikidata VALUES (?,?)", batch)
        conn.commit()
    print(f"  Done: {n:,} freebase_wikidata rows", flush=True)

    if skip_names:
        conn.close()
        print(f"Saved → {db_path}", flush=True)
        return

    # ── Table 2: entity_names from entities_sup_names.tsv ────────────────────
    print(f"\nLoading {ENTITIES_SUP_NAMES} …", flush=True)
    if not ENTITIES_SUP_NAMES.exists():
        print(f"  [WARN] {ENTITIES_SUP_NAMES} not found — skipping entity_names table.", flush=True)
        conn.close()
        return

    batch, n = [], 0
    with open(ENTITIES_SUP_NAMES, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            mid, name = parts
            name_lower = name.lower().strip()
            batch.append((mid, name, name_lower))
            n += 1
            if len(batch) >= 100_000:
                cur.executemany(
                    "INSERT OR REPLACE INTO entity_names VALUES (?,?,?)", batch
                )
                conn.commit()
                batch = []
                if n % 2_000_000 == 0:
                    print(f"  {n:,} rows inserted …", flush=True)

    if batch:
        cur.executemany("INSERT OR REPLACE INTO entity_names VALUES (?,?,?)", batch)
        conn.commit()
    print(f"  Done: {n:,} entity_names rows", flush=True)

    conn.close()
    size_gb = db_path.stat().st_size / (1 << 30)
    print(f"\nSaved → {db_path}  ({size_gb:.2f} GB)", flush=True)
    print("entity_mappings.db is ready for seed_entity_agent_llama.py --offline-mode", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--skip-names", action="store_true",
                        help="Only build freebase_wikidata table (fast, ~45 MB)")
    args = parser.parse_args()
    build_db(Path(args.db_path), args.skip_names)


if __name__ == "__main__":
    main()