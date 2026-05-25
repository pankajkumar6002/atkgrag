#!/usr/bin/env python3
"""
Convert sr-cwq data from the 'simple_debug' format (dict entities + integer subgraph IDs)
to the NSM MID-string format used by SubgraphRetrievalKBQA/src/nsm_reader.

sr-cwq/train.json format:
  entities: [{kb_id: "m.xxx", text: "..."}]   <- dict
  subgraph.entities: [int, ...]                <- integer global IDs  (NOT indices into entities.txt)
  subgraph.tuples: [[int, "rel_str", int], ...]

The integers are NOT line-indices into entities.txt.  They come from a separate
global-entity numbering used during CWQ pre-processing.  We need to invert the
entities_cid mapping (entities_cid[i] = global_int for entities[i]) to build a
global_int -> MID dict, then also decode subgraph.entities the same way.

NSM target format (same as sr-webqsp):
  entities: ["m.xxx", ...]                     <- plain MID strings
  subgraph.entities: ["m.xxx", ...]            <- plain MID strings
  subgraph.tuples: [["m.xxx", "rel", "m.xxx"], ...]
  answers: [{kb_id: "m.xxx", text: "..."}]     <- unchanged
"""

import json
import os
import sys
from collections import defaultdict
from tqdm import tqdm

SRC = str(REPO_ROOT / "external" / "GNN-RAG" / "data" / "sr-cwq")
DST = str(REPO_ROOT / "external" / "SubgraphRetrievalKBQA" / "src" / "nsm_reader" / "data" / "cwq")
os.makedirs(DST, exist_ok=True)

# ── build global-int → MID mapping by scanning ALL three splits ──────────────
print("Building global_int → MID map from all splits…")
int_to_mid: dict[int, str] = {}

for split in ("train", "dev", "test"):
    path = os.path.join(SRC, f"{split}.json")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        continue
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip truncated/corrupt lines
            for ent_dict, cid in zip(r.get("entities", []),
                                     r.get("entities_cid", [])):
                mid = ent_dict["kb_id"]
                int_to_mid[cid] = mid
            for ans_dict, cid in zip(r.get("answers", []),
                                     r.get("answers_cid", [])):
                mid = ans_dict["kb_id"]
                int_to_mid[cid] = mid

print(f"  int→MID mapping: {len(int_to_mid):,} entries")

# ── also build a mapping from subgraph-integer edges to MIDs using the facts
#    that appear in train/dev/test.  Any integer not covered → skip triple.
# We need to scan subgraph.entities too, but those ints don't appear in
# entities_cid or answers_cid directly.  We can collect them per-record
# when we convert, and fall back to the int_to_mid we already have.

def convert_split(split: str) -> int:
    src_path = os.path.join(SRC, f"{split}.json")
    dst_path = os.path.join(DST, f"{split}_simple.json")
    if not os.path.exists(src_path):
        print(f"  [{split}] source not found, skipping")
        return 0

    kept = dropped_ent = dropped_tpl = 0
    with open(src_path) as fin, open(dst_path, "w") as fout:
        for line in tqdm(fin, desc=f"  {split}"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 1. entities: dict list → MID string list
            seed_mids = []
            for ent_dict in r.get("entities", []):
                mid = ent_dict.get("kb_id", "")
                if mid:
                    seed_mids.append(mid)

            if not seed_mids:
                dropped_ent += 1
                continue

            # 2. subgraph entities: int list → MID string list
            sg_int_ents = r.get("subgraph", {}).get("entities", [])
            sg_mid_ents = [int_to_mid[i] for i in sg_int_ents if i in int_to_mid]
            # ensure seed MIDs are included
            sg_mid_set = set(sg_mid_ents) | set(seed_mids)
            sg_mid_ents = list(sg_mid_set)

            # 3. subgraph tuples: [int, rel_str, int] → [MID, rel, MID]
            sg_tuples_out = []
            for tpl in r.get("subgraph", {}).get("tuples", []):
                if len(tpl) != 3:
                    continue
                h_int, rel, t_int = tpl
                h_mid = int_to_mid.get(h_int)
                t_mid = int_to_mid.get(t_int)
                if h_mid is None or t_mid is None:
                    dropped_tpl += 1
                    continue
                sg_tuples_out.append([h_mid, rel, t_mid])

            # 4. paths: convert list of [int, path_list] → [MID, path_list]
            paths_out = []
            for entry in r.get("paths", []):
                if len(entry) == 2:
                    ent_int, path = entry
                    ent_mid = int_to_mid.get(ent_int)
                    if ent_mid:
                        paths_out.append([ent_mid, path])

            out = {
                "id": r["id"],
                "question": r["question"],
                "entities": seed_mids,
                "answers": r.get("answers", []),
                "subgraph": {
                    "tuples": sg_tuples_out,
                    "entities": sg_mid_ents,
                },
                "paths": paths_out,
            }
            fout.write(json.dumps(out) + "\n")
            kept += 1

    print(f"  [{split}] kept={kept}  dropped(no-seed)={dropped_ent}  "
          f"dropped-tuples={dropped_tpl}")
    return kept


for split in ("train", "dev", "test"):
    convert_split(split)

# ── build extended entities.txt (base + any MIDs referenced in the data) ─────
print("\nBuilding extended entities.txt…")
base_entities_path = os.path.join(SRC, "entities.txt")
with open(base_entities_path) as f:
    base_ents = [l.strip() for l in f]
base_ent_set = set(base_ents)

extra_mids: set[str] = set()
for split in ("train", "dev", "test"):
    fpath = os.path.join(DST, f"{split}_simple.json")
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        for line in f:
            r = json.loads(line)
            for e in r["entities"]:
                if e not in base_ent_set:
                    extra_mids.add(e)
            for e in r["subgraph"]["entities"]:
                if e not in base_ent_set:
                    extra_mids.add(e)
            for ans in r["answers"]:
                mid = ans.get("kb_id", "")
                if mid and mid not in base_ent_set:
                    extra_mids.add(mid)

print(f"  Base entities: {len(base_ents):,}  Extra: {len(extra_mids):,}")
ext_path = os.path.join(DST, "entities.txt")
if os.path.lexists(ext_path):
    os.remove(ext_path)
with open(ext_path, "w") as f:
    for e in base_ents:
        f.write(e + "\n")
    for e in sorted(extra_mids):
        f.write(e + "\n")
print(f"  Written: {ext_path}  ({len(base_ents)+len(extra_mids):,} total)")

# ── symlink vocab/embeddings/relation dict (entities.txt already written above)
for fname, src_name in [
    ("vocab_new.txt",   "vocab.txt"),
    ("word_emb_300d.npy", "word_emb.npy"),
    ("relations.txt",   "relations.txt"),
]:
    target = os.path.join(DST, fname)
    source = os.path.join(SRC, src_name)
    if os.path.lexists(target):
        os.remove(target)
    os.symlink(source, target)
    print(f"  symlink {fname} -> {source}")

print("\nDone. CWQ NSM data written to:", DST)
