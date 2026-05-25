#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
graphrag_pert_experiment.py — graphrag-lexical-graph KGQA perturbation experiment.

Builds a LexicalGraphIndex from per-question KG subgraph triples, then queries
it with clean and perturbed questions to measure Hit@1 robustness.

Pipeline:
  1. Load subgraph triples (MID-based, new_exp/subgraphs/{ds}_{ptype}_nsm_gem/)
  2. Convert triples → natural language sentences via entities_names.json
  3. Build LexicalGraphIndex (one per question, isolated via tenant_id)
  4. Query with clean / perturbed question via LexicalGraphQueryEngine
  5. Evaluate Hit@1 (predicted entity name matches gold answer text)

Coverage note: only ~66.7% of CWQ p4 questions have the gold answer entity present
in the pre-built NSM-GEM subgraph. The graphrag Hit@1 is bounded by this coverage.

LLM backend: local Llama 3.1-8B-Instruct served via Ollama.
  ollama serve &
  ollama pull llama3.1:8b

Modes (run in order):
  sanity  — data + import check, NO LLM, NO Docker
  dry-run — N questions with real LLM + dummy stores
  index   — build Neo4j/pgvector indexes for all questions (no querying)
  clean   — query all questions with original questions (requires prior index run)
  pert    — query all questions with perturbed questions (requires prior index run)
  all     — index → clean → pert back-to-back

Docker quickstart for Neo4j + pgvector (needed for index/clean/pert/all):
  cd tmlr/Repo/graphrag-toolkit/examples/lexical-graph-local-dev/docker
  ./start-containers.sh

Usage:
  python new_exp/scripts/graphrag_pert_experiment.py --dataset cwq --ptype p4 --mode sanity
  python new_exp/scripts/graphrag_pert_experiment.py --dataset cwq --ptype p4 --mode dry-run --n 5
  python new_exp/scripts/graphrag_pert_experiment.py --dataset cwq --ptype p4 p7 --mode all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# graphrag_toolkit tries to init Bedrock on import — stub the region to prevent crash.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT     = REPO_ROOT
NEW_EXP       = REPO_ROOT / "new_exp"
SUBGRAPH_BASE = NEW_EXP / "subgraphs"
RESULTS_DIR   = NEW_EXP / "results" / "graphrag_rag"
ENTITIES_NAMES = REPO_ROOT / "tmlr/Repo/GNN-RAG/llm/entities_names.json"

NEO4J_URI    = "neo4j://neo4j:password@localhost:7687"
PGVECTOR_URI = "postgresql+psycopg2://admin:password@localhost:5432/vectordb"
DUMMY_URI    = "dummy://"

DEFAULT_LLM_URL   = "http://localhost:11434/v1"
DEFAULT_LLM_MODEL = "llama3.1:8b"   # llama3.3:8b doesn't exist; 3.3 is 70B-only

EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
SUBGRAPH_VARIANT = "nsm_gem"

# Save a checkpoint every N questions during index/clean/pert runs
CHECKPOINT_EVERY = 100


# ── Tenant-ID sanitization ────────────────────────────────────────────────────

def make_tenant_id(qid: str) -> str:
    """
    graphrag_toolkit requires tenant_id: 1-25 lowercase alphanumeric + periods,
    not starting or ending with a period.  QIDs like 'WebQTest-1000_abc...' fail
    that constraint, so we hash to a stable 16-char lowercase hex string.
    """
    return hashlib.md5(qid.encode()).hexdigest()[:16]


# ── Entity / relation name resolution ─────────────────────────────────────────

_ENTITIES_NAMES: Optional[Dict[str, str]] = None


def load_entities_names() -> Dict[str, str]:
    global _ENTITIES_NAMES
    if _ENTITIES_NAMES is None:
        with open(ENTITIES_NAMES) as f:
            _ENTITIES_NAMES = json.load(f)
    return _ENTITIES_NAMES


def mid_to_name(mid: str, names: Dict[str, str]) -> str:
    return names.get(mid, mid)


def rel_to_text(rel: str) -> str:
    """'sports.sports_team.championships' → 'sports sports team championships'"""
    return rel.replace(".", " ").replace("_", " ")


def triple_to_sentence(subj: str, rel: str, obj: str, names: Dict[str, str]) -> str:
    s = mid_to_name(subj, names)
    r = rel_to_text(rel)
    o = mid_to_name(obj, names)
    return f"{s} {r} {o}"


# ── Subgraph loading ──────────────────────────────────────────────────────────

def list_subgraph_files(dataset: str, ptype: str) -> List[Path]:
    sg_dir = SUBGRAPH_BASE / f"{dataset}_{ptype}_{SUBGRAPH_VARIANT}"
    if not sg_dir.exists():
        raise FileNotFoundError(f"Subgraph directory not found: {sg_dir}")
    files = sorted(sg_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files in {sg_dir}")
    return files


def _fix_mojibake(s: str) -> str:
    """Fix latin-1→utf-8 double-encoding (e.g. 'HagÃ¥tÃ±a' → 'Hagåtña')."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def load_subgraph(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        rec = json.load(f)
    for field in ("question", "original_question"):
        if field in rec and isinstance(rec[field], str):
            rec[field] = _fix_mojibake(rec[field])
    return rec


def subgraph_to_documents(record: Dict[str, Any], names: Dict[str, str]) -> List[Any]:
    """
    Convert a per-question subgraph to llama_index Documents.
    One Document per unique triple sentence + one seed-entity summary Document.
    """
    from llama_index.core import Document

    qid     = record["id"]
    triples = record.get("triples", [])
    seeds   = record.get("seed_mids", [])

    docs: List[Any] = []
    seen: set = set()

    for t in triples:
        if len(t) != 3:
            continue
        s, r, o = t
        sentence = triple_to_sentence(s, r, o, names)
        if sentence not in seen:
            seen.add(sentence)
            docs.append(Document(
                text=sentence,
                metadata={"qid": qid, "subj": s, "rel": r, "obj": o},
            ))

    if seeds:
        seed_names = [mid_to_name(m, names) for m in seeds]
        docs.append(Document(
            text="Topic entities: " + ", ".join(seed_names),
            metadata={"qid": qid, "type": "seed_summary"},
        ))

    return docs


# ── Answer resolution & evaluation ────────────────────────────────────────────

def answers_to_names(answers: List[Any], names: Dict[str, str]) -> List[str]:
    """
    Resolve gold answers to lower-cased text strings for Hit@1 comparison.
    Handles dicts with kb_id/text fields, and bare MID strings.
    Returns empty list if no gold answer can be resolved — that question scores 0.
    """
    result = []
    for a in answers:
        if isinstance(a, str):
            name = mid_to_name(a, names)
            if name != a:
                result.append(name.lower())
        elif isinstance(a, dict):
            kb_id = a.get("kb_id", "")
            text  = a.get("text", "")
            name = mid_to_name(kb_id, names) if kb_id else ""
            if name and name != kb_id:
                result.append(name.lower())
            elif text and not text.startswith(("m.", "g.")):
                result.append(text.lower())
    return result


def hit_at_1(predicted: List[str], gold_names: List[str]) -> float:
    """1.0 if any gold answer substring appears in the joint prediction string."""
    if not predicted or not gold_names:
        return 0.0
    pred_joined = " ".join(p.lower() for p in predicted)
    return float(any(g in pred_joined for g in gold_names if g))


def compute_coverage(files: List[Path]) -> Dict[str, int]:
    """
    Count how many questions have their gold answer entity present in the subgraph.
    Called at the start of run_experiment to warn about the coverage ceiling.
    """
    names = load_entities_names()
    total = gold_in_subgraph = gold_resolvable = 0
    for fpath in files:
        rec = load_subgraph(fpath)
        gold_mids = [a.get("kb_id", "") if isinstance(a, dict) else a
                     for a in rec.get("answers", [])]
        gold_mids = [m for m in gold_mids if m]
        triple_mids = set()
        for t in rec.get("triples", []):
            if len(t) == 3:
                triple_mids.add(t[0]); triple_mids.add(t[2])
        if any(m in triple_mids or m in set(rec.get("entities", []))
               for m in gold_mids):
            gold_in_subgraph += 1
        if any(mid_to_name(m, names) != m for m in gold_mids):
            gold_resolvable += 1
        total += 1
    return {"total": total, "in_subgraph": gold_in_subgraph, "resolvable": gold_resolvable}


# ── LLM / embedding construction ──────────────────────────────────────────────

def make_llm(llm_url: str, llm_model: str):
    """OpenAI-compatible local LLM (Ollama or vLLM)."""
    from llama_index.llms.openai_like import OpenAILike
    return OpenAILike(
        model=llm_model,
        api_base=llm_url,
        api_key="ollama",
        max_tokens=256,
        timeout=60.0,
        is_chat_model=True,
    )


def make_embed():
    """Local sentence-transformer — no GPU or API key needed."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(model_name=EMBED_MODEL)


def ping_llm(llm) -> bool:
    """Return True if LLM responds, False otherwise."""
    try:
        resp = llm.complete("Reply with one word: ready").text.strip()
        print(f"[OK] LLM ping: '{resp[:30]}'")
        return True
    except Exception as e:
        print(f"[FAIL] LLM unreachable: {e}")
        return False


# ── Graph / vector store construction ─────────────────────────────────────────

def make_stores(use_dummy: bool):
    """Return (graph_store, vector_store)."""
    from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory, VectorStoreFactory

    graph_uri  = DUMMY_URI if use_dummy else NEO4J_URI
    vector_uri = DUMMY_URI if use_dummy else PGVECTOR_URI

    graph_store  = GraphStoreFactory.for_graph_store(graph_uri)
    vector_store = VectorStoreFactory.for_vector_store(
        vector_uri, index_names=["chunk", "statement", "topic"]
    )
    return graph_store, vector_store


# ── Per-question index and query ───────────────────────────────────────────────

def index_one(docs: List[Any], graph_store, vector_store, tenant_id: str) -> None:
    """
    Build the lexical graph for one question's subgraph into Neo4j/pgvector.
    Separated from querying so clean and pert modes can reuse the same index.
    """
    from graphrag_toolkit.lexical_graph import LexicalGraphIndex
    idx = LexicalGraphIndex(
        graph_store=graph_store,
        vector_store=vector_store,
        tenant_id=tenant_id,
    )
    idx.extract_and_build(docs)


def query_one(
    question: str,
    docs: List[Any],
    graph_store,
    vector_store,
    llm,
    embed,
    use_dummy: bool,
    tenant_id: str = "default",
    verbose: bool = False,
) -> Tuple[List[str], str]:
    """
    Query the lexical graph for one question.
    In dummy mode: LLM is called directly on the first 150 named-entity docs.
    In real mode: LexicalGraphQueryEngine retrieves from Neo4j/pgvector.

    Returns (predicted_answers, raw_response_text).
    """
    from graphrag_toolkit.lexical_graph import LexicalGraphIndex, LexicalGraphQueryEngine
    from llama_index.core import Settings

    Settings.llm         = llm
    Settings.embed_model = embed

    if use_dummy:
        # Dummy stores don't persist — skip extract_and_build.
        # Sort docs: named entities first (no raw MID prefix), then CVT nodes.
        # Send up to 150 docs; Llama 3.1 context is 131K so this is cheap.
        named_docs = [d for d in docs if not d.text.startswith(("m.", "g."))]
        other_docs  = [d for d in docs if d.text.startswith(("m.", "g."))]
        context_text = "\n".join(d.text for d in (named_docs + other_docs)[:150])
        prompt = (
            f"You are a knowledge graph QA assistant.\n"
            f"Answer with entity names only — list multiple if needed, comma-separated.\n"
            f"Facts:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        )
        response_text = llm.complete(prompt).text.strip()
        if verbose:
            print(f"    [dummy-LLM] {response_text[:100]}")
        return [response_text], response_text

    # Real mode: query the pre-built lexical graph via tenant-scoped engine.
    # index_one() must have been called first for this tenant_id.
    engine = LexicalGraphQueryEngine(
        graph_store=graph_store,
        vector_store=vector_store,
        tenant_id=tenant_id,
        llm=llm,
    )
    response_text = str(engine.query(question)).strip()
    if verbose:
        print(f"    [graphrag] {response_text[:120]}")
    return [response_text], response_text


# ── Stage 0: Sanity check ─────────────────────────────────────────────────────

def run_sanity(dataset: str, ptype: str, n: int = 5) -> bool:
    ok = True
    print(f"\n{'='*65}")
    print(f"SANITY CHECK (data only): {dataset} {ptype}")
    print(f"{'='*65}")

    try:
        names = load_entities_names()
        print(f"[OK] entities_names.json — {len(names):,} entries")
    except Exception as e:
        print(f"[FAIL] entities_names.json: {e}")
        return False

    try:
        files = list_subgraph_files(dataset, ptype)
        print(f"[OK] subgraph dir — {len(files):,} files  ({dataset}_{ptype}_{SUBGRAPH_VARIANT}/)")
    except FileNotFoundError as e:
        print(f"[FAIL] {e}")
        return False

    # Coverage ceiling — quick sample
    print(f"\nCoverage check (full {len(files)} files — may take ~30s):")
    cov = compute_coverage(files)
    pct_sg  = 100 * cov["in_subgraph"] / cov["total"]
    pct_res = 100 * cov["resolvable"]  / cov["total"]
    print(f"  Gold entity in subgraph    : {cov['in_subgraph']}/{cov['total']}  ({pct_sg:.1f}%)")
    print(f"  Gold name resolvable       : {cov['resolvable']}/{cov['total']}  ({pct_res:.1f}%)")
    print(f"  → Max achievable Hit@1 ≈ {pct_sg:.1f}%  (questions with gold in subgraph)")

    gold_empty = 0
    n_total_triples = 0
    print(f"\nSpot-checking first {n} files:")
    for i, fpath in enumerate(files[:n]):
        rec = load_subgraph(fpath)
        docs = subgraph_to_documents(rec, names)
        gold = answers_to_names(rec.get("answers", []), names)
        n_total_triples += len(rec.get("triples", []))
        if not gold:
            gold_empty += 1
        sample_doc = docs[0].text if docs else "—"
        print(f"\n  [{i+1}] {rec['id'][:50]}")
        print(f"       question : {rec.get('question','')[:70]}")
        print(f"       triples  : {len(rec.get('triples',[]))}  docs: {len(docs)}")
        print(f"       sample   : {sample_doc[:90]}")
        print(f"       gold     : {gold[:3]}  {'⚠ unresolvable' if not gold else ''}")

    print(f"\n{'─'*65}")
    print(f"  avg triples/question : {n_total_triples / min(n, len(files)):.0f}")
    print(f"  gold empty (sample)  : {gold_empty}/{min(n, len(files))}")
    if gold_empty > 0:
        ok = False

    for label, import_fn in [
        ("llama_index.core.Document",            lambda: __import__("llama_index.core", fromlist=["Document"])),
        ("graphrag_toolkit imports",              lambda: __import__("graphrag_toolkit.lexical_graph", fromlist=["LexicalGraphIndex"])),
        ("llama_index HuggingFaceEmbedding",      lambda: __import__("llama_index.embeddings.huggingface", fromlist=["HuggingFaceEmbedding"])),
        ("llama_index OpenAILike",                lambda: __import__("llama_index.llms.openai_like", fromlist=["OpenAILike"])),
    ]:
        try:
            import_fn()
            print(f"[OK] {label}")
        except Exception as e:
            print(f"[FAIL] {label}: {e}")
            ok = False

    try:
        gs, vs = make_stores(use_dummy=True)
        print(f"[OK] dummy graph store: {type(gs).__name__}  vector store: {type(vs).__name__}")
    except Exception as e:
        print(f"[FAIL] dummy stores: {e}")
        ok = False

    print(f"\n{'='*65}")
    print(f"SANITY RESULT: {'PASS' if ok else 'FAIL (see above)'}")
    print(f"{'='*65}")
    return ok


# ── Stage 1: Dry-run (LLM + dummy stores, N questions) ────────────────────────

def run_dry_run(dataset: str, ptype: str, n: int,
                llm_url: str, llm_model: str) -> None:
    print(f"\n{'='*65}")
    print(f"DRY-RUN: {dataset} {ptype} — {n} questions  (LLM + dummy stores)")
    print(f"  LLM  : {llm_model}  @ {llm_url}")
    print(f"{'='*65}")

    names = load_entities_names()
    files = list_subgraph_files(dataset, ptype)
    gs, vs = make_stores(use_dummy=True)

    try:
        llm = make_llm(llm_url, llm_model)
        if not ping_llm(llm):
            return
    except Exception as e:
        print(f"[FAIL] LLM init: {e}")
        return

    try:
        embed = make_embed()
        dim   = len(embed.get_text_embedding("hello world"))
        print(f"[OK] Embedding model loaded, dim={dim}")
    except Exception as e:
        print(f"[FAIL] Embedding model: {e}")
        return

    hits_pert, hits_orig, total, errors = 0.0, 0.0, 0, 0
    for i, fpath in enumerate(files[:n]):
        rec    = load_subgraph(fpath)
        qid    = rec["id"]
        q_pert = rec.get("question", rec.get("original_question", ""))
        q_orig = rec.get("original_question", q_pert)
        docs   = subgraph_to_documents(rec, names)
        gold   = answers_to_names(rec.get("answers", []), names)
        tid    = make_tenant_id(qid)

        print(f"\n[{i+1}/{n}] {qid[:50]}")
        print(f"  Q clean: {q_orig[:75]}")
        print(f"  Q pert : {q_pert[:75]}")
        print(f"  gold   : {gold[:3]}")
        print(f"  docs   : {len(docs)}")

        try:
            p_pert, _ = query_one(q_pert, docs, gs, vs, llm, embed,
                                  use_dummy=True, tenant_id=tid, verbose=True)
            p_orig, _ = query_one(q_orig, docs, gs, vs, llm, embed,
                                  use_dummy=True, tenant_id=tid, verbose=False)
            hp = hit_at_1(p_pert, gold)
            ho = hit_at_1(p_orig, gold)
            hits_pert += hp; hits_orig += ho; total += 1
            print(f"  hit@1  clean={int(ho)}  pert={int(hp)}")
            print(f"  pred_clean='{p_orig[0][:60]}'  pred_pert='{p_pert[0][:60]}'")
        except Exception as e:
            errors += 1
            print(f"  [ERROR] {type(e).__name__}: {e}")

    print(f"\n{'='*65}")
    if total > 0:
        print(f"DRY-RUN SUMMARY  ({n} questions):")
        print(f"  Hit@1 clean : {hits_orig}/{total} = {hits_orig/total*100:.1f}%")
        print(f"  Hit@1 pert  : {hits_pert}/{total} = {hits_pert/total*100:.1f}%")
        print(f"  NOTE: dummy mode — LLM uses prior knowledge, NOT graphrag retrieval.")
        print(f"  Run --mode all with Neo4j+pgvector for real retrieval-augmented scores.")
    else:
        print(f"DRY-RUN SUMMARY: 0 questions succeeded ({errors} errors)")
    print(f"{'='*65}")


# ── Stage 2: Index all questions into Neo4j / pgvector ────────────────────────

def run_index(dataset: str, ptype: str, n: Optional[int],
              force: bool, llm_url: str, llm_model: str) -> None:
    """
    Pre-index all questions into Neo4j/pgvector (one tenant_id per question).
    Must be run before --mode clean or --mode pert.
    Skips questions whose tenant already has data, unless --force.
    """
    print(f"\n{'='*65}")
    print(f"INDEXING: {dataset} {ptype}")
    print(f"  graph_store : {NEO4J_URI}")
    print(f"  vector_store: {PGVECTOR_URI}")
    print(f"{'='*65}")

    names = load_entities_names()
    files = list_subgraph_files(dataset, ptype)
    if n:
        files = files[:n]

    gs, vs = make_stores(use_dummy=False)
    embed  = make_embed()

    from llama_index.core import Settings
    Settings.embed_model = embed

    done_path = RESULTS_DIR / f"{dataset}_{ptype}" / "indexed_qids.json"
    done_path.parent.mkdir(parents=True, exist_ok=True)
    indexed: set = set()
    if done_path.exists() and not force:
        indexed = set(json.load(open(done_path)))
        print(f"  Resuming: {len(indexed)} already indexed")

    t0 = time.time()
    new_count = 0
    for i, fpath in enumerate(files):
        rec = load_subgraph(fpath)
        qid = rec["id"]
        tid = make_tenant_id(qid)

        if tid in indexed and not force:
            continue

        docs = subgraph_to_documents(rec, names)
        try:
            index_one(docs, gs, vs, tid)
            indexed.add(tid)
            new_count += 1
        except Exception as e:
            print(f"  [WARN] {qid[:40]}: {e}")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(done_path, "w") as f:
                json.dump(list(indexed), f)
            elapsed = time.time() - t0
            rate    = new_count / elapsed if elapsed > 0 else 0
            eta     = (len(files) - i - 1) / rate if rate > 0 else float("inf")
            print(f"  [{i+1}/{len(files)}] indexed {new_count} new  "
                  f"({elapsed:.0f}s elapsed, ETA {eta/60:.0f}min)")

    with open(done_path, "w") as f:
        json.dump(list(indexed), f)
    print(f"\n  Indexing complete: {len(indexed)} total tenants in store")
    print(f"  Manifest → {done_path}")


# ── Stage 3: Query questions (clean or pert) ──────────────────────────────────

def run_query(dataset: str, ptype: str, mode: str,
              n: Optional[int], force: bool,
              llm_url: str, llm_model: str) -> Dict[str, Any]:
    """
    Query mode — separate from indexing.
    mode='clean': query with original questions.
    mode='pert' : query with perturbed questions.
    Assumes index has already been built (run --mode index first).
    """
    assert mode in ("clean", "pert")
    print(f"\n{'='*65}")
    print(f"QUERY {mode.upper()}: {dataset} {ptype}")
    print(f"  graph_store : {NEO4J_URI}")
    print(f"  vector_store: {PGVECTOR_URI}")
    print(f"  LLM         : {llm_model} @ {llm_url}")
    print(f"{'='*65}")

    result_dir = RESULTS_DIR / f"{dataset}_{ptype}"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path      = result_dir / f"{mode}_results.json"
    partial_path  = result_dir / f"{mode}_partial.json"

    if out_path.exists() and not force:
        print(f"  Results already exist — loading: {out_path}")
        return json.load(open(out_path))

    names = load_entities_names()
    files = list_subgraph_files(dataset, ptype)
    if n:
        files = files[:n]

    # Coverage warning
    print(f"  Computing coverage ceiling…")
    cov = compute_coverage(files)
    print(f"  Gold in subgraph: {cov['in_subgraph']}/{cov['total']} ({100*cov['in_subgraph']/cov['total']:.1f}%)")
    print(f"  Max achievable Hit@1 ≈ {100*cov['in_subgraph']/cov['total']:.1f}%")

    gs, vs = make_stores(use_dummy=False)
    llm    = make_llm(llm_url, llm_model)
    embed  = make_embed()

    if not ping_llm(llm):
        print("[ABORT] LLM not reachable.")
        sys.exit(1)

    from llama_index.core import Settings
    Settings.llm         = llm
    Settings.embed_model = embed

    # Resume from partial if exists
    per_question: List[Dict] = []
    done_ids: set = set()
    if partial_path.exists() and not force:
        per_question = json.load(open(partial_path))
        done_ids     = {r["id"] for r in per_question}
        print(f"  Resuming from partial: {len(done_ids)} done")

    hits, total = sum(r["hit1"] for r in per_question), len(per_question)
    t0 = time.time()

    for i, fpath in enumerate(files):
        rec  = load_subgraph(fpath)
        qid  = rec["id"]
        if qid in done_ids:
            continue

        q_pert = rec.get("question", "")
        q_orig = rec.get("original_question", q_pert)
        question = q_orig if mode == "clean" else q_pert

        docs = subgraph_to_documents(rec, names)
        gold = answers_to_names(rec.get("answers", []), names)
        tid  = make_tenant_id(qid)

        try:
            preds, raw = query_one(
                question=question,
                docs=docs,
                graph_store=gs,
                vector_store=vs,
                llm=llm,
                embed=embed,
                use_dummy=False,
                tenant_id=tid,
            )
            h = hit_at_1(preds, gold)
        except Exception as e:
            print(f"  [WARN] {qid[:40]}: {e}")
            preds, raw, h = [], str(e), 0.0

        hits  += h
        total += 1
        per_question.append({
            "id": qid, "question": question,
            "gold": gold, "predicted": preds, "hit1": h,
        })

        if total % CHECKPOINT_EVERY == 0:
            with open(partial_path, "w") as f:
                json.dump(per_question, f)
            print(f"  [{i+1}/{len(files)}]  Hit@1={hits/total*100:.1f}%  "
                  f"({time.time()-t0:.0f}s)")

    hit1_pct = hits / total * 100 if total else 0.0
    summary  = {
        "dataset": dataset, "ptype": ptype, "mode": mode,
        "n": total, "hit1": hit1_pct,
        "coverage": cov,
        "per_question": per_question,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    if partial_path.exists():
        partial_path.unlink()
    print(f"\n  Hit@1 = {hit1_pct:.2f}%  (n={total})")
    print(f"  Saved → {out_path}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def print_table(results: List[Dict]) -> None:
    if not results:
        return
    print(f"\n{'='*65}")
    print("RESULTS SUMMARY")
    print(f"  {'dataset':<8} {'ptype':<5} {'mode':<6} {'n':>5} {'Hit@1':>8}  coverage")
    print(f"  {'-'*55}")
    for r in results:
        cov = r.get("coverage", {})
        pct_cov = (100 * cov.get("in_subgraph", 0) / cov.get("total", 1)
                   if cov.get("total") else 0)
        print(f"  {r['dataset']:<8} {r['ptype']:<5} {r['mode']:<6} "
              f"{r['n']:>5} {r['hit1']:>7.2f}%  "
              f"(ceiling {pct_cov:.0f}%)")
    print(f"{'='*65}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset",   nargs="+", choices=["cwq", "webqsp"], default=["cwq"])
    p.add_argument("--ptype",     nargs="+", choices=["p1", "p4", "p7"], default=["p4"])
    p.add_argument("--mode",      choices=["sanity", "dry-run", "index", "clean", "pert", "all"],
                   default="sanity")
    p.add_argument("--n",         type=int, default=None,
                   help="Limit to first N questions (dry-run default: 5)")
    p.add_argument("--force",     action="store_true",
                   help="Overwrite existing results / re-index")
    p.add_argument("--llm-url",   default=DEFAULT_LLM_URL)
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    args = p.parse_args()

    results: List[Dict] = []
    for dataset in args.dataset:
        for ptype in args.ptype:
            if args.mode == "sanity":
                run_sanity(dataset, ptype, n=args.n or 5)

            elif args.mode == "dry-run":
                run_dry_run(dataset, ptype, n=args.n or 5,
                            llm_url=args.llm_url, llm_model=args.llm_model)

            elif args.mode == "index":
                run_index(dataset, ptype, args.n, args.force,
                          args.llm_url, args.llm_model)

            elif args.mode == "all":
                # Index once, then clean + pert reuse the same indexed tenants
                run_index(dataset, ptype, args.n, args.force,
                          args.llm_url, args.llm_model)
                for m in ("clean", "pert"):
                    r = run_query(dataset, ptype, m, args.n, args.force,
                                  args.llm_url, args.llm_model)
                    results.append(r)

            else:  # clean or pert
                r = run_query(dataset, ptype, args.mode, args.n, args.force,
                              args.llm_url, args.llm_model)
                results.append(r)

    print_table(results)


if __name__ == "__main__":
    main()