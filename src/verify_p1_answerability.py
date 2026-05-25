#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
verify_p1_answerability.py — Task 2 from To_do.md.

For each P1 question, verify whether the substituted entity has a valid answer
path in Freebase for the relevant relation.

Method:
  1. Extract the substituted entity MID from the perturbed question's predicted_sparql
     (the new ns:<m.xxx> that replaced the original entity in the SPARQL).
  2. Extract the answer MID from perturbation_meta.perturbed_answers.
  3. Run:
       ASK { ns:<substituted_mid> <relation_from_SPARQL> <answer_mid> }
     against Virtuoso at http://localhost:8890/sparql.
  4. If ASK returns true → question is answerable by construction.
  5. Report: % of P1 questions that are answerable, % unanswerable (data artifact).

Why this matters:
  If a large fraction of P1 questions are unanswerable (the substituted entity
  has NO path to the original answer in Freebase), then near-zero EM on P1 is a
  DATA ARTIFACT, not a model failure finding.  Only the answerable subset is valid
  for evaluating architectural robustness.

Usage:
  python verify_p1_answerability.py --dataset cwq --n-sample 200
  python verify_p1_answerability.py --dataset webqsp --full
  python verify_p1_answerability.py --all --n-sample 200
  python verify_p1_answerability.py --dataset cwq --full --out-file tmlr/results/p1_answerability_cwq.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen, Request

REPO_ROOT = REPO_ROOT
TMLR    = REPO_ROOT / "tmlr"
RESULTS = TMLR / "results"

DATASET_DIR = TMLR / "pert" / "dataset"
SPARQL_URL  = "http://localhost:3001/sparql"  # port 3001 has Freebase data (port 8890 is empty)
FB_PREFIX   = "http://rdf.freebase.com/ns/"


# ── SPARQL helpers ────────────────────────────────────────────────────────────

def sparql_ask(query: str, timeout: int = 10) -> Optional[bool]:
    """Run a SPARQL ASK query; return True/False or None on error."""
    params = urlencode({
        "query":  query,
        "format": "application/sparql-results+json",
    })
    url = f"{SPARQL_URL}?{params}"
    try:
        req = Request(url, headers={"Accept": "application/sparql-results+json"})
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return bool(data.get("boolean", False))
    except Exception:
        return None


def sparql_select_mids(query: str, var: str = "x", timeout: int = 15) -> Set[str]:
    """Run a SPARQL SELECT; return set of MIDs in ?var column."""
    params = urlencode({
        "query":  query,
        "format": "application/sparql-results+json",
    })
    url = f"{SPARQL_URL}?{params}"
    mids: Set[str] = set()
    try:
        req = Request(url, headers={"Accept": "application/sparql-results+json"})
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        for row in data.get("results", {}).get("bindings", []):
            val = row.get(var, {}).get("value", "")
            if val.startswith(FB_PREFIX):
                mid = val[len(FB_PREFIX):]
                if mid.startswith("m.") or mid.startswith("g."):
                    mids.add(mid)
    except Exception:
        pass
    return mids


# ── Extract substituted entity MID from predicted_sparql ─────────────────────

_NS_MID_RE = re.compile(r"ns:(m\.[^\s\.\)>]+|g\.[^\s\.\)>]+)")
_FAKE_MID_RE = re.compile(r"0{5,}|^m\.0{1,2}\d{5,}$")  # malformed MIDs from perturbation


def extract_sparql_mids(sparql_str: str) -> Set[str]:
    """Extract all ns:m.xxx MIDs from a SPARQL string."""
    mids = {m.rstrip(")>.,;") for m in _NS_MID_RE.findall(sparql_str)}
    # Filter obviously fake MIDs (too long, all-zeros-suffix)
    return {m for m in mids if len(m) <= 18 and "000000" not in m}


def get_substituted_mid(orig_sparql: str, pert_sparql: str) -> Optional[str]:
    """Return the MID in pert_sparql that is NOT in orig_sparql (the substituted one)."""
    orig_mids = extract_sparql_mids(orig_sparql)
    pert_mids = extract_sparql_mids(pert_sparql)
    new_mids  = pert_mids - orig_mids
    if not new_mids:
        return None
    return min(new_mids, key=len)


def classify_p1_type(orig_sparql: str, pert_sparql: str) -> str:
    """Classify whether P1 changed the entity in SPARQL (type_a) or just text (type_b)."""
    if not pert_sparql:
        return "no_predicted_sparql"
    subst = get_substituted_mid(orig_sparql, pert_sparql)
    return "type_a_entity_subst" if subst else "type_b_text_only"


def get_sparql_relations(sparql_str: str) -> List[str]:
    """Extract relation predicates from SPARQL (ns:xxx.yyy.zzz patterns)."""
    rel_re = re.compile(r"ns:([a-z_][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)+)")
    return list({m for m in rel_re.findall(sparql_str)
                 if not m.startswith("m.") and not m.startswith("g.")})


# ── Per-question answerability check ─────────────────────────────────────────

def check_answerability(rec: Dict) -> Dict:
    """
    Returns dict with:
      subst_mid: substituted entity MID (from predicted_sparql)
      answer_mids: expected answer MIDs
      answerable: True/False/None
      method: "sparql_ask" | "sparql_select" | "no_subst_mid" | "error"
    """
    meta      = rec.get("perturbation_meta", {})
    orig_sp   = rec.get("sparql", "")
    pert_sp   = meta.get("predicted_sparql", "")
    pert_ans  = meta.get("perturbed_answers", [])

    result = {
        "qid":        rec["id"],
        "orig_q":     rec.get("original_question", "")[:60],
        "pert_q":     rec.get("perturbed_question", "")[:60],
        "subst_mid":  None,
        "answer_mids": [],
        "answerable": None,
        "method":     None,
    }

    if not pert_sp:
        result["method"] = "no_predicted_sparql"
        return result

    subst_mid = get_substituted_mid(orig_sp, pert_sp)
    result["subst_mid"] = subst_mid
    result["p1_type"]   = classify_p1_type(orig_sp, pert_sp)

    if not subst_mid:
        result["method"] = "type_b_text_only" if pert_sp else "no_predicted_sparql"
        # For type B (text-only), the SPARQL entities unchanged.
        # Run the ORIGINAL SPARQL against Freebase to see if it still returns results
        if pert_sp and orig_sp:
            orig_result_mids = sparql_select_mids(orig_sp)
            if orig_result_mids:
                result["answerable"] = True   # Original SPARQL still works
                result["method"] = "type_b_original_sparql_ok"
            else:
                result["answerable"] = None  # Unknown — Virtuoso may not support full query
                result["method"] = "type_b_original_sparql_empty"
        return result

    # Get expected answer MIDs from perturbed_answers
    ans_mids: Set[str] = set()
    for a in (pert_ans or []):
        if isinstance(a, dict):
            kb_id = a.get("kb_id") or a.get("entity_id", "")
            if kb_id and (kb_id.startswith("m.") or kb_id.startswith("g.")):
                ans_mids.add(kb_id)
            # Also try text if it looks like a MID
            txt = a.get("text", "")
            if txt and (txt.startswith("m.") or txt.startswith("g.")):
                ans_mids.add(txt)
        elif isinstance(a, str) and (a.startswith("m.") or a.startswith("g.")):
            ans_mids.add(a)

    result["answer_mids"] = list(ans_mids)

    # Strategy 1: Run predicted_sparql directly with substituted entity
    # Replace the original entity MID in the SPARQL with the substituted one
    # (already done in predicted_sparql), then execute and check if any answer
    # The simplest check: run the predicted SPARQL and see if it returns results
    if pert_sp.strip().upper().startswith("PREFIX"):
        result_mids = sparql_select_mids(pert_sp)
        if result_mids:
            result["answerable"] = True
            result["method"] = "sparql_select_got_results"
            result["found_mids"] = list(result_mids)[:5]
            return result
        else:
            result["answerable"] = False
            result["method"] = "sparql_select_empty"
            return result

    # Strategy 2: ASK query with each known relation
    if ans_mids:
        relations = get_sparql_relations(orig_sp)
        for rel in relations[:5]:
            for ans_mid in list(ans_mids)[:3]:
                q = (f"PREFIX ns: <http://rdf.freebase.com/ns/> "
                     f"ASK {{ ns:{subst_mid} ns:{rel} ns:{ans_mid} }}")
                ok = sparql_ask(q)
                if ok is True:
                    result["answerable"] = True
                    result["method"] = f"ask_direct ({rel})"
                    return result
        result["answerable"] = False
        result["method"] = "ask_all_false"
    else:
        # No answer MIDs — run the SPARQL and check if it's non-empty
        result["method"] = "no_answer_mids"

    return result


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_dataset(dataset: str, p_type: str, n_sample: Optional[int]) -> Dict:
    ds_path = DATASET_DIR / f"{dataset}_{p_type}.json"
    if not ds_path.exists():
        print(f"  [SKIP] {ds_path} not found")
        return {}

    records = json.load(open(ds_path))
    if n_sample and n_sample < len(records):
        records = random.Random(42).sample(records, n_sample)

    print(f"\n  {dataset} {p_type}: n={len(records)}")

    results_per_q = []
    n_answerable  = 0
    n_unanswerable = 0
    n_no_subst    = 0
    n_error       = 0

    t0 = time.time()
    for i, rec in enumerate(records):
        res = check_answerability(rec)
        results_per_q.append(res)

        if res["answerable"] is True:
            n_answerable += 1
        elif res["answerable"] is False:
            n_unanswerable += 1
        elif res["method"] in ("no_subst_mid", "no_predicted_sparql"):
            n_no_subst += 1
        else:
            n_error += 1

        if (i + 1) % max(1, len(records) // 10) == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(records)}] ans={n_answerable} "
                  f"unans={n_unanswerable} no_subst={n_no_subst}  "
                  f"({elapsed:.0f}s)", flush=True)

    n_checked = n_answerable + n_unanswerable
    return {
        "dataset":        dataset,
        "p_type":         p_type,
        "n_total":        len(records),
        "n_answerable":   n_answerable,
        "n_unanswerable": n_unanswerable,
        "n_no_subst_mid": n_no_subst,
        "n_error":        n_error,
        "n_checked":      n_checked,
        "answerable_pct": round(100 * n_answerable / max(1, n_checked), 2),
        "unanswerable_pct": round(100 * n_unanswerable / max(1, n_checked), 2),
        "coverage_pct":   round(100 * n_checked / max(1, len(records)), 2),
        "samples": results_per_q,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["cwq", "webqsp"])
    ap.add_argument("--p-type",  default="p1",
                    help="Perturbation type (default: p1). Can specify p2..p6 too.")
    ap.add_argument("--all",     action="store_true",
                    help="Run all datasets × p-types")
    ap.add_argument("--n-sample", type=int, default=200,
                    help="Questions per combo (default: 200)")
    ap.add_argument("--full",    action="store_true",
                    help="Run on full dataset")
    ap.add_argument("--out-file", default=None)
    args = ap.parse_args()

    n_sample = None if args.full else args.n_sample

    if args.all:
        combos = [(ds, pt)
                  for ds in ["cwq", "webqsp"]
                  for pt in ["p1", "p2", "p3", "p4", "p5", "p6"]]
    elif args.dataset:
        combos = [(args.dataset, args.p_type)]
    else:
        ap.error("Specify --dataset or --all")

    # Test SPARQL connectivity first
    test_q = "ASK { <http://rdf.freebase.com/ns/m.05kh3> ?p ?o }"
    ok = sparql_ask(test_q)
    if ok is None:
        print(f"[WARN] Virtuoso not reachable at {SPARQL_URL} — results will be empty.")
    else:
        print(f"Virtuoso OK (test ASK returned {ok})")

    all_results: Dict = {}
    for dataset, p_type in combos:
        key = f"{dataset}_{p_type}"
        print(f"\n── {key} ──")
        res = evaluate_dataset(dataset, p_type, n_sample)
        if res:
            all_results[key] = res
            r = res
            print(f"\n  ✓ {key}: answerable={r['n_answerable']}/{r['n_checked']} "
                  f"({r['answerable_pct']:.1f}%)  "
                  f"unanswerable={r['unanswerable_pct']:.1f}%  "
                  f"no_subst={r['n_no_subst_mid']}")

    # Summary
    print(f"\n{'='*70}")
    print("P1 Answerability Summary")
    print(f"{'='*70}")
    print(f"  {'Key':<18} {'n':>5}  {'Chk':>5}  {'Ans%':>7}  {'Unans%':>8}  {'NoSubst':>8}")
    print("  " + "-"*65)
    for ds in ["cwq", "webqsp"]:
        for pt in ["p1", "p2", "p3", "p4", "p5", "p6"]:
            key = f"{ds}_{pt}"
            if key not in all_results:
                continue
            r = all_results[key]
            print(f"  {key:<18} {r['n_total']:>5}  {r['n_checked']:>5}  "
                  f"{r['answerable_pct']:>6.1f}%  {r['unanswerable_pct']:>7.1f}%  "
                  f"{r['n_no_subst_mid']:>8}")

    out_path = Path(args.out_file) if args.out_file else (RESULTS / "p1_answerability.json")
    # Strip per-question samples for summary JSON (keep separate)
    out_summary = {k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                   for k, v in all_results.items()}
    out_path.write_text(json.dumps(out_summary, indent=2))
    print(f"\nSaved → {out_path}")

    # Also save full per-question results
    full_path = out_path.parent / (out_path.stem + "_full.json")
    full_path.write_text(json.dumps(all_results, indent=2))
    print(f"Full per-question results → {full_path}")


if __name__ == "__main__":
    main()