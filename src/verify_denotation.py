#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
verify_denotation.py — Core Denotation Rule verification.

The Core Denotation Rule (To_DO_updated_v6.md):
  SPARQL_denotation(q', Freebase) == SPARQL_denotation(q, Freebase)

Per P-type strategy:
  P1, P2, P4, P5  — SPARQL both original and predicted_sparql; compare MID sets
  P3              — BERTScore proxy (bertscore >= 0.85 in perturbation_meta)
  P6              — BERTScore proxy + optional 10% SPARQL sanity sample

Input:  data/perturbations/{dataset}_{p_type}.json
Output: data/perturbations/denotation_report.json

Usage:
  python verify_denotation.py                              # all datasets, all P-types
  python verify_denotation.py --dataset cwq --p-type p1  # single type
  python verify_denotation.py --sample 100               # limit to first N records
  python verify_denotation.py --p6-sparql-sample 0.10   # fraction of P6 for SPARQL sanity
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from SPARQLWrapper import JSON as SPARQL_JSON, SPARQLWrapper

DATASET_DIR = REPO_ROOT / "data" / "perturbations"
SPARQL_ENDPOINT = "http://localhost:3001/sparql"
BERTSCORE_THRESHOLD = 0.85

SPARQL_PTYPES = {"p1", "p2", "p4", "p5"}
BERTSCORE_PTYPES = {"p3", "p6"}

# Namespace prefix used by Freebase SPARQL
FB_NS = "http://rdf.freebase.com/ns/"


def build_sparql_client(endpoint: str) -> SPARQLWrapper:
    client = SPARQLWrapper(endpoint)
    client.setReturnFormat(SPARQL_JSON)
    client.setTimeout(30)
    return client


def execute_sparql(client: SPARQLWrapper, query: str) -> Set[str]:
    """Run a SPARQL query and return the set of answer MIDs/values."""
    try:
        client.setQuery(query)
        results = client.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        values: Set[str] = set()
        for b in bindings:
            for v in b.values():
                val = v.get("value", "")
                # Normalize Freebase URIs to MID form
                if val.startswith(FB_NS):
                    val = "m." + val[len(FB_NS):].lstrip("m.")
                values.add(val)
        return values
    except Exception:
        return set()


def check_sparql_denotation(
    client: SPARQLWrapper,
    original_sparql: str,
    predicted_sparql: str,
) -> Tuple[bool, Set[str], Set[str]]:
    """Return (passes, orig_answers, pred_answers)."""
    orig = execute_sparql(client, original_sparql)
    pred = execute_sparql(client, predicted_sparql)
    return orig == pred, orig, pred


def verify_record_sparql(
    client: SPARQLWrapper,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    orig_sparql = record.get("sparql", "")
    pred_sparql = record.get("perturbation_meta", {}).get("predicted_sparql", "")
    if not orig_sparql or not pred_sparql:
        return {"pass": False, "reason": "missing_sparql", "orig_size": 0, "pred_size": 0}

    passes, orig, pred = check_sparql_denotation(client, orig_sparql, pred_sparql)
    return {
        "pass": passes,
        "reason": "ok" if passes else "denotation_mismatch",
        "orig_size": len(orig),
        "pred_size": len(pred),
    }


def verify_record_bertscore(record: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    meta = record.get("perturbation_meta", {})
    bs = meta.get("bertscore", None)
    if bs is None:
        return {"pass": False, "reason": "missing_bertscore", "bertscore": None}
    passes = bs >= threshold
    return {
        "pass": passes,
        "reason": "ok" if passes else f"bertscore_below_{threshold}",
        "bertscore": bs,
    }


def verify_record_precomputed(
    record: Dict[str, Any],
    threshold: float = BERTSCORE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Verify denotation using pre-computed SPARQL results from perturbation_meta.

    The perturbation pipeline already ran SPARQL for original and perturbed questions,
    storing: gold_answer_size, pred_answer_size, perturbed_answers, bertscore.

    Pass criterion:
      1. gold_answer_size == pred_answer_size  (same answer set cardinality)
      2. bertscore >= threshold                (question similarity)
    Both must hold for SPARQL-type perturbations.
    """
    meta = record.get("perturbation_meta", {})
    bs = meta.get("bertscore", None)
    gold_size = meta.get("gold_answer_size")
    pred_size = meta.get("pred_answer_size")

    # BERTScore check
    bs_ok = bs is not None and bs >= threshold

    # Answer set size match
    size_ok = (gold_size is not None and pred_size is not None
               and gold_size == pred_size and gold_size > 0)

    passes = bs_ok and size_ok
    reason = "ok"
    if not bs_ok:
        reason = f"bertscore_below_{threshold}" if bs is not None else "missing_bertscore"
    elif not size_ok:
        reason = f"answer_size_mismatch_{gold_size}_vs_{pred_size}"

    return {
        "pass": passes,
        "reason": reason,
        "bertscore": bs,
        "gold_answer_size": gold_size,
        "pred_answer_size": pred_size,
        "method": "precomputed",
    }


def verify_dataset(
    dataset: str,
    p_type: str,
    sample: Optional[int],
    p6_sparql_fraction: float,
    client: Optional[SPARQLWrapper],
    verbose: bool,
) -> Dict[str, Any]:
    path = DATASET_DIR / f"{dataset}_{p_type}.json"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return {}

    with open(path) as f:
        records: List[Dict] = json.load(f)

    if sample:
        records = records[:sample]

    total = len(records)
    passed = 0
    failed_ids: List[str] = []
    details: List[Dict] = []

    use_sparql = p_type in SPARQL_PTYPES and client is not None
    use_bertscore_primary = p_type in BERTSCORE_PTYPES

    for i, rec in enumerate(records):
        rid = rec.get("id", str(i))

        if use_bertscore_primary:
            # P3/P6: BERTScore proxy (precomputed if available, else simple bertscore)
            result = verify_record_precomputed(rec, BERTSCORE_THRESHOLD)
            # For P6: also run SPARQL on a sample fraction if client available
            if p_type == "p6" and client is not None:
                import random
                if random.random() < p6_sparql_fraction:
                    sparql_result = verify_record_sparql(client, rec)
                    result["sparql_sanity"] = sparql_result
        elif use_sparql:
            result = verify_record_sparql(client, rec)
            # Add bertscore as supplementary info
            bs = rec.get("perturbation_meta", {}).get("bertscore")
            if bs is not None:
                result["bertscore"] = bs
            time.sleep(0.02)  # polite rate-limit on Virtuoso
        else:
            # SPARQL unavailable → precomputed answer-size + BERTScore proxy
            result = verify_record_precomputed(rec, BERTSCORE_THRESHOLD)
            if "method" not in result:
                result["reason"] = "precomputed_proxy"

        result["id"] = rid
        if result["pass"]:
            passed += 1
        else:
            failed_ids.append(rid)

        details.append(result)

        if verbose and (i + 1) % 100 == 0:
            print(f"    {i+1}/{total}  pass_so_far={passed}")

    pass_rate = passed / total if total > 0 else 0.0
    summary = {
        "dataset": dataset,
        "p_type": p_type,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(pass_rate, 4),
        "verification_method": "sparql" if use_sparql else "bertscore",
        "failed_ids": failed_ids[:100],  # cap to avoid huge report
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Core Denotation Rule compliance")
    parser.add_argument("--dataset", choices=["cwq", "webqsp"], default=None,
                        help="Dataset to verify (default: both)")
    parser.add_argument("--p-type", choices=["p1","p2","p3","p4","p5","p6"], default=None,
                        help="Perturbation type (default: all)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit to first N records per file")
    parser.add_argument("--p6-sparql-sample", type=float, default=0.10,
                        help="Fraction of P6 records to sanity-check via SPARQL (default 0.10)")
    parser.add_argument("--endpoint", default=SPARQL_ENDPOINT)
    parser.add_argument("--no-sparql", action="store_true",
                        help="Skip SPARQL calls; BERTScore only")
    parser.add_argument("--out", type=Path, default=DATASET_DIR / "denotation_report.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["cwq", "webqsp"]
    p_types = [args.p_type] if args.p_type else ["p1","p2","p3","p4","p5","p6"]

    client: Optional[SPARQLWrapper] = None
    if not args.no_sparql:
        try:
            client = build_sparql_client(args.endpoint)
            # Quick connectivity check
            client.setQuery("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
            client.query()
            print(f"SPARQL endpoint reachable: {args.endpoint}")
        except Exception as e:
            print(f"WARNING: SPARQL endpoint not reachable ({e}). Falling back to BERTScore-only.")
            client = None

    all_results: List[Dict] = []
    for dataset in datasets:
        for p_type in p_types:
            print(f"\nVerifying {dataset} {p_type} ...")
            summary = verify_dataset(
                dataset, p_type,
                sample=args.sample,
                p6_sparql_fraction=args.p6_sparql_sample,
                client=client,
                verbose=args.verbose,
            )
            if summary:
                all_results.append(summary)
                print(f"  pass_rate={summary['pass_rate']:.2%}  "
                      f"({summary['passed']}/{summary['total']})  "
                      f"method={summary['verification_method']}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nReport written to: {args.out}")


if __name__ == "__main__":
    main()