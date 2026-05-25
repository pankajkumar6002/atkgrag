#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
exp_cross_attack_transfer.py
============================
Experiment #3: Cross-attack subgraph transfer.

Mixes perturbed questions from one attack type with subgraphs from another,
causally decomposing whether a performance drop stems from:
  (a) subgraph retrieval failure (entity-linking / PPR stage), or
  (b) GNN decoder failure (instruction-sensitivity stage).

Three transfer cells:

  [3A] CR-Q + RS-SG   → oracle CR upper bound
       Question: p4 (compositional restructuring)
       Subgraph: p7 ELQ (relation synonym — near-clean entity topology)
       Expected: EM >> 0.68% (pipeline CR); approaches fast-path decoder EM (~50%)
       Claim tested: CR collapse is driven by subgraph failure, not decoder.

  [3B] RS-Q + CR-SG   → RS with wrong-topology subgraph
       Question: p7 (relation synonym)
       Subgraph: p4 ELQ (compositional restructuring — degraded entity topology)
       Expected: EM < RS pipeline EM (20.31%); CR subgraph actively harms RS.

  [3C] S3-Q + RS-SG   → entity-insertion distractor in cleaner graph
       Question: p5 (adversarial entity insertion)
       Subgraph: p7 ELQ (RS — seed entities anchored to answer, not distractor)
       Expected: partial recovery vs S3 pipeline; RS seeds suppress distractor entity.

Each cell runs two GNN-RAG passes:
  1. Clean baseline: original question + original clean subgraph (test.json)
  2. Transfer: perturbed question (q_ptype) + foreign subgraph (sg_ptype)

Results are saved per-cell and a LaTeX-ready comparison table is printed.

Coverage note
-------------
nsm_gem / graftnetv3_gem ablation subgraphs exist only for p4 and p7 in
new_exp/gnn_input/. Using --sg-el-source nsm_gem for sg_ptype=p5 will raise
FileNotFoundError with a clear message — run exp_s1s4_subgraph_build.py first.

Usage:
  # All cells, CWQ, ELQ subgraphs (recommended first run):
  python exp_cross_attack_transfer.py --cell all --dataset cwq

  # Single cell:
  python exp_cross_attack_transfer.py --cell 3a --dataset cwq

  # WebQSP as well:
  python exp_cross_attack_transfer.py --cell 3a 3b --dataset both

  # Use NSM subgraph as the foreign graph (p4/p7 only, requires build first):
  python exp_cross_attack_transfer.py --cell 3a --dataset cwq --sg-el-source nsm_gem

  # Sanity check (first 20 questions):
  python exp_cross_attack_transfer.py --cell all --dataset cwq --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Paths ───────────────────────────────────────────────────────────────────────
REPO_ROOT = REPO_ROOT
GNN_DATA    = REPO_ROOT / "tmlr" / "Repo" / "GNN-RAG" / "data"
GNN_RAG_GNN = REPO_ROOT / "tmlr" / "Repo" / "GNN-RAG" / "gnn"
DATASET_DIR = REPO_ROOT / "tmlr" / "pert" / "dataset"
NEW_EXP     = REPO_ROOT / "new_exp"
RESULTS_DIR = REPO_ROOT / "tmlr" / "results" / "e3_cross_attack"
SCRIPTS_DIR = Path(__file__).parent

CKPT_CWQ    = GNN_DATA / "pretrain" / "ReaRev_CWQ.ckpt"
CKPT_WEBQSP = GNN_DATA / "pretrain" / "ReaRev_webqsp.ckpt"

# Add GNN-RAG and scripts dir to path (needed for lazy imports inside functions)
for _p in (str(GNN_RAG_GNN), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Transfer cell definitions ────────────────────────────────────────────────────
TRANSFER_CELLS: Dict[str, Dict[str, str]] = {
    "3a": {
        "q_ptype":    "p4",
        "sg_ptype":   "p7",
        "label":      "CR-Q + RS-SG",
        "hypothesis": (
            "CR collapses because ELQ cannot link entities in the restructured question. "
            "Using the RS subgraph (same answer entity, valid topology) should recover "
            "EM to near fast-path level (~50%), confirming subgraph failure dominates."
        ),
    },
    "3b": {
        "q_ptype":    "p7",
        "sg_ptype":   "p4",
        "label":      "RS-Q + CR-SG",
        "hypothesis": (
            "RS resilience is partially decoder-driven (relation synonym is transparent). "
            "Swapping in a degraded CR subgraph should worsen RS below its 20.31% pipeline EM, "
            "quantifying how much of RS robustness comes from the intact subgraph."
        ),
    },
    "3c": {
        "q_ptype":    "p5",
        "sg_ptype":   "p7",
        "label":      "S3-Q + RS-SG",
        "hypothesis": (
            "S3 inserts a distractor entity in the question. The RS subgraph seeds are "
            "anchored to the original answer entity (not the distractor), so PPR suppresses "
            "the irrelevant entity. Expect partial EM recovery vs S3 ELQ pipeline."
        ),
    },
}

# Paper baselines (mean EM %) — update after running full experiments
BASELINES: Dict[str, Dict[str, Any]] = {
    "cwq": {
        "clean":    {"em": 52.9},
        "p4":       {"em": 0.68,  "label": "CR (pipeline)"},
        "p7":       {"em": 20.31, "label": "RS (pipeline)"},
        "p5":       {"em": None,  "label": "S3 (pipeline)"},
    },
    "webqsp": {
        "clean":    {"em": 74.3},
        "p4":       {"em": None,  "label": "CR (pipeline)"},
        "p7":       {"em": None,  "label": "RS (pipeline)"},
        "p5":       {"em": None,  "label": "S3 (pipeline)"},
    },
}


# ── Data loading ─────────────────────────────────────────────────────────────────

def load_pert_questions(ptype: str, dataset: str) -> Dict[str, str]:
    """Load {qid: perturbed_question} for a given (dataset, ptype)."""
    path = DATASET_DIR / f"{dataset}_{ptype}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Perturbation file not found: {path}\n"
            f"  Run reorganize_pert.py first to create {dataset}_{ptype}.json"
        )
    with open(path) as f:
        records = json.load(f)
    return {r["id"]: r["perturbed_question"] for r in records}


def _subgraph_path(ptype: str, dataset: str, el_source: str) -> Path:
    """Return the path to test_subgraph.json for (dataset, ptype, el_source)."""
    _ablation = {"nsm_gem", "nsm_elq", "graftnetv3_gem", "graftnetv3_elq"}
    if el_source == "elq":
        return DATASET_DIR / "gnn_input" / f"{dataset}_{ptype}" / "test_subgraph.json"
    elif el_source in _ablation:
        return NEW_EXP / "gnn_input" / f"{dataset}_{ptype}_{el_source}" / "test_subgraph.json"
    else:
        return DATASET_DIR / "gnn_input" / f"{dataset}_{ptype}_{el_source}" / "test_subgraph.json"


def load_subgraph_records(ptype: str, dataset: str, el_source: str = "elq") -> Dict[str, Dict[str, Any]]:
    """Load {qid: record} from the gnn_input directory for (dataset, ptype, el_source)."""
    sg_path = _subgraph_path(ptype, dataset, el_source)
    if not sg_path.exists():
        _ablation = {"nsm_gem", "nsm_elq", "graftnetv3_gem", "graftnetv3_elq"}
        hint = ""
        if el_source in _ablation:
            method   = el_source.split("_")[0]
            seed_mod = el_source.split("_")[1]
            hint = (
                f"\n  This ablation cell requires exp_s1s4_subgraph_build.py to run first:"
                f"\n    python exp_s1s4_subgraph_build.py "
                f"--ptype {ptype} --dataset {dataset} --method {method} --seed-mode {seed_mod}"
            )
        raise FileNotFoundError(f"Subgraph not found: {sg_path}{hint}")
    records: Dict[str, Dict[str, Any]] = {}
    with open(sg_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["id"]] = r
    return records


def load_gnn_orig(dataset: str) -> Dict[str, Dict[str, Any]]:
    """Load the original clean GNN-RAG test records."""
    subdir = "CWQ" if dataset == "cwq" else "webqsp"
    path = GNN_DATA / subdir / "test.json"
    records: Dict[str, Dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["id"]] = r
    return records


def _load_gold(dataset: str) -> Dict[str, List[str]]:
    """Load gold answers for a dataset, delegating to the e1/e2 loader."""
    subdir = "CWQ" if dataset == "cwq" else "webqsp"
    path = GNN_DATA / subdir / "test.json"
    if dataset == "cwq":
        import importlib.util
        spec = importlib.util.spec_from_file_location("e1_mod", str(SCRIPTS_DIR / "e1_cwq_gnn_rag.py"))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_cwq_gold_answers(path)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("e2_mod", str(SCRIPTS_DIR / "e2_webqsp_gnn_rag.py"))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_webqsp_gold_answers(path)


def _run_gnn(dataset: str, records: List[Dict], word_emb: Path, vocab: Path,
             entity_path: Optional[Path] = None) -> Tuple[Dict, Dict, Dict]:
    """Run GNN-RAG inference for the given dataset. Returns (pred_map, cands_map, gnn_metrics)."""
    import importlib.util
    if dataset == "cwq":
        spec = importlib.util.spec_from_file_location("e1_gnn", str(SCRIPTS_DIR / "e1_cwq_gnn_rag.py"))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.run_gnn_rag_on_records(
            records, CKPT_CWQ, GNN_DATA,
            dataset="cwq", dataset_subdir="CWQ",
            word_emb_path=word_emb, vocab_path=vocab,
            entity_path=entity_path,
        )
    else:
        spec = importlib.util.spec_from_file_location("e2_gnn", str(SCRIPTS_DIR / "e2_webqsp_gnn_rag.py"))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.run_gnn_rag_on_records(
            records, CKPT_WEBQSP, GNN_DATA,
            dataset="webqsp",
            word_emb_path=word_emb, vocab_path=vocab,
            entity_path=entity_path,
        )


# ── Transfer record construction ─────────────────────────────────────────────────

def build_transfer_records(
    q_ptype: str,
    sg_ptype: str,
    dataset: str,
    sg_el_source: str = "elq",
    gnn_orig: Optional[Dict[str, Dict]] = None,
    dry_run: bool = False,
) -> Tuple[List[Dict], List[str]]:
    """
    Build inference records: question text from q_ptype + subgraph from sg_ptype.

    The transplant is:
      record["question"]  ← perturbed_question from cwq_{q_ptype}.json
      record["subgraph"]  ← subgraph from gnn_input/{dataset}_{sg_ptype}/test_subgraph.json
      record["entities"]  ← entity seeds from the sg_ptype subgraph

    Gold answers remain those from the original question (same answer, different phrasing).

    Returns (transfer_records, common_ids).
    """
    pert_qs    = load_pert_questions(q_ptype, dataset)
    sg_records = load_subgraph_records(sg_ptype, dataset, sg_el_source)

    common_ids = sorted(set(pert_qs) & set(sg_records))
    if gnn_orig is not None:
        common_ids = [qid for qid in common_ids if qid in gnn_orig]
    if not common_ids:
        raise ValueError(
            f"No overlapping IDs: q_ptype={q_ptype} × sg_ptype={sg_ptype} × gnn_orig"
        )
    if dry_run:
        common_ids = common_ids[:20]

    records = []
    for qid in common_ids:
        r = copy.deepcopy(sg_records[qid])
        r["question"] = pert_qs[qid]   # transplant perturbed question text only
        records.append(r)

    return records, common_ids


# ── Per-cell evaluation ───────────────────────────────────────────────────────────

def run_transfer_cell(
    cell_name: str,
    dataset: str,
    sg_el_source: str = "elq",
    force: bool = False,
    dry_run: bool = False,
    n_boot: int = 1000,
) -> Dict[str, Any]:
    """Run one cross-attack transfer cell. Returns results dict (cached if exists)."""
    from metrics import exact_match_kgqa, f1_kgqa, mean_ci

    cell     = TRANSFER_CELLS[cell_name]
    q_ptype  = cell["q_ptype"]
    sg_ptype = cell["sg_ptype"]
    label    = cell["label"]

    suffix   = f"_{sg_el_source}" if sg_el_source != "elq" else ""
    out_dir  = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"cell_{cell_name}{suffix}.json"

    if out_file.exists() and not force:
        with open(out_file) as f:
            cached = json.load(f)
        em_t = cached.get("EM_transfer", {}).get("mean", 0) * 100
        print(f"  [CACHED] {label} ({dataset}) — n={cached.get('n','?')}, "
              f"EM_transfer={em_t:.1f}%  [{out_file.name}]")
        return cached

    print(f"\n  [{cell_name.upper()}] {label} | dataset={dataset.upper()} | sg_el={sg_el_source}")
    print(f"  Q-type: {q_ptype} (perturbed question) | SG-type: {sg_ptype} (subgraph topology)")
    print(f"  Hypothesis: {cell['hypothesis'][:80]}...")

    emb_dir    = DATASET_DIR / "embeddings"
    word_emb   = emb_dir / f"{dataset}_pert_word_emb.npy"
    vocab_path = emb_dir / f"{dataset}_pert_vocab.txt"

    # Load clean originals (for baseline pass and gold answers)
    print(f"  Loading clean GNN records...")
    gnn_orig     = load_gnn_orig(dataset)
    gold_answers = _load_gold(dataset)

    # Build transfer records: Q from q_ptype, SG from sg_ptype
    print(f"  Building transfer records (Q={q_ptype} + SG={sg_ptype}/{sg_el_source})...")
    try:
        transfer_records, common_ids = build_transfer_records(
            q_ptype, sg_ptype, dataset, sg_el_source,
            gnn_orig=gnn_orig, dry_run=dry_run,
        )
    except FileNotFoundError as exc:
        print(f"  [SKIP] {exc}")
        return {"skipped": True, "cell": cell_name, "dataset": dataset, "reason": str(exc)}
    except ValueError as exc:
        print(f"  [SKIP] {exc}")
        return {"skipped": True, "cell": cell_name, "dataset": dataset, "reason": str(exc)}

    print(f"  {len(common_ids)} aligned questions")

    # Clean baseline: original question + original clean subgraph
    clean_records = [gnn_orig[qid] for qid in common_ids]

    # Extended entity vocab for ablation subgraphs (NSM/GraftNet add new entity MIDs)
    _ablation = {"nsm_gem", "nsm_elq", "graftnetv3_gem", "graftnetv3_elq"}
    sg_dir    = _subgraph_path(sg_ptype, dataset, sg_el_source).parent
    entity_path = sg_dir / "entities.txt" if sg_el_source in _ablation else None

    print(f"  [Pass 1/2] Clean baseline — original question + original subgraph...")
    t0 = time.time()
    clean_preds, _, _ = _run_gnn(dataset, clean_records, word_emb, vocab_path)
    print(f"    Done in {(time.time()-t0)/60:.1f}min")

    print(f"  [Pass 2/2] Transfer — Q={q_ptype} + SG={sg_ptype}/{sg_el_source}...")
    t0 = time.time()
    transfer_preds, _, _ = _run_gnn(dataset, transfer_records, word_emb, vocab_path,
                                    entity_path=entity_path)
    print(f"    Done in {(time.time()-t0)/60:.1f}min")

    # Compute metrics
    em_clean_list, em_transfer_list = [], []
    f1_clean_list, f1_transfer_list = [], []
    per_q = []

    for qid in common_ids:
        gold  = gold_answers.get(qid, [])
        c_p   = clean_preds.get(qid, "")
        t_p   = transfer_preds.get(qid, "")
        em_c  = exact_match_kgqa(c_p, gold)
        em_t  = exact_match_kgqa(t_p, gold)
        f1_c  = f1_kgqa(c_p, gold)
        f1_t  = f1_kgqa(t_p, gold)
        em_clean_list.append(em_c)
        em_transfer_list.append(em_t)
        f1_clean_list.append(f1_c)
        f1_transfer_list.append(f1_t)
        per_q.append({
            "id":            qid,
            "em_clean":      em_c,
            "em_transfer":   em_t,
            "f1_clean":      f1_c,
            "f1_transfer":   f1_t,
            "clean_pred":    c_p,
            "transfer_pred": t_p,
        })

    dag_em = [c - t for c, t in zip(em_clean_list, em_transfer_list)]
    dag_f1 = [c - t for c, t in zip(f1_clean_list, f1_transfer_list)]

    result = {
        "cell":              cell_name,
        "label":             label,
        "dataset":           dataset,
        "q_ptype":           q_ptype,
        "sg_ptype":          sg_ptype,
        "sg_el_source":      sg_el_source,
        "hypothesis":        cell["hypothesis"],
        "n":                 len(common_ids),
        "EM_clean":          mean_ci(em_clean_list, n_boot),
        "EM_transfer":       mean_ci(em_transfer_list, n_boot),
        "delta_clean_transfer": mean_ci(dag_em, n_boot),
        "F1_clean":          mean_ci(f1_clean_list, n_boot),
        "F1_transfer":       mean_ci(f1_transfer_list, n_boot),
        "baseline_pipeline_EM": BASELINES.get(dataset, {}).get(q_ptype, {}).get("em"),
        "baseline_clean_EM":    BASELINES.get(dataset, {}).get("clean", {}).get("em"),
        "per_question":      per_q,
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    em_t_mean = result["EM_transfer"]["mean"] * 100
    print(f"  Saved → {out_file}  (EM_transfer={em_t_mean:.1f}%)")
    return result


# ── Summary table ─────────────────────────────────────────────────────────────────

def print_summary(results: List[Dict[str, Any]], dataset: str) -> None:
    bsl = BASELINES.get(dataset, {})
    print(f"\n{'='*90}")
    print(f"  CROSS-ATTACK TRANSFER — {dataset.upper()}")
    print(f"  Paper baselines:  clean={bsl.get('clean',{}).get('em','?')}%  "
          f"CR={bsl.get('p4',{}).get('em','?')}%  "
          f"RS={bsl.get('p7',{}).get('em','?')}%")
    print(f"{'='*90}")
    print(f"  {'Cell':<6} {'Setup':<22} {'n':>5}  {'EM_clean':>9} {'EM_xfer':>9} "
          f"{'Δ':>7}  {'Pipeline':>9}  Verdict")
    print(f"  {'-'*88}")

    for r in results:
        if r.get("skipped"):
            cell = r.get("cell", "?")
            reason = r.get("reason", "")[:50]
            print(f"  {cell:<6} {'SKIPPED':<22}  {reason}")
            continue

        em_c   = r["EM_clean"]["mean"] * 100
        em_t   = r["EM_transfer"]["mean"] * 100
        delta  = r["delta_clean_transfer"]["mean"] * 100
        bsl_em = r.get("baseline_pipeline_EM")
        bsl_s  = f"{bsl_em:.1f}%" if bsl_em is not None else "n/a"

        verdict = ""
        if bsl_em is not None:
            gap = em_t - bsl_em
            if r["cell"] == "3a" and gap > 10:
                verdict = f"✓ subgraph failure confirmed (+{gap:.0f}% vs pipeline)"
            elif r["cell"] == "3b" and gap < -3:
                verdict = f"✓ CR-SG actively degrades RS ({gap:+.0f}%)"
            elif r["cell"] == "3c" and gap > 3:
                verdict = f"✓ RS-SG suppresses S3 distractor (+{gap:.0f}%)"

        print(f"  {r['cell']:<6} {r['label']:<22} {r['n']:>5}  "
              f"{em_c:>8.1f}% {em_t:>8.1f}% {delta:>6.1f}%  {bsl_s:>9}  {verdict}")

    print(f"\n  Interpretation guide:")
    print(f"    3A EM_transfer >> CR pipeline (0.68%) → subgraph failure dominates CR collapse")
    print(f"    3A EM_transfer ≈ fast-path EM (~50%) → confirms GNN decoder is intact under CR")
    print(f"    3B EM_transfer < RS pipeline (20.31%) → CR subgraph actively harmful (not neutral)")
    print(f"    3C EM_transfer > S3 pipeline → RS PPR seeds suppress distractor entity")


# ── Main ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Cross-attack subgraph transfer (Experiment #3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cell", nargs="+", default=["all"],
        choices=list(TRANSFER_CELLS) + ["all"],
        help="Transfer cell(s) to run. all = 3a 3b 3c",
    )
    p.add_argument("--dataset", default="cwq", choices=["cwq", "webqsp", "both"])
    p.add_argument(
        "--sg-el-source", default="elq",
        choices=["elq", "nsm_gem", "graftnetv3_gem", "nsm_elq", "graftnetv3_elq"],
        help=(
            "EL source for the FOREIGN subgraph (the SG in Q+SG transplant). "
            "nsm_gem/graftnetv3_gem cells require exp_s1s4_subgraph_build.py "
            "to have completed for the relevant ptype first."
        ),
    )
    p.add_argument("--force", action="store_true",
                   help="Re-run even if output file already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Limit to first 20 questions per cell (sanity check)")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap iterations for CI (default: 1000)")
    args = p.parse_args()

    cells    = list(TRANSFER_CELLS.keys()) if "all" in args.cell else args.cell
    datasets = ["cwq", "webqsp"] if args.dataset == "both" else [args.dataset]

    print(f"Cross-attack transfer: cells={cells} datasets={datasets} sg_el={args.sg_el_source}")
    if args.dry_run:
        print("[DRY RUN] Limited to 20 questions per cell.")

    # Coverage warning: ablation sources only cover p4/p7 natively
    _ablation = {"nsm_gem", "nsm_elq", "graftnetv3_gem", "graftnetv3_elq"}
    if args.sg_el_source in _ablation:
        print(f"\n[COVERAGE NOTE] sg_el_source={args.sg_el_source} ablation subgraphs exist only "
              f"for p4 and p7 in new_exp/gnn_input/.")
        for cell_name in cells:
            sg_ptype = TRANSFER_CELLS[cell_name]["sg_ptype"]
            if sg_ptype not in {"p4", "p7"}:
                print(f"  Cell {cell_name} uses sg_ptype={sg_ptype} — "
                      f"run exp_s1s4_subgraph_build.py --ptype {sg_ptype} first.")

    all_results: Dict[str, List] = {ds: [] for ds in datasets}

    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset.upper()}")
        for cell_name in cells:
            res = run_transfer_cell(
                cell_name=cell_name,
                dataset=dataset,
                sg_el_source=args.sg_el_source,
                force=args.force,
                dry_run=args.dry_run,
                n_boot=args.n_boot,
            )
            all_results[dataset].append(res)

    for dataset in datasets:
        print_summary(all_results[dataset], dataset)

    print("\n[NEXT STEPS]")
    print("  1. If 3A EM > 40%: update analysis.tex §4 — subgraph failure is the dominant factor")
    print("  2. If 3A EM ≈ fast-path EM (~52%): strengthen paper claim that GNN decoder is intact")
    print("  3. If 3B EM < RS pipeline: add 'CR subgraph actively harmful' note to ablate.tex")
    print("  4. Add transfer oracle row to Appendix Table (tab:secondary) in ablate.tex")
    print("  5. Compare 3A EM vs PPR mass α_pert (from exp_ppr_mass.py Part A)")


if __name__ == "__main__":
    main()
