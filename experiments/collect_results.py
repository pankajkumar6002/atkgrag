#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
collect_results.py
==================
Reads all experiment JSON/JSONL result files and writes tmlr/results/sum.md.
Safe to run repeatedly — overwrites sum.md with latest values each time.

Usage:
  python collect_results.py              # update sum.md once
  python collect_results.py --watch 60  # re-run every 60s (for monitoring)
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = REPO_ROOT
RESULTS     = REPO_ROOT / "results"
SUM_MD      = RESULTS / "sum.md"

# ── Loaders ──────────────────────────────────────────────────────────────────

def _jload(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.load(open(path))
    except Exception:
        return None

def _load_per_question(path: Path) -> Optional[List[Dict]]:
    try:
        return json.load(open(path))
    except Exception:
        return None

def _em_pert(recs: List[Dict]) -> float:
    return 100 * sum(r["em_pert"] for r in recs) / len(recs)

def _em_clean(recs: List[Dict]) -> float:
    return 100 * sum(r["em_clean"] for r in recs) / len(recs)

def _cell(val: Optional[float], fmt: str = ".1f") -> str:
    return f"{val:{fmt}}%" if val is not None else "—"


# ── Section: Full-pipeline EM (perturbed subgraph) ───────────────────────────

def section_full_pipeline() -> str:
    """All full-pipeline EM results: new subgraph built from perturbed question."""
    ptypes = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
    labels = {"p1":"ES","p2":"S1","p3":"S2","p4":"CR","p5":"S3","p6":"S4","p7":"RS"}
    variants = [
        ("full_elq",              "ELQ"),
        ("full_nsm_gem",          "NSM-GEM"),
        ("full_graftnetv3_fixed", "GraftNet-GEM★"),  # ★ = bug-fixed canonical build (May 12)
        ("full_nsm_elq",          "NSM-ELQ"),
        ("full_graftnetv3_elq",   "GraftNet-ELQ"),
    ]
    # Clean baselines
    clean = {"cwq": 52.9, "webqsp": 74.3}

    lines = []
    for ds, exp in [("CWQ","e1_cwq_gnn_rag"), ("WebQSP","e2_webqsp_gnn_rag")]:
        lines.append(f"\n### {ds} (clean baseline = {clean[ds.lower()]}%)\n")
        header = "| Variant |" + "".join(f" {labels[p]}/{p} |" for p in ptypes)
        sep    = "|---------|" + "".join(" --------- |" for _ in ptypes)
        lines.append(header)
        lines.append(sep)
        for var_key, var_label in variants:
            row = f"| {var_label} |"
            has_any = False
            for pt in ptypes:
                fpath = RESULTS / exp / var_key / f"{pt}_per_question.json"
                recs = _load_per_question(fpath)
                if recs:
                    em = _em_pert(recs)
                    row += f" {em:.1f}% |"
                    has_any = True
                else:
                    row += " — |"
            if has_any:
                lines.append(row)

    return f"""## Full-Pipeline EM (new subgraph from perturbed question)

**All variants run on the full test set. ELQ = standard entity linking.
NSM/GraftNet = relation-weighted PPR. GEM seeds = gold entity MIDs (oracle EL).
ELQ seeds = ELQ entity links applied to the perturbed question.
★ GraftNet-GEM★ = bug-fixed canonical build (subgraph_var/, May 12); use these numbers for the paper.
  The older `graftnetv3_gem` build (new_exp/, May 7) had a subgraph construction bug — do NOT use.**
{"".join(lines)}

**Key findings:**
- ELQ CR (p4) collapses to <1% — ELQ cannot link entities in restructured questions
- NSM-GEM / GraftNet-GEM recover CR to ~25–31% by using oracle seeds (isolates EL failure)
- ELQ RS (p7) = 20% CWQ / 51% WebQSP — RS preserves entity names so ELQ partially survives
- S1-S4 (p2,p3,p5,p6) for NSM-GEM/GraftNet-GEM: PENDING (subgraph build in progress)
"""


# ── Section: Fast-path EM (decoder isolation) ─────────────────────────────────

def section_fast_path() -> str:
    """Table 5 fast-path EM: clean subgraph + perturbed question."""
    ptypes = [
        ("p2", "S1/Rel-para"),
        ("p3", "S2/Voice"),
        ("p4", "CR/Comp-restr"),
        ("p5", "S3/Ent-ins"),
        ("p6", "S4/Mixed"),
        ("p7", "RS/Rel-syn"),
    ]
    clean = {"cwq": 52.9, "webqsp": 74.3}

    rows_cwq, rows_webqsp = [], []
    for pt, label in ptypes:
        for ds, exp, rows in [
            ("cwq",    "e1_cwq_gnn_rag",    rows_cwq),
            ("webqsp", "e2_webqsp_gnn_rag", rows_webqsp),
        ]:
            fpath = RESULTS / exp / "fast_elq" / f"{pt}_per_question.json"
            recs = _load_per_question(fpath)
            if recs:
                em_f  = _em_pert(recs)
                drop  = clean[ds] - em_f   # positive = performance dropped
                note  = "★ NEW" if pt in ("p2","p6") else ""
                # Show drop as negative (−Xpp = decoder lost Xpp vs clean)
                rows.append(f"| {label} | {em_f:.1f}% | {-drop:+.1f}pp | {note} |")
            else:
                rows.append(f"| {label} | — | — | PENDING |")

    cwq_table    = "\n".join(rows_cwq)
    webqsp_table = "\n".join(rows_webqsp)

    return f"""## Fast-Path EM — GNN Decoder Isolation (Table 5)

**Design:** Clean ELQ subgraph + perturbed question → isolates decoder from subgraph failure.
Drop vs clean baseline = decoder sensitivity.

### CWQ (clean = {clean['cwq']}%)
| Type | EM_fast | Δ_decoder | Note |
|------|---------|-----------|------|
{cwq_table}

### WebQSP (clean = {clean['webqsp']}%)
| Type | EM_fast | Δ_decoder | Note |
|------|---------|-----------|------|
{webqsp_table}

**Key findings:**
- S1 (p2) and S4 (p6) decoder drops are <3pp — decoder is robust; full-pipeline drops are subgraph failures
- CR (p4) fast-path not shown here (fast mode uses clean subgraph, CR pipeline failure is EL-driven)
"""


# ── Section: NSM Isolation ────────────────────────────────────────────────────

def section_nsm_isolation() -> str:
    """EXP-3: NSM decoder isolation — full test set."""
    cells = [
        ("cwq",    "p4", "CR"),
        ("cwq",    "p7", "RS"),
        ("webqsp", "p4", "CR"),
        ("webqsp", "p7", "RS"),
    ]
    rows = []
    for ds, ptype, label in cells:
        fpath = RESULTS / "nsm_isolation" / f"nsm_isolation_{ds}_{ptype}.json"
        r = _jload(fpath)
        if not r:
            rows.append(f"| {ds.upper()} {label} ({ptype}) | — | — | — | — | PENDING |")
            continue
        n       = r.get("n", "?")
        h1_c    = r.get("hit1_clean", 0)
        h1_p    = r.get("hit1_pert", 0)
        # delta_hit1 stored as clean−pert (positive = drop); display as negative for clarity
        drop    = r.get("delta_hit1", 0)
        delta_display = -drop   # negative means degradation
        verdict = "robust ✓" if abs(drop) < 5 else f"**sensitive** ({delta_display:+.1f}pp) ✗"
        rows.append(
            f"| {ds.upper()} {label} ({ptype}) | {n} "
            f"| {h1_c:.2f}% | {h1_p:.2f}% | {delta_display:+.2f}pp | {verdict} |"
        )
    table = "\n".join(rows)
    return f"""## NSM Decoder Isolation (EXP-3, full test set)

**Design:** Clean NSM subgraph + perturbed question → measures decoder robustness independent of subgraph failure.
NSM clean baselines: CWQ=40.44%, WebQSP=68.33% (vs ReaRev: 52.9% / 74.3%).

| Cell | n | Hit@1 Clean | Hit@1 Pert | Δ | Verdict |
|------|---|------------|-----------|---|---------|
{table}

**Interpretation:**
- CWQ: NSM decoder is **robust** to both CR and RS (Δ<1pp) — confirms claim B6 for CWQ ✓
- WebQSP CR: Δ=−10.86pp — NSM decoder **sensitive** to CR on WebQSP ✗ (nuances claim B6)
- WebQSP RS: Δ=−4.88pp — moderate sensitivity

**Paper note:** Claim B6 ("decoder robustness generalises beyond ReaRev") holds for CWQ but NOT WebQSP CR.
Paper should say: "NSM decoder is robust on CWQ; WebQSP CR shows 10.9pp decoder sensitivity."
"""


# ── Section: Cross-Attack Transfer ───────────────────────────────────────────

def section_cross_attack() -> str:
    """Cross-attack subgraph transfer results (all 6 cells)."""
    # Pipeline EMs (from full_elq results)
    pipeline = {
        ("cwq",    "p4"): 0.68,  ("cwq",    "p7"): 20.31, ("cwq",    "p5"): 0.40,
        ("webqsp", "p4"): 0.49,  ("webqsp", "p7"): 50.95, ("webqsp", "p5"): 0.43,
    }
    cells = [
        ("3a", "CR-Q + RS-SG",  "p4", "p7"),
        ("3b", "RS-Q + CR-SG",  "p7", "p4"),
        ("3c", "S3-Q + RS-SG",  "p5", "p7"),
    ]
    rows = []
    for ds in ("cwq", "webqsp"):
        for cell, label, qp, sgp in cells:
            fpath = RESULTS / "e3_cross_attack" / ds / f"cell_{cell}.json"
            r = _jload(fpath)
            if not r or r.get("skipped"):
                rows.append(f"| {ds.upper()} {cell.upper()} | {label} | — | — | — | — | PENDING |")
                continue
            n          = r.get("n", "?")
            em_c       = r.get("EM_clean", {}).get("mean", 0) * 100
            em_t       = r.get("EM_transfer", {}).get("mean", 0) * 100
            # delta_clean_transfer = clean - transfer (positive = transfer is worse than clean)
            drop_vs_clean = r.get("delta_clean_transfer", {}).get("mean", 0) * 100
            pipe_em    = pipeline.get((ds, qp), None)
            pipe_str   = f"{pipe_em:.2f}%" if pipe_em is not None else "—"
            # Gain = transfer - pipeline (positive = recovery from pipeline baseline)
            gain_str   = f"{em_t - (pipe_em or 0):+.1f}pp" if pipe_em is not None else "—"
            # Drop vs clean = transfer - clean (negative = transfer is below clean)
            drop_str   = f"{em_t - em_c:+.1f}pp"
            rows.append(
                f"| {ds.upper()} {cell.upper()} | {label} | {n} "
                f"| {em_c:.1f}% | {em_t:.1f}% | {drop_str} "
                f"| {pipe_str} | {gain_str} |"
            )
    table = "\n".join(rows)

    return f"""## Cross-Attack Subgraph Transfer (Exp #3)

**Design:** Swap question text from attack-type A onto subgraph from attack-type B.
Causally isolates subgraph retrieval failure vs GNN decoder failure.
Pipeline EM = full pipeline with perturbed Q+SG (the baseline we want to beat).
Gain = EM_transfer − Pipeline_EM (positive = subgraph swap helped). Drop = EM_transfer − EM_clean (negative = below clean).

| Dataset | Cell | Setup | n | EM_clean | EM_transfer | Drop(vs clean) | Pipeline_EM | Gain(vs pipeline) |
|---------|------|-------|---|----------|------------|----------------|-------------|-------------------|
{table}

**Interpretation:**
- **3A** CR-Q + RS-SG: CWQ 0.68%→20.1% (+29pp), WebQSP 0.49%→49.7% (+49pp)
  Subgraph retrieval dominates CR failure. WebQSP near-full recovery — CR is almost purely EL failure.
  CWQ partial recovery (20% vs clean 52.9%) — CR question text also degrades GNN decoding.
- **3B** RS-Q + CR-SG: CWQ 20.31%→11.7% (−8.6pp), WebQSP 50.95%→33.4% (−17.6pp)
  CR subgraph actively degrades RS performance — bad subgraph harms even robust RS decoder.
- **3C** S3-Q + RS-SG: CWQ 0.40%→19.8% (+19pp), WebQSP 0.43%→50.8% (+50pp)
  RS subgraph anchoring suppresses distractor entity — near-full recovery on WebQSP.
"""


# ── Section: S1-S4 CWQ ablation ──────────────────────────────────────────────

def _subgraph_stats(ds: str, pt: str, src: str) -> str:
    """Return avg_sg_ents for a subgraph dir (for diagnosis)."""
    import json as _json
    NEW_EXP = REPO_ROOT.parent / "new_exp"
    exp_dir = NEW_EXP / "gnn_input" / f"{ds}_{pt}_{src}"
    f = exp_dir / "test_subgraph.json"
    if not f.exists():
        return "?"
    try:
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        recs = [_json.loads(l) for l in lines[:100]]  # sample 100 for speed
        avg = sum(len(r["subgraph"]["entities"]) for r in recs) / len(recs)
        return f"{avg:.0f}"
    except Exception:
        return "?"


def section_s1s4_cwq() -> str:
    """S1-S4 CWQ subgraph ablation (ELQ, NSM-GEM, GraftNet-GEM) with diagnosis."""
    ptypes = [("p2","S1/Rel-para"), ("p3","S2/Voice"), ("p5","S3/Ent-ins"), ("p6","S4/Mixed")]
    sources = [
        ("full_elq",           "ELQ",           None),
        ("full_nsm_gem",       "NSM-GEM",       "nsm_gem"),
        ("full_graftnetv3_gem","GraftNet-GEM",   "graftnetv3_gem"),
    ]
    rows = []
    for pt, label in ptypes:
        row = f"| {label} |"
        for src_key, _, sg_src in sources:
            fpath = RESULTS / "e1_cwq_gnn_rag" / src_key / f"{pt}_per_question.json"
            recs = _load_per_question(fpath)
            em_str = f"{_em_pert(recs):.1f}%" if recs else "PENDING"
            if sg_src:
                avg_e = _subgraph_stats("cwq", pt, sg_src)
                row += f" {em_str} (sg={avg_e}e) |"
            else:
                row += f" {em_str} |"
        rows.append(row)
    header = "| Type | ELQ | NSM-GEM (sg size) | GraftNet-GEM (sg size) |"
    sep    = "|------|-----|------------------|----------------------|"
    table  = "\n".join(rows)

    # Reference: p4 and p7 for context
    ref_rows = []
    for pt, label in [("p4","CR/p4"), ("p7","RS/p7")]:
        row = f"| {label} |"
        for src_key, _, sg_src in sources:
            fpath = RESULTS / "e1_cwq_gnn_rag" / src_key / f"{pt}_per_question.json"
            recs = _load_per_question(fpath)
            em_str = f"{_em_pert(recs):.1f}%" if recs else "—"
            if sg_src:
                avg_e = _subgraph_stats("cwq", pt, sg_src)
                row += f" {em_str} (sg={avg_e}e) |"
            else:
                row += f" {em_str} |"
        ref_rows.append(row)
    ref_table = "\n".join(ref_rows)

    return f"""## S1-S4 CWQ Subgraph Ablation

**Claim:** NSM relation-weighted PPR recovers S1 (relation paraphrase); GraftNet cosine PPR recovers S3.
Clean baseline = 52.9%. ELQ clean subgraph avg 1338 entities.

{header}
{sep}
{table}

**Reference (p4/p7 for comparison):**

{header}
{sep}
{ref_table}

### Diagnosis: Why only 15-21% EM?

**Root cause: subgraph sparsity, not perturbation sensitivity.**

Subgraph sizes (avg entities per question):
- ELQ clean (training distribution): **1338 entities, 4273 tuples**
- S1-S4 NSM-GEM: **238-243 entities, 390-394 tuples** (5.5× smaller than clean)
- S1-S4 GraftNet-GEM: **81-85 entities, 167-174 tuples** (16× smaller than clean)
- p4 NSM-GEM: **647 entities** — 2.7× larger than S1-S4 NSM → explains p4 EM=30% vs S1-S4 EM=15%
- p7 NSM-GEM: **755 entities** — 3.1× larger → explains p7 EM=37%

**Why are S1-S4 subgraphs so small?**
NSM 2-hop BFS uses the **perturbed question text** for relation scoring during PPR.
S1-S4 perturbations (relation paraphrase, voice flip, entity insertion, mixed noise)
alter the question's lexical surface more severely than p4/p7, causing NSM's relation
scoring to match fewer triples → smaller BFS expansion → sparse subgraph.
p4/p7 preserve enough semantic signal for NSM to retrieve a richer neighbourhood.

**EM decomposition:**
- Drop from 52.9% → 15% is NOT primarily due to perturbation confusing the GNN decoder
- It is primarily due to gold answer MIDs being absent from the sparse subgraph (low α_pert)
- This is the same mechanism as ELQ (EL failure), just a different failure mode (PPR scoring failure)

**Paper implication:**
Cannot claim "NSM/GraftNet recover S1-S4." Instead:
"NSM/GraftNet subgraphs for secondary perturbations are severely sparse (~238 vs 1338 entities),
limiting EM to 15-21% — primarily a subgraph coverage failure, not decoder robustness."
The partial recovery vs ELQ (<1%) is real but modest; the ceiling is set by subgraph recall.

**LaTeX target:** `ablate.tex` Table (tab:secondary) — add NSM-GEM and GraftNet-GEM columns with caveat footnote.
"""


# ── Section: S1-S4 WebQSP ablation ───────────────────────────────────────────

def section_s1s4_webqsp() -> str:
    """S1-S4 WebQSP subgraph ablation (ELQ, NSM-GEM, GraftNet-GEM) with diagnosis."""
    ptypes = [("p2","S1/Rel-para"), ("p3","S2/Voice"), ("p5","S3/Ent-ins"), ("p6","S4/Mixed")]
    sources = [
        ("full_elq",           "ELQ",           None),
        ("full_nsm_gem",       "NSM-GEM",       "nsm_gem"),
        ("full_graftnetv3_gem","GraftNet-GEM",   "graftnetv3_gem"),
    ]
    rows = []
    for pt, label in ptypes:
        row = f"| {label} |"
        for src_key, _, sg_src in sources:
            fpath = RESULTS / "e2_webqsp_gnn_rag" / src_key / f"{pt}_per_question.json"
            recs = _load_per_question(fpath)
            em_str = f"{_em_pert(recs):.1f}%" if recs else "PENDING"
            if sg_src:
                avg_e = _subgraph_stats("webqsp", pt, sg_src)
                row += f" {em_str} (sg={avg_e}e) |"
            else:
                row += f" {em_str} |"
        rows.append(row)
    header = "| Type | ELQ | NSM-GEM (sg size) | GraftNet-GEM (sg size) |"
    sep    = "|------|-----|------------------|----------------------|"
    table  = "\n".join(rows)

    return f"""## S1-S4 WebQSP Subgraph Ablation

**Claim:** S1-S4 secondary types show subgraph-method sensitivity on WebQSP (not just CWQ).
Clean baseline = 74.3%.

{header}
{sep}
{table}

**Notable:** WebQSP GraftNet-GEM reaches ~49% (vs 74.3% clean) — much better than CWQ (~21%).
WebQSP questions are simpler (1-hop), so even sparse GraftNet subgraphs contain the answer.
Same subgraph sparsity issue applies but WebQSP is more forgiving: answer MID is reachable
in fewer hops → sparse subgraph still covers it.

**LaTeX target:** `ablate.tex` Table (tab:secondary) — add WebQSP columns.
"""


# ── Section: PPR Mass ─────────────────────────────────────────────────────────

def section_ppr_mass() -> str:
    """EXP-1: PPR mass / gold-answer reachability."""
    fpath = RESULTS / "ppr_mass_cwq_p4.json"
    r = _jload(fpath)
    if not r:
        return "## EXP-1: PPR Mass — PENDING\n"

    # Part A — variants nested under part_a["variants"]
    part_a = r.get("part_a", {})
    variants_a = part_a.get("variants", part_a)  # fallback: old format where variants were top-level
    rows_a = []
    for var, stats in variants_a.items():
        if not isinstance(stats, dict):
            continue
        n    = stats.get("n", "?")
        corr = stats.get("correct_pct", 0)
        pres = stats.get("present_wrong_pct", 0)
        abs_ = stats.get("absent_pct", 0)
        # stored as "alpha_pert" (plain float, already a percentage)
        alph = stats.get("alpha_pert", stats.get("alpha_pert_pct", 0))
        rows_a.append(f"| {var.upper()} | {n} | {corr:.1f}% | {pres:.1f}% | {abs_:.1f}% | {alph:.1f}% |")
    table_a = "\n".join(rows_a) if rows_a else "| (no data) |"

    # Part B
    part_b = r.get("part_b", {})
    part_b_str = "PENDING"
    if part_b:
        n_proc  = part_b.get("n_processed", 0)
        med_c   = part_b.get("clean_median_ppr", 0)
        med_p   = part_b.get("pert_median_ppr", 0)
        ratio   = part_b.get("collapse_ratio", 1.0)
        part_b_str = (
            f"n={n_proc} | Clean median={med_c:.4f} | Pert median={med_p:.4f} | "
            f"Ratio={ratio:.3f}× (**NULL**: PPR routing unchanged — decoder failure, not PPR)"
        )

    return f"""## EXP-1: PPR Mass / Gold-Answer Reachability (CWQ CR/p4)

**Claim (B5):** "Answer present but GNN cannot reach it" — answer in subgraph but EM≈0%.

### Part A — Answer Reachability Breakdown (n=3531)

| Variant | n | EM=1 (correct) | Present+Wrong | Absent | α_pert |
|---------|---|----------------|--------------|--------|--------|
{table_a}

**Key finding (ELQ):** Gold answer is present in **74.0%** of CR subgraphs yet EM=0.68%.
Supports claim B5: answer reachable topologically but GNN cannot decode it.
NSM-GEM/GraftNet-GEM have lower α_pert (~61–64%) but higher EM (~30%) — they route mass better.

### Part B — PPR Score Distribution (200-question sample)

{part_b_str}

**Mechanistic interpretation:** ELQ CR — PPR seeding is unchanged (same entity seeds).
Failure is in GNN cross-attention (message passing), not PPR routing.
CR restructures compositional hop order → GNN attention weights wrong edges.
This is consistent with 3A partial recovery: even with RS subgraph, CR question confuses GNN.

**LaTeX target:** `analysis.tex` §4 — add α_pert=74% statistic and decoder-failure interpretation.
"""


# ── Section: Run Status ───────────────────────────────────────────────────────

def section_run_status() -> str:
    """Current run status for all experiments."""
    def check(paths: list) -> str:
        return "DONE ✓" if all(Path(p).exists() for p in paths) else "PENDING"

    nsm_iso  = check([str(RESULTS / "nsm_isolation" / f"nsm_isolation_{d}_{p}.json")
                      for d in ("cwq","webqsp") for p in ("p4","p7")])
    cross    = check([str(RESULTS / "e3_cross_attack" / d / f"cell_{c}.json")
                      for d in ("cwq","webqsp") for c in ("3a","3b","3c")])
    ppr_a    = "DONE ✓" if (RESULTS / "ppr_mass_cwq_p4.json").exists() else "PENDING"
    ppr_b    = "DONE ✓" if (_jload(RESULTS / "ppr_mass_cwq_p4.json") or {}).get("part_b") else "PENDING"
    fast_cwq = check([str(RESULTS / "e1_cwq_gnn_rag" / "fast_elq" / f"{p}_per_question.json")
                      for p in ("p2","p6")])
    fast_wsp = check([str(RESULTS / "e2_webqsp_gnn_rag" / "fast_elq" / f"{p}_per_question.json")
                      for p in ("p2","p6")])

    # Count s1s4 cells done
    s1s4_done = sum(
        1 for ds in ("cwq","webqsp")
        for pt in ("p2","p3","p5","p6")
        for method in ("nsm_gem","graftnetv3_gem")
        if (REPO_ROOT / "data" / "new_exp_gnn_input" / f"{ds}_{pt}_{method}" / "test_subgraph.json").exists()
    )
    s1s4_status = f"{s1s4_done}/16 cells done (building...)" if s1s4_done < 16 else "DONE ✓"

    return f"""## Run Status

| Experiment | Status | Notes |
|-----------|--------|-------|
| NSM Isolation (CWQ+WebQSP, p4+p7, full set) | {nsm_iso} | `results/nsm_isolation/` |
| Cross-Attack Transfer (3A/3B/3C, both datasets) | {cross} | `results/e3_cross_attack/` |
| PPR Mass Part A (CWQ CR breakdown) | {ppr_a} | `results/ppr_mass_cwq_p4.json` |
| PPR Mass Part B (PPR score distribution, n=200) | {ppr_b} | Null result — decoder failure |
| Fast-Path EM S1+S4 CWQ | {fast_cwq} | `results/e1_cwq_gnn_rag/fast_elq/` |
| Fast-Path EM S1+S4 WebQSP | {fast_wsp} | `results/e2_webqsp_gnn_rag/fast_elq/` |
| S1-S4 Subgraph Build (16 cells) | {s1s4_status} | `new_exp/gnn_input/<ds>_<pt>_<method>/` |
| S1-S4 GNN Inference (NSM+GraftNet GEM) | {"DONE ✓" if s1s4_done >= 16 else "PENDING"} | `results/e1_cwq_gnn_rag/full_nsm_gem/` etc. |
| EXPLAIGNN RS (ConvMix Wikidata) | DONE ✓ | `results/convmix_rs/` RS valid=76.1% |

## LaTeX Targets (priority order)

1. `exp.tex` Table 5 — add NSM row from NSM isolation (Hit@1 clean/pert, Δ)
2. `exp.tex` Table 5 — add S1 (p2) and S4 (p6) fast-path EM_fast rows (★ NEW)
3. `analysis.tex` §4 — add cross-attack oracle (3A: CR recovery 0.68%→20.1% CWQ / 0.49%→49.7% WebQSP)
4. `analysis.tex` §4 — add PPR mass: α_pert=74% yet EM=0.68%, decoder-failure interpretation
5. `ablate.tex` Tab:secondary — add NSM-GEM/GraftNet-GEM columns with subgraph sparsity caveat footnote
6. `disc.tex` Limitations — "scope to ReaRev and NSM; WebQSP CR shows NSM decoder sensitivity"

## Critical Result Caveats

⚠️ **Claim B6 (decoder robustness):** NSM decoder robust on CWQ (Δ<1pp) but WebQSP CR shows
Δ=−10.86pp — paper must not over-claim generalisation. Say: "robust on CWQ; WebQSP CR is dataset-specific."

⚠️ **3A partial recovery (CWQ):** Transfer=20.1% vs clean=52.9%. Even with RS subgraph,
CR question text degrades GNN decoding by ~33pp. Subgraph failure dominates (30× improvement
from pipeline) but decoder also contributes. WebQSP 3A recovers to 49.7% — closer to clean.

⚠️ **PPR Part B null result:** Clean and pert PPR mass are identical (0.987×). Do NOT claim
"PPR mass collapses under CR." Instead: "ELQ CR subgraph seeding is unchanged; GNN cross-attention fails."

⚠️ **S1-S4 NSM/GraftNet EM (15-21%) is NOT decoder robustness:** The drop from 52.9% is primarily
due to subgraph sparsity. S1-S4 NSM subgraphs have only ~238 entities vs 1338 in clean ELQ subgraph
(5.5× smaller). GraftNet even sparser (~84 entities). Gold answer MID likely absent from subgraph.
This is a PPR scoring failure (NSM relation weights fail on perturbed questions), not a decoder issue.
Do NOT write "NSM/GraftNet recover secondary perturbations" — they only recover vs ELQ (<1%) which
had complete EL failure; the absolute EM (15-21%) is still severely degraded.
"""


# ── Section: All-Results Snapshot JSON ───────────────────────────────────────

def section_snapshot() -> str:
    """Embed all_results_snapshot.json with interpretation."""
    fpath = RESULTS / "all_results_snapshot.json"
    r = _jload(fpath)
    if not r:
        return "## All-Results Snapshot — MISSING\n"

    generated = r.get("generated", "unknown")

    # ── Full pipeline summary table ──
    # Drop = em_pert - em_clean (negative = performance dropped under perturbation)
    fp = r.get("full_pipeline", {})
    fp_rows = []
    for key, v in sorted(fp.items()):
        n = v.get("n", "?")
        em_c = v.get("em_clean", 0)
        em_p = v.get("em_pert", 0)
        drop = em_p - em_c   # negative means degradation
        fp_rows.append(f"| {key} | {n} | {em_c:.1f}% | {em_p:.1f}% | {drop:+.1f}pp |")
    fp_table = "\n".join(fp_rows)

    # ── Fast path summary ──
    # delta = em_pert - em_clean (negative = decoder degradation under perturbation)
    fast = r.get("fast_path", {})
    fast_rows = []
    for key, v in sorted(fast.items()):
        em_c = v.get("em_clean", 0)
        em_p = v.get("em_pert", 0)
        delta = em_p - em_c   # negative = drop
        fast_rows.append(f"| {key} | {v.get('n','?')} | {em_c:.1f}% | {em_p:.1f}% | {delta:+.1f}pp |")
    fast_table = "\n".join(fast_rows)

    # ── NSM isolation summary ──
    # delta_hit1 stored as clean-pert (positive = drop in performance); display as negative
    nsm = r.get("nsm_isolation", {})
    nsm_rows = []
    for key, v in sorted(nsm.items()):
        h1c  = v.get("hit1_clean", 0)
        h1p  = v.get("hit1_pert", 0)
        drop = v.get("delta_hit1", 0)   # = clean - pert, positive means degradation
        nsm_rows.append(f"| {key} | {v.get('n','?')} | {h1c:.2f}% | {h1p:.2f}% | {-drop:+.2f}pp |")
    nsm_table = "\n".join(nsm_rows)

    # ── Cross-attack summary ──
    # gain = EM_transfer - pipeline_EM (positive = subgraph swap helped)
    ca = r.get("cross_attack", {})
    # Supplement stored pipeline EMs with known values (WebQSP not stored in cell JSONs)
    known_pipeline = {
        "cwq_3a": 0.68, "cwq_3b": 20.31, "cwq_3c": 0.40,
        "webqsp_3a": 0.49, "webqsp_3b": 50.95, "webqsp_3c": 0.43,
    }
    ca_rows = []
    for key, v in sorted(ca.items()):
        em_c = v.get("EM_clean_pct", 0)
        em_t = v.get("EM_transfer_pct", 0)
        pipe = v.get("baseline_pipeline_EM") or known_pipeline.get(key)
        pipe_s = f"{pipe:.2f}%" if pipe is not None else "—"
        gain_s = f"{em_t - pipe:+.1f}pp" if pipe is not None else "—"
        ca_rows.append(
            f"| {key} | {v.get('label','')} | {v.get('n','?')} "
            f"| {em_c:.1f}% | {em_t:.1f}% | {pipe_s} | {gain_s} |"
        )
    ca_table = "\n".join(ca_rows)

    # ── PPR mass summary ──
    ppr = r.get("ppr_mass", {})
    ppr_b = ppr.get("part_b", {})
    ppr_str = "MISSING"
    if ppr_b:
        ppr_str = (
            f"n={ppr_b.get('n_processed','?')}, "
            f"clean_median={ppr_b.get('clean_median_ppr',0):.4f}, "
            f"pert_median={ppr_b.get('pert_median_ppr',0):.4f}, "
            f"ratio={ppr_b.get('collapse_ratio',0):.3f}×"
        )

    return f"""## All-Results Snapshot (`all_results_snapshot.json`)

Generated: {generated}

⚠️ **WARNING — snapshot uses `graftnetv3_gem` (OLD, bugged build from May 7).**
Canonical GraftNetV3-GEM results are in `full_graftnetv3_fixed` (May 12 bug-fixed rebuild).
Key differences: CWQ p7 gem=9.86% vs **fixed=13.71%**; WebQSP p4 gem=31.18% vs **fixed=36.55%**.
Use the "Full-Pipeline EM" section below (which reads `full_graftnetv3_fixed`) for paper numbers.
The snapshot is preserved for audit trail only.

### Full-Pipeline EM from snapshot (⚠️ GraftNetV3 values are from bugged _gem build)

| Cell | n | EM_clean | EM_pert | Drop (clean−pert) |
|------|---|----------|---------|-------------------|
{fp_table}

**Interpretation:**
- All ELQ full-pipeline cells for non-RS perturbations collapse to <1% — ELQ cannot entity-link on
  perturbed questions (entity alias substitution, voice flip, composition restructuring, entity insertion).
- RS (p7) is the only perturbation where ELQ partially survives: CWQ=20.3%, WebQSP=51.0%.
  RS preserves surface entity names so ELQ's NER partially works.
- GEM seeds (oracle entity linking) dramatically recover CR: CWQ NSM-GEM=30.3%, **GraftNet-GEM(fixed)=29.82%**.
  (Snapshot has GraftNet-GEM=29.79% from bugged build — close for p4, but p7 and WebQSP differ.)
- NSM-ELQ and GraftNet-ELQ on CR: CWQ≈15% — partial EL recovery compared to NSM-GEM=30.3%.

### Fast-Path EM from snapshot (Δ = EM_fast − EM_clean; negative = decoder degradation)

| Cell | n | EM_clean | EM_fast | Δ_decoder |
|------|---|----------|---------|-----------|
{fast_table}

**Interpretation:**
- CWQ: decoder sensitivity <1.3pp for all perturbation types tested.
- WebQSP CR: EM_fast=69.74% vs clean=74.31% → **−4.57pp** decoder drop.
  Not negligible — CR restructuring confuses GNN attention even with clean subgraph.
- WebQSP RS: −3.72pp. WebQSP S1: −1.83pp. WebQSP S4: −2.50pp.
- CWQ S1(p2)=−0.23pp, S4(p6)=−1.25pp — near-zero decoder sensitivity for secondary types on CWQ.

### NSM Decoder Isolation from snapshot (Δ = EM_pert − EM_clean; negative = degradation)

| Cell | n | Hit@1_clean | Hit@1_pert | Δ |
|------|---|------------|-----------|---|
{nsm_table}

**Interpretation:**
- CWQ CR: Δ=−0.85pp, CWQ RS: Δ=−0.99pp. **Both robust** (Δ<1pp) — confirms claim B6 for CWQ. ✓
- WebQSP CR: Δ=**−10.86pp** — NSM decoder sensitive to CR on WebQSP. ✗ Caveat required in paper.
- WebQSP RS: Δ=**−4.88pp** — moderate sensitivity, not negligible.

### Cross-Attack Transfer from snapshot (Gain = EM_transfer − pipeline_EM; positive = recovery)

| Cell | Label | n | EM_clean | EM_transfer | Pipeline | Gain |
|------|-------|---|----------|------------|---------|------|
{ca_table}

**Interpretation:**
- **3A CR-Q+RS-SG**: Pipeline→Transfer: CWQ 0.68%→20.1% (+19.5pp), WebQSP 0.49%→49.7% (+49.2pp).
  Swapping in RS subgraph recovers most of the CR drop. WebQSP near-complete recovery (49.7% vs clean 74.3%)
  → CR failure on WebQSP is almost entirely EL failure. CWQ partial recovery (20.1% vs clean 52.9%)
  → both EL failure AND decoder sensitivity (~4.6pp from fast-path) contribute to CWQ CR collapse.
- **3B RS-Q+CR-SG**: Transfer: CWQ 20.31%→11.7% (−8.6pp), WebQSP 50.95%→33.4% (−17.5pp).
  Bad CR subgraph actively degrades RS — confirms subgraph quality is load-bearing even for robust RS decoder.
- **3C S3-Q+RS-SG**: Pipeline→Transfer: CWQ 0.40%→19.8% (+19.4pp), WebQSP 0.43%→50.8% (+50.3pp).
  RS subgraph suppresses S3 distractor entities → near-full WebQSP recovery.
  Confirms S3 failure is EL-driven: distractor MIDs get linked, PPR mass goes to wrong sub-tree.

### PPR Mass Part B (ELQ CR, n=200 sample)

{ppr_str}

**Interpretation:**
Collapse ratio=0.987 — PPR score distribution is statistically identical for clean vs CR perturbed questions.
This is a **null result**: ELQ entities are chosen from the question's named entities; CR restructures
syntax/composition but typically leaves surface entity mentions intact. So ELQ seeds the same entities
→ same PPR → same subgraph. The GNN failure (EM 52.9%→0.68%) is thus entirely in GNN cross-attention
(how the decoder attends to question tokens to weight edge traversals), NOT in PPR routing.

**LaTeX implication:** Do NOT write "CR collapses PPR mass." Write:
"CR preserves PPR seeding (same surface entities, α_pert=74%) but destroys GNN attention alignment —
confirming the failure is in the instruction-following capacity of the GNN decoder, not graph retrieval."
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def build_sum_md() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = [
        f"# Experiment Results Summary\n\nLast updated: {now}\n",
        "---",
        section_snapshot(),
        "---",
        section_nsm_isolation(),
        "---",
        section_cross_attack(),
        "---",
        section_fast_path(),
        "---",
        section_ppr_mass(),
        "---",
        section_full_pipeline(),
        "---",
        section_s1s4_cwq(),
        "---",
        section_s1s4_webqsp(),
        "---",
        section_run_status(),
    ]
    return "\n\n".join(sections)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0,
                    help="If > 0, re-run every N seconds")
    args = ap.parse_args()

    while True:
        content = build_sum_md()
        SUM_MD.parent.mkdir(parents=True, exist_ok=True)
        SUM_MD.write_text(content)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Updated {SUM_MD}")
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()