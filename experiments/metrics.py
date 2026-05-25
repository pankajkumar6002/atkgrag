"""
metrics.py — Exact implementations of G1–G4, G8, G9 from the CascadeAttack spec.

All functions are pure Python / NumPy; no GPU needed.
Import this module from experiment scripts; do not run it directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# G1 — Δ_EL: Entity Linking Degradation
# ---------------------------------------------------------------------------

def delta_EL(mids_orig: set, mids_pert: set) -> float:
    """1 - token-level F1 over Freebase MID sets."""
    if not mids_orig and not mids_pert:
        return 0.0
    if not mids_orig or not mids_pert:
        return 1.0
    tp = len(mids_orig & mids_pert)
    p = tp / len(mids_pert)
    r = tp / len(mids_orig)
    return 1.0 - (2 * p * r / (p + r + 1e-10))


# ---------------------------------------------------------------------------
# G2 — Δ_SR: Subgraph Retrieval Degradation
# ---------------------------------------------------------------------------

def delta_SR(triples_orig: set, triples_pert: set) -> float:
    """1 - edge-level Jaccard similarity over retrieved triple sets.

    Each triple is a (subject_MID, predicate_MID, object_MID) tuple or string.
    """
    if not triples_orig and not triples_pert:
        return 0.0
    union = triples_orig | triples_pert
    return 1.0 - len(triples_orig & triples_pert) / len(union)


# ---------------------------------------------------------------------------
# G3 — Δ_EA: Evidence Aggregation Degradation
# ---------------------------------------------------------------------------

def delta_EA(context_orig: str, context_pert: str, device: str = "cuda") -> float:
    """1 - BERTScore F1 between original and perturbed evidence contexts.

    Uses microsoft/deberta-xlarge-mnli — FIXED, do not change.
    """
    from bert_score import score as bertscore_fn  # lazy import

    _, _, F1 = bertscore_fn(
        [context_pert],
        [context_orig],
        model_type="microsoft/deberta-xlarge-mnli",
        lang="en",
        rescale_with_baseline=False,
        device=device,
        verbose=False,
    )
    return float(1.0 - F1[0])


# ---------------------------------------------------------------------------
# G4 — Δ_AG: Answer Generation Degradation
# ---------------------------------------------------------------------------

def exact_match_kgqa(prediction: str, gold_answers: List[str]) -> float:
    """EM: 1.0 if any gold answer appears (case-insensitive) in prediction."""
    pred = prediction.strip().lower()
    return float(any(g.strip().lower() in pred for g in gold_answers))


def f1_kgqa(prediction: str, gold_answers: List[str]) -> float:
    """Token-level F1 between prediction and best-matching gold answer."""
    def tokenise(s: str) -> List[str]:
        return s.strip().lower().split()

    pred_tokens = set(tokenise(prediction))
    if not pred_tokens:
        return 0.0
    best = 0.0
    for g in gold_answers:
        gold_tokens = set(tokenise(g))
        if not gold_tokens:
            continue
        tp = len(pred_tokens & gold_tokens)
        if tp == 0:
            continue
        p = tp / len(pred_tokens)
        r = tp / len(gold_tokens)
        best = max(best, 2 * p * r / (p + r))
    return best


def delta_AG(em_clean: float, em_pert: float) -> float:
    """ΔAG = EM_clean - EM_pert (averaged over the test set before calling)."""
    return em_clean - em_pert


# ---------------------------------------------------------------------------
# G8 — γ_k: Cascade Influence Coefficient
# ---------------------------------------------------------------------------

DELTA_MIN = 0.05


def cascade_influence(
    delta_AG_list: List[float],
    delta_k_list: List[float],
    delta_min: float = DELTA_MIN,
) -> Union[float, str]:
    """γ_k = E[ΔAG] / E[Δk].  Returns 'N/A' if E[Δk] < delta_min."""
    mean_k = float(np.mean(delta_k_list))
    if mean_k < delta_min:
        return "N/A"
    return float(np.mean(delta_AG_list)) / mean_k


# ---------------------------------------------------------------------------
# G9 — Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Returns (mean, lower_ci, upper_ci) with (1-alpha) confidence."""
    arr = np.array(values)
    boots = [
        np.mean(np.random.choice(arr, len(arr), replace=True))
        for _ in range(n_boot)
    ]
    return (
        float(np.mean(arr)),
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_ci(values: List[float], n_boot: int = 1000) -> Dict[str, float]:
    """Convenience: returns dict with mean and 95% CI bounds."""
    mu, lo, hi = bootstrap_ci(values, n_boot=n_boot)
    return {"mean": mu, "ci_lo": lo, "ci_hi": hi}


def summarise_run(
    per_question: List[Dict[str, Any]],
    n_boot: int = 1000,
) -> Dict[str, Any]:
    """Aggregate per-question metric dicts into a summary with bootstrap CIs.

    Expected keys per question: em_clean, em_pert, f1_clean, f1_pert,
    delta_EL (optional), delta_SR (optional), delta_EA (optional).
    """
    em_clean = [r["em_clean"] for r in per_question]
    em_pert = [r["em_pert"] for r in per_question]
    f1_clean = [r.get("f1_clean", 0.0) for r in per_question]
    f1_pert = [r.get("f1_pert", 0.0) for r in per_question]

    summary: Dict[str, Any] = {
        "n": len(per_question),
        "EM_clean": mean_ci(em_clean, n_boot),
        "EM_pert": mean_ci(em_pert, n_boot),
        "delta_AG_EM": mean_ci(
            [a - b for a, b in zip(em_clean, em_pert)], n_boot
        ),
        "F1_clean": mean_ci(f1_clean, n_boot),
        "F1_pert": mean_ci(f1_pert, n_boot),
        "delta_AG_F1": mean_ci(
            [a - b for a, b in zip(f1_clean, f1_pert)], n_boot
        ),
    }

    for key in ("delta_EL", "delta_SR", "delta_EA"):
        vals = [r[key] for r in per_question if key in r]
        if vals:
            summary[key] = mean_ci(vals, n_boot)
            summary[f"gamma_{key.split('_')[1]}"] = cascade_influence(
                [a - b for a, b in zip(em_clean, em_pert)], vals
            )

    return summary


def save_results(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {path}")
