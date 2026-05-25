#!/usr/bin/env python3
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *  # noqa: F401,F403
visualise_ppr_mismatch.py — PPR topology mismatch figure (reviewer B9).

For 3–5 CWQ CR (p4) questions where:
  - em_clean = 1.0  (gold answer reachable with clean question)
  - em_pert  = 0.0  (gold answer unreachable with CR-perturbed question)
  - gold answer MID IS in the perturbed subgraph (answer present ≠ reachable)

Renders a 2-panel plot per example:
  Left:  PPR mass distribution on the 2-hop subgraph under the CLEAN question
  Right: PPR mass distribution on the same subgraph under the CR-PERTURBED question
  Gold answer node: outlined in red in both panels.

Output:
  - ACL_v2/fig/ppr_mismatch.pdf          (main figure for LaTeX)
  - tmlr/results/ppr_mismatch_data.json  (raw PPR scores, for TikZ template)

Usage:
  python visualise_ppr_mismatch.py
  python visualise_ppr_mismatch.py --n-examples 5 --out-dir ACL_v2/fig
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = REPO_ROOT
DATASET_DIR  = REPO_ROOT / "data" / "perturbations"
SUBGRAPH_DIR = DATASET_DIR / "subgraph_var/subgraphs/cwq_p4_graftnetv3"
PERT_FILE    = DATASET_DIR / "cwq_p4.json"
PER_Q_RESULTS = REPO_ROOT / "tmlr/results/e1_cwq_gnn_rag/full_elq/p4_per_question.json"
RESULTS_DIR  = REPO_ROOT / "tmlr/results"
GNN_RAG_GNN  = REPO_ROOT / "tmlr/Repo/GNN-RAG/gnn"

# Entity name lookup (MID → label) via GNN-RAG entities.txt
ENTITIES_TXT = REPO_ROOT / "tmlr/Repo/GNN-RAG/data/CWQ/entities.txt"

# Build_graftnetv3 imports
BUILD_SCRIPT = REPO_ROOT / "tmlr/scripts/build_graftnetv3.py"

sys.path.insert(0, str(REPO_ROOT / "tmlr/scripts"))


# ── Import PPR functions from build_graftnetv3 ───────────────────────────────
def _import_ppr():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_graftnetv3", BUILD_SCRIPT)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Sentence encoder not needed — PPR uses word embeddings (vocab+matrix) directly.


# ── Entity MID → human label ─────────────────────────────────────────────────
def load_entity_names(path: Path, limit: int = 0) -> Dict[str, str]:
    """Load MID → row-index from entities.txt. Returns {} if file too large/missing."""
    if not path.exists():
        return {}
    names: Dict[str, str] = {}
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                mid = line.strip()
                if mid:
                    names[mid] = mid  # fallback: use MID itself as label
    except Exception:
        pass
    return names


def mid_label(mid: str, names: Dict[str, str], max_len: int = 12) -> str:
    """Return a short display label for a MID."""
    label = names.get(mid, mid)
    # Strip namespace prefix
    if "." in label:
        label = label.split(".")[-1].replace("_", " ")
    if len(label) > max_len:
        label = label[:max_len] + "…"
    return label


# ── Select representative examples ───────────────────────────────────────────
def select_examples(
    per_q_file: Path,
    pert_file: Path,
    subgraph_dir: Path,
    n: int = 5,
    max_triples: int = 80,
    min_triples: int = 15,
) -> List[Dict]:
    """Find questions where em_clean=1, em_pert=0, gold MID in subgraph."""
    with open(per_q_file) as f:
        per_q: List[Dict] = json.load(f)

    with open(pert_file) as f:
        pert_data: List[Dict] = json.load(f)
    pert_map = {r["id"]: r for r in pert_data}

    candidates = []
    for item in per_q:
        if item.get("em_clean") != 1.0 or item.get("em_pert") != 0.0:
            continue
        qid = item["id"]
        sg_path = subgraph_dir / f"{qid}.json"
        if not sg_path.exists():
            continue
        with open(sg_path) as f:
            sg = json.load(f)

        triples = sg.get("triples", [])
        n_triples = len(triples)
        if n_triples < min_triples or n_triples > max_triples:
            continue

        pert_rec = pert_map.get(qid)
        if not pert_rec:
            continue

        gold_mids = [a["kb_id"] if isinstance(a, dict) else a
                     for a in pert_rec.get("answers", [])]
        entities_in_sg = set(sg.get("entities", []))
        if not any(g in entities_in_sg for g in gold_mids):
            continue  # gold not present → not the "present but unreachable" case

        candidates.append({
            "id": qid,
            "orig_q": pert_rec["original_question"],
            "pert_q": pert_rec["perturbed_question"],
            "gold_mids": gold_mids,
            "seed_mids": sg.get("seed_mids", []),
            "triples": triples,
            "entities": list(entities_in_sg),
            "gold_relation_map": pert_rec.get("gold_relation_map", {}),
            "n_triples": n_triples,
        })
        if len(candidates) >= n * 3:  # over-select, then prune to diverse set
            break

    # Pick examples with varied n_triples for visual diversity
    candidates.sort(key=lambda x: x["n_triples"])
    step = max(1, len(candidates) // n)
    selected = candidates[::step][:n]
    print(f"Selected {len(selected)} examples (from {len(candidates)} candidates)")
    return selected


# ── Run PPR for one example ───────────────────────────────────────────────────
def run_ppr_both(
    example: Dict,
    bv3_mod,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (ppr_clean, ppr_pert) as MID→score dicts."""
    triples = example["triples"]
    seed_mids = example["seed_mids"]
    gold_rel_map = example["gold_relation_map"]

    def _ppr(question: str) -> Dict[str, float]:
        rel_scores = bv3_mod.compute_relation_scores(question, gold_rel_map,
                                                     bv3_mod._g_vocab,
                                                     bv3_mod._g_emb)
        entity_map, adj_mat = bv3_mod._graftnetv3_read_facts(triples, rel_scores)
        if adj_mat is None or not entity_map:
            return {}
        n = adj_mat.shape[0]
        seed_idx = [entity_map[m] for m in seed_mids if m in entity_map]
        seed_vec = np.zeros((n, 1))
        for idx in seed_idx:
            seed_vec[idx] = 1.0 / max(len(seed_idx), 1)
        ppr_vec = bv3_mod._graftnet_ppr(seed_vec, adj_mat)
        inv = {v: k for k, v in entity_map.items()}
        return {inv[i]: float(ppr_vec[i]) for i in range(n) if i in inv}

    ppr_clean = _ppr(example["orig_q"])
    ppr_pert  = _ppr(example["pert_q"])
    return ppr_clean, ppr_pert


# ── NetworkX visualisation ───────────────────────────────────────────────────

def plot_examples_networkx(
    examples: List[Dict],
    ppr_pairs: List[Tuple[Dict, Dict]],
    out_path: Path,
    entity_names: Dict[str, str],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    n_ex = len(examples)
    fig, axes = plt.subplots(n_ex, 2, figsize=(14, 4 * n_ex))
    if n_ex == 1:
        axes = [axes]

    cmap = plt.cm.YlOrRd

    for row, (ex, (ppr_clean, ppr_pert)) in enumerate(zip(examples, ppr_pairs)):
        gold_set = set(ex["gold_mids"])
        seed_set = set(ex["seed_mids"])

        # Build graph from triples (use top-30 entities by max PPR)
        all_ppr = {m: max(ppr_clean.get(m, 0), ppr_pert.get(m, 0))
                   for m in ex["entities"]}
        top_mids = sorted(all_ppr, key=all_ppr.get, reverse=True)[:28]
        top_set = set(top_mids) | gold_set | seed_set

        G = nx.DiGraph()
        for s, r, o in ex["triples"]:
            if s in top_set and o in top_set:
                G.add_edge(s, o)
        for m in top_set:
            G.add_node(m)

        # Stable layout
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42)

        def _draw_panel(ax, ppr: Dict[str, float], title: str):
            scores = np.array([ppr.get(m, 0.0) for m in G.nodes()])
            max_s = scores.max() if scores.max() > 0 else 1.0
            scores_norm = scores / max_s

            node_colors = [cmap(s * 0.85 + 0.05) for s in scores_norm]
            node_sizes  = [600 + 1400 * s for s in scores_norm]
            edge_colors = [mpatches.FancyArrowPatch]

            # Draw edges
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25,
                                   arrowsize=8, edge_color="#888888",
                                   connectionstyle="arc3,rad=0.1")

            # Draw regular nodes
            regular = [m for m in G.nodes() if m not in gold_set and m not in seed_set]
            gold_in = [m for m in G.nodes() if m in gold_set]
            seed_in = [m for m in G.nodes() if m in seed_set and m not in gold_set]

            nx.draw_networkx_nodes(G, pos, nodelist=regular,
                                   node_color=[node_colors[list(G.nodes()).index(m)] for m in regular],
                                   node_size=[node_sizes[list(G.nodes()).index(m)] for m in regular],
                                   ax=ax, alpha=0.9)
            # Gold answer nodes: thick red outline
            if gold_in:
                nx.draw_networkx_nodes(G, pos, nodelist=gold_in,
                                       node_color=[node_colors[list(G.nodes()).index(m)] for m in gold_in],
                                       node_size=[node_sizes[list(G.nodes()).index(m)] + 200 for m in gold_in],
                                       edgecolors="red", linewidths=3.0,
                                       ax=ax, alpha=0.95)
            # Seed nodes: blue outline
            if seed_in:
                nx.draw_networkx_nodes(G, pos, nodelist=seed_in,
                                       node_color=[node_colors[list(G.nodes()).index(m)] for m in seed_in],
                                       node_size=[node_sizes[list(G.nodes()).index(m)] + 200 for m in seed_in],
                                       edgecolors="blue", linewidths=2.5,
                                       ax=ax)

            # Labels for gold and seed only (to keep it readable)
            labels = {}
            for m in G.nodes():
                if m in gold_set or m in seed_set:
                    labels[m] = mid_label(m, entity_names, max_len=10)
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7,
                                    font_weight="bold")

            ax.set_title(title, fontsize=9, pad=4)
            ax.axis("off")

            # Colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.6, label="PPR mass (normalised)")

        orig_short = ex["orig_q"][:55] + ("…" if len(ex["orig_q"]) > 55 else "")
        pert_short = ex["pert_q"][:55] + ("…" if len(ex["pert_q"]) > 55 else "")

        _draw_panel(axes[row][0], ppr_clean,
                    f'Clean: "{orig_short}"')
        _draw_panel(axes[row][1], ppr_pert,
                    f'CR-perturbed: "{pert_short}"')

        # Row annotation
        axes[row][0].set_ylabel(f"Ex {row+1}", fontsize=9, rotation=0,
                                 labelpad=30, va="center")

    fig.suptitle(
        "PPR mass under clean vs. CR-perturbed question\n"
        "(red outline = gold answer; blue = seed entity; colour intensity = PPR mass)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved networkx figure: {out_path}")
    plt.close(fig)


# ── Export data for TikZ ─────────────────────────────────────────────────────

def export_for_tikz(
    examples: List[Dict],
    ppr_pairs: List[Tuple[Dict, Dict]],
    out_json: Path,
) -> None:
    """Export PPR scores and graph structure for TikZ figure generation."""
    export = []
    for ex, (ppr_clean, ppr_pert) in zip(examples, ppr_pairs):
        gold_set = set(ex["gold_mids"])
        seed_set = set(ex["seed_mids"])
        all_mids = set(ex["entities"])
        top_mids = sorted(all_mids,
                          key=lambda m: max(ppr_clean.get(m, 0), ppr_pert.get(m, 0)),
                          reverse=True)[:20]
        top_set = set(top_mids) | gold_set | seed_set

        edges = list({(s, o) for s, r, o in ex["triples"]
                      if s in top_set and o in top_set})[:40]

        nodes = []
        for m in top_set:
            nodes.append({
                "mid": m,
                "is_gold": m in gold_set,
                "is_seed": m in seed_set,
                "ppr_clean": round(ppr_clean.get(m, 0.0), 5),
                "ppr_pert":  round(ppr_pert.get(m, 0.0), 5),
            })

        export.append({
            "id": ex["id"],
            "orig_q": ex["orig_q"],
            "pert_q": ex["pert_q"],
            "nodes": nodes,
            "edges": [list(e) for e in edges],
        })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Saved TikZ data: {out_json}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--max-triples", type=int, default=70)
    parser.add_argument("--min-triples", type=int, default=15)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "ACL_v2/fig")
    parser.add_argument("--no-networkx", action="store_true")
    args = parser.parse_args()

    print("Loading build_graftnetv3 PPR module…")
    bv3 = _import_ppr()

    # Pre-load vocab + word embeddings (used by compute_relation_scores)
    try:
        bv3._g_vocab, bv3._g_emb = bv3.load_word_emb("cwq")
        print(f"Loaded word embeddings: vocab={len(bv3._g_vocab)}")
    except Exception as e:
        print(f"[WARN] Word embeddings not loaded ({e}) — cosine scores will be uniform")
        bv3._g_vocab, bv3._g_emb = {}, None

    print("Selecting examples…")
    examples = select_examples(
        PER_Q_RESULTS, PERT_FILE, SUBGRAPH_DIR,
        n=args.n_examples,
        max_triples=args.max_triples,
        min_triples=args.min_triples,
    )
    if not examples:
        print("No suitable examples found. Try relaxing --max-triples.")
        return

    print("Running PPR (clean + perturbed) for each example…")
    ppr_pairs = []
    for i, ex in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] {ex['id'][:60]}")
        ppr_clean, ppr_pert = run_ppr_both(ex, bv3)
        ppr_pairs.append((ppr_clean, ppr_pert))
        # Print gold PPR mass for sanity
        for g in ex["gold_mids"]:
            c = ppr_clean.get(g, 0.0)
            p = ppr_pert.get(g, 0.0)
            print(f"    gold {g}: clean_mass={c:.4f}  pert_mass={p:.4f}")

    # Export data for TikZ
    tikz_data_path = RESULTS_DIR / "ppr_mismatch_data.json"
    export_for_tikz(examples, ppr_pairs, tikz_data_path)

    # NetworkX figure
    if not args.no_networkx:
        entity_names = load_entity_names(ENTITIES_TXT, limit=500_000)
        nx_out = args.out_dir / "ppr_mismatch_networkx.pdf"
        plot_examples_networkx(examples, ppr_pairs, nx_out, entity_names)

    print("\nDone. Next step: run generate_tikz_ppr.py to generate fig6.tex")


if __name__ == "__main__":
    main()