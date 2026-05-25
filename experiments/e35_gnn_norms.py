#!/usr/bin/env python3
"""
e35_gnn_norms.py — Experiment E35: GNN-RAG Weight Norms

Computes ||W_l||_op (largest singular value) for each GNN layer's weight
matrix and ||Ã_l||_2 (spectral norm of normalised adjacency) via power
iteration.  The 3-layer product ||Ã||_2^3 · Π||W_l||_op is the amplification
bound that must exceed 1.0 to support Remark A.1.

Paper target: 3-layer product ≈ 1.98.

Usage:
  python e35_gnn_norms.py \
    --ckpt external/GNN-RAG/data/pretrain/ReaRev_CWQ.ckpt \
    --data external/GNN-RAG/data/CWQ/test.json \
    --out  results/e35_gnn_norms.json

Can run immediately (no perturbation data or SPARQL needed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Checkpoint inspection
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # Handle both raw state-dict and wrapped checkpoints
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def extract_gnn_weights(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Extract weight matrices from GNN (GAT / ReaRev) layers.

    ReaRev uses keys like:
      reasoning_layer.0.W_q.weight, reasoning_layer.0.W_e.weight, ...
      node_encoder.layers.0.W.weight, ...
    We collect every 2-D weight tensor whose name suggests a linear layer in
    the message-passing stack.
    """
    gnn_weights: Dict[str, torch.Tensor] = {}
    keywords = ("reasoning_layer", "gnn", "node_encoder", "W_q", "W_e", "W_v", "W_k",
                 "linear", "fc", "attn")
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.dim() != 2:
            continue
        if any(kw in k.lower() for kw in keywords):
            gnn_weights[k] = v.float()
    return gnn_weights


def spectral_norm(W: torch.Tensor) -> float:
    """Largest singular value of W."""
    sv = torch.linalg.svdvals(W)
    return float(sv[0])


# ---------------------------------------------------------------------------
# Adjacency spectral norm via power iteration
# ---------------------------------------------------------------------------

def load_test_jsonl(path: Path, max_graphs: int = 200) -> List[Dict[str, Any]]:
    """Load JSONL records (GNN-RAG format with 'subgraph' field)."""
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= max_graphs:
                break
    return records


def build_adjacency_matrix(record: Dict[str, Any]) -> Optional[torch.Tensor]:
    """
    Build a normalised adjacency matrix Ã from the record's subgraph.

    GNN-RAG subgraph format:
      {"tuples": [[head_id, rel_id, tail_id], ...], "entities": [...]}

    We build A[i,j] = 1 if (i, rel, j) or (j, rel, i) in tuples, then
    compute Ã = D^{-1/2} A D^{-1/2}.
    """
    sg = record.get("subgraph")
    if not sg:
        return None
    tuples = sg.get("tuples") or sg.get("edges") or []
    entities = sg.get("entities") or record.get("entities") or []

    if not entities or not tuples:
        return None

    ent2idx = {e: i for i, e in enumerate(entities)}
    n = len(entities)
    if n < 2:
        return None

    A = torch.zeros(n, n)
    for triple in tuples:
        if len(triple) < 3:
            continue
        h, _, t = triple[0], triple[1], triple[2]
        if h in ent2idx and t in ent2idx:
            i, j = ent2idx[h], ent2idx[t]
            A[i, j] = 1.0
            A[j, i] = 1.0  # undirected for norm bound

    deg = A.sum(dim=1).clamp(min=1.0)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat


def spectral_norm_power_iter(A: torch.Tensor, n_iter: int = 30) -> float:
    """||A||_2 via power iteration."""
    if A.shape[0] == 0:
        return 0.0
    v = torch.randn(A.shape[1])
    v = v / v.norm()
    for _ in range(n_iter):
        v = A @ v
        norm = v.norm()
        if norm < 1e-10:
            return 0.0
        v = v / norm
    return float((A @ v).norm())


def compute_adj_norms(records: List[Dict[str, Any]], max_graphs: int = 100) -> List[float]:
    norms = []
    for rec in records[:max_graphs]:
        A_hat = build_adjacency_matrix(rec)
        if A_hat is not None and A_hat.shape[0] > 1:
            norms.append(spectral_norm_power_iter(A_hat))
    return norms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E35: GNN-RAG weight norms")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=Path("external/GNN-RAG/data/pretrain/ReaRev_CWQ.ckpt"),
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path("external/GNN-RAG/data/CWQ/test.json"),
        help="GNN-RAG JSONL test file for adjacency norm estimation",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/e35_gnn_norms.json"),
    )
    p.add_argument("--max-graphs", type=int, default=200,
                   help="Number of subgraphs to use for adjacency norm estimation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Weight norms ---
    print(f"Loading checkpoint: {args.ckpt}")
    state_dict = load_checkpoint(args.ckpt)
    gnn_weights = extract_gnn_weights(state_dict)

    print(f"\nFound {len(gnn_weights)} GNN weight matrices:")
    weight_norms: Dict[str, float] = {}
    for name, W in sorted(gnn_weights.items()):
        sn = spectral_norm(W)
        weight_norms[name] = sn
        print(f"  {name:<60s}  shape={tuple(W.shape)}  ||W||_op={sn:.4f}")

    # Group by layer index if key contains digits
    layer_groups: Dict[str, List[float]] = {}
    for name, sn in weight_norms.items():
        # Extract layer number from key (e.g. reasoning_layer.0.W_q.weight → layer 0)
        parts = name.split(".")
        layer_key = "unknown"
        for part in parts:
            if part.isdigit():
                layer_key = f"layer_{part}"
                break
        layer_groups.setdefault(layer_key, []).append(sn)

    layer_max_norms = {k: max(v) for k, v in layer_groups.items()}
    print("\nMax ||W||_op per layer group:")
    product_W = 1.0
    for k in sorted(layer_max_norms):
        print(f"  {k}: {layer_max_norms[k]:.4f}")
        product_W *= layer_max_norms[k]
    print(f"  Product across layers: {product_W:.4f}")

    # --- Adjacency norms ---
    print(f"\nLoading {args.max_graphs} subgraphs from {args.data} …")
    records = load_test_jsonl(args.data, max_graphs=args.max_graphs)
    adj_norms = compute_adj_norms(records, max_graphs=args.max_graphs)

    if adj_norms:
        mean_adj = float(np.mean(adj_norms))
        median_adj = float(np.median(adj_norms))
        max_adj = float(np.max(adj_norms))
        print(f"\n||Ã||_2 over {len(adj_norms)} subgraphs:")
        print(f"  mean={mean_adj:.4f}  median={median_adj:.4f}  max={max_adj:.4f}")
        # 3-layer amplification product using mean adjacency norm
        amp_product = (mean_adj ** 3) * product_W
        print(f"\n3-layer amplification product (mean Ã)^3 × Π||W||_op = {amp_product:.4f}")
        print(f"  (Paper target ≈ 1.98; amplification condition holds if > 1.0)")
    else:
        mean_adj = median_adj = max_adj = amp_product = None
        print("  No valid adjacency matrices found in test data.")

    results = {
        "weight_norms": weight_norms,
        "layer_max_norms": layer_max_norms,
        "product_W": product_W,
        "adj_norms": {
            "n_graphs": len(adj_norms),
            "mean": mean_adj,
            "median": median_adj,
            "max": max_adj,
        },
        "amplification_product_3layer": amp_product,
        "checkpoint": str(args.ckpt),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
