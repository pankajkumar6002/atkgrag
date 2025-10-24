"""
Utility functions for the attack framework.
"""

import numpy as np
import torch
from typing import List, Dict, Any


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def evaluate_query_perturbation(
    original_query: str,
    perturbed_query: str
) -> Dict[str, Any]:
    """
    Evaluate the perturbation applied to a query.
    
    Args:
        original_query: Original query string
        perturbed_query: Perturbed query string
        
    Returns:
        Dictionary with perturbation metrics
    """
    orig_tokens = original_query.split()
    pert_tokens = perturbed_query.split()
    
    # Token-level changes
    tokens_added = len(pert_tokens) - len(orig_tokens)
    
    # Character-level changes
    char_changes = abs(len(perturbed_query) - len(original_query))
    
    # Simple edit distance (approximation)
    common_tokens = set(orig_tokens) & set(pert_tokens)
    token_overlap = len(common_tokens) / max(len(orig_tokens), 1)
    
    return {
        "tokens_added": tokens_added,
        "char_changes": char_changes,
        "token_overlap": token_overlap,
        "perturbation_ratio": tokens_added / max(len(orig_tokens), 1)
    }


def batch_process_queries(
    queries: List[str],
    attack_fn: callable,
    **attack_kwargs
) -> List[str]:
    """
    Process multiple queries with an attack function.
    
    Args:
        queries: List of queries
        attack_fn: Attack function to apply
        **attack_kwargs: Additional arguments for attack function
        
    Returns:
        List of adversarial queries
    """
    adversarial_queries = []
    
    for query in queries:
        adv_query = attack_fn(query, **attack_kwargs)
        adversarial_queries.append(adv_query)
    
    return adversarial_queries


def compute_attack_metrics(
    original_outputs: List[Any],
    adversarial_outputs: List[Any],
    metric_type: str = "accuracy"
) -> Dict[str, float]:
    """
    Compute attack success metrics.
    
    Args:
        original_outputs: Outputs from original queries
        adversarial_outputs: Outputs from adversarial queries
        metric_type: Type of metric to compute
        
    Returns:
        Dictionary with metrics
    """
    if len(original_outputs) != len(adversarial_outputs):
        raise ValueError("Output lists must have same length")
    
    if metric_type == "accuracy":
        # Count how many outputs changed
        changes = sum(1 for o, a in zip(original_outputs, adversarial_outputs) if o != a)
        success_rate = changes / len(original_outputs)
    else:
        success_rate = 0.0
    
    return {
        "attack_success_rate": success_rate,
        "total_samples": len(original_outputs),
        "successful_attacks": sum(1 for o, a in zip(original_outputs, adversarial_outputs) if o != a)
    }


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit length.
    
    Args:
        embeddings: Input embeddings
        
    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def convert_to_torch(data: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Convert numpy array to torch tensor.
    
    Args:
        data: Numpy array
        device: Device to place tensor on
        
    Returns:
        PyTorch tensor
    """
    return torch.from_numpy(data).float().to(device)


def create_edge_index_from_adjacency(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Create edge index from adjacency matrix.
    
    Args:
        adj_matrix: Adjacency matrix
        
    Returns:
        Edge index in COO format [2, num_edges]
    """
    edges = np.argwhere(adj_matrix > 0)
    return edges.T
