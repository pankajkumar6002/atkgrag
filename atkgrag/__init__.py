"""
ATK-GRAG: Prompt-based Attacks on GNN-RAG Systems

This package provides tools for implementing and evaluating prompt-based attacks
on Graph Neural Network enhanced Retrieval-Augmented Generation (GNN-RAG) systems.
"""

__version__ = "0.1.0"

from .gnn.graph_builder import GraphBuilder
from .gnn.gnn_model import GNNReasoner
from .rag.retriever import RAGRetriever
from .attack.query_attack import QueryLevelAttack
from .attack.prompt_attack import PromptBasedAttack

__all__ = [
    "GraphBuilder",
    "GNNReasoner",
    "RAGRetriever",
    "QueryLevelAttack",
    "PromptBasedAttack",
]
