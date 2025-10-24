"""
RAG Retriever Module

Implements retrieval component for Retrieval-Augmented Generation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import torch


class RAGRetriever:
    """
    Retrieval module for RAG systems.
    Retrieves relevant documents based on query similarity.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        top_k: int = 5
    ):
        """
        Initialize RAG Retriever.
        
        Args:
            embedding_dim: Dimension of embeddings
            top_k: Number of top documents to retrieve
        """
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.document_embeddings = None
        self.documents = []
        
    def index_documents(self, documents: List[str], embeddings: Optional[np.ndarray] = None):
        """
        Index documents for retrieval.
        
        Args:
            documents: List of documents to index
            embeddings: Optional pre-computed embeddings
        """
        self.documents = documents
        
        if embeddings is not None:
            self.document_embeddings = embeddings
        else:
            # Generate random embeddings (in practice, use sentence transformers)
            self.document_embeddings = np.random.randn(len(documents), self.embedding_dim)
            # Normalize
            norms = np.linalg.norm(self.document_embeddings, axis=1, keepdims=True)
            self.document_embeddings = self.document_embeddings / (norms + 1e-8)
    
    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        k: Optional[int] = None
    ) -> List[Tuple[int, str, float]]:
        """
        Retrieve top-k relevant documents for a query.
        
        Args:
            query: Query string
            query_embedding: Optional pre-computed query embedding
            k: Number of documents to retrieve (defaults to self.top_k)
            
        Returns:
            List of tuples (doc_id, document, similarity_score)
        """
        if k is None:
            k = self.top_k
        
        if query_embedding is None:
            # Generate random query embedding (in practice, use sentence transformers)
            query_embedding = np.random.randn(self.embedding_dim)
        
        # Normalize query embedding
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(self.document_embeddings, query_embedding)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = [
            (int(idx), self.documents[idx], float(similarities[idx]))
            for idx in top_k_indices
        ]
        
        return results
    
    def retrieve_with_graph_context(
        self,
        query_embedding: np.ndarray,
        graph_node_embeddings: np.ndarray,
        relevant_node_indices: List[int],
        k: Optional[int] = None
    ) -> List[Tuple[int, str, float]]:
        """
        Retrieve documents using both query and graph context.
        
        Args:
            query_embedding: Query embedding
            graph_node_embeddings: Embeddings from GNN
            relevant_node_indices: Indices of relevant nodes from graph
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        if k is None:
            k = self.top_k
        
        # Combine query embedding with graph context
        if len(relevant_node_indices) > 0:
            graph_context = np.mean(graph_node_embeddings[relevant_node_indices], axis=0)
            # Weighted combination
            combined_embedding = 0.7 * query_embedding + 0.3 * graph_context
        else:
            combined_embedding = query_embedding
        
        # Normalize
        combined_embedding = combined_embedding / (np.linalg.norm(combined_embedding) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(self.document_embeddings, combined_embedding)
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = [
            (int(idx), self.documents[idx], float(similarities[idx]))
            for idx in top_k_indices
        ]
        
        return results
    
    def get_document_embedding(self, doc_id: int) -> np.ndarray:
        """Get embedding for a specific document."""
        return self.document_embeddings[doc_id]
    
    def get_num_documents(self) -> int:
        """Get total number of indexed documents."""
        return len(self.documents)
