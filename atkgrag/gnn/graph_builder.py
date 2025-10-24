"""
Graph Builder Module

Constructs knowledge graphs from text documents for GNN reasoning.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class GraphBuilder:
    """
    Builds knowledge graphs from text documents by extracting entities and relationships.
    """
    
    def __init__(self, embedding_dim: int = 768):
        """
        Initialize GraphBuilder.
        
        Args:
            embedding_dim: Dimension of node embeddings
        """
        self.embedding_dim = embedding_dim
        self.node_to_id = {}
        self.id_to_node = {}
        self.edges = []
        self.node_features = []
        
    def build_graph_from_documents(
        self, 
        documents: List[str],
        entity_extractor: Optional[callable] = None
    ) -> nx.Graph:
        """
        Build a knowledge graph from a list of documents.
        
        Args:
            documents: List of text documents
            entity_extractor: Optional custom entity extraction function
            
        Returns:
            NetworkX graph object
        """
        graph = nx.Graph()
        
        # Simple entity extraction based on noun phrases
        # In a real implementation, this would use NER models
        for doc_id, doc in enumerate(documents):
            entities = self._extract_entities(doc)
            
            # Add nodes for entities
            for entity in entities:
                if entity not in self.node_to_id:
                    node_id = len(self.node_to_id)
                    self.node_to_id[entity] = node_id
                    self.id_to_node[node_id] = entity
                    # Random embedding for now - in practice would use sentence transformers
                    self.node_features.append(np.random.randn(self.embedding_dim))
                    graph.add_node(node_id, label=entity, doc_id=doc_id)
            
            # Add edges between co-occurring entities
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    id1 = self.node_to_id[e1]
                    id2 = self.node_to_id[e2]
                    if not graph.has_edge(id1, id2):
                        graph.add_edge(id1, id2, weight=1.0)
                        self.edges.append((id1, id2))
                    else:
                        # Increase edge weight for repeated co-occurrence
                        graph[id1][id2]['weight'] += 1.0
        
        return graph
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Simple entity extraction from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        # Simple word-based extraction
        # In practice, would use NER or more sophisticated methods
        words = text.lower().split()
        # Filter out common words and keep potential entities (length > 3)
        entities = [w for w in words if len(w) > 3 and w.isalpha()]
        return list(set(entities))[:10]  # Limit to top 10 per document
    
    def get_node_features(self) -> np.ndarray:
        """Get node feature matrix."""
        return np.array(self.node_features)
    
    def get_edge_index(self) -> np.ndarray:
        """Get edge index in COO format for PyG."""
        if not self.edges:
            return np.array([[], []])
        return np.array(self.edges).T
    
    def add_query_node(self, query: str, graph: nx.Graph) -> int:
        """
        Add a query node to the graph.
        
        Args:
            query: Query text
            graph: Existing graph
            
        Returns:
            Node ID of the query node
        """
        query_id = len(self.node_to_id)
        self.node_to_id[f"QUERY_{query_id}"] = query_id
        self.id_to_node[query_id] = f"QUERY_{query_id}"
        self.node_features.append(np.random.randn(self.embedding_dim))
        graph.add_node(query_id, label=f"QUERY_{query_id}", is_query=True)
        
        # Connect query to relevant nodes
        query_entities = self._extract_entities(query)
        for entity in query_entities:
            if entity in self.node_to_id:
                entity_id = self.node_to_id[entity]
                graph.add_edge(query_id, entity_id, weight=1.0)
        
        return query_id
