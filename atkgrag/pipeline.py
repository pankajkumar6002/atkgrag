"""
GNN-RAG Attack Pipeline

Integrates all components for end-to-end attack on GNN-RAG systems.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import networkx as nx

from .gnn.graph_builder import GraphBuilder
from .gnn.gnn_model import GNNReasoner
from .rag.retriever import RAGRetriever
from .attack.query_attack import QueryLevelAttack
from .attack.prompt_attack import PromptBasedAttack
from .utils.helpers import convert_to_torch, compute_attack_metrics


class GNNRAGPipeline:
    """
    Complete pipeline for GNN-RAG with attack capabilities.
    
    This class integrates the graph construction, GNN reasoning, RAG retrieval,
    and adversarial attack components.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_gnn_layers: int = 2,
        top_k_retrieval: int = 5,
        device: str = "cpu"
    ):
        """
        Initialize GNN-RAG Pipeline.
        
        Args:
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension for GNN
            output_dim: Output dimension for GNN
            num_gnn_layers: Number of GNN layers
            top_k_retrieval: Number of documents to retrieve
            device: Device to run on (cpu/cuda)
        """
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Initialize components
        self.graph_builder = GraphBuilder(embedding_dim)
        self.gnn_model = GNNReasoner(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_gnn_layers
        ).to(device)
        self.retriever = RAGRetriever(embedding_dim, top_k_retrieval)
        
        # Attack components
        self.query_attacker = QueryLevelAttack()
        self.prompt_attacker = PromptBasedAttack(self.gnn_model)
        
        # State
        self.graph = None
        self.documents = []
        
    def setup(self, documents: List[str]):
        """
        Setup the pipeline with documents.
        
        Args:
            documents: List of text documents
        """
        self.documents = documents
        
        # Build knowledge graph
        self.graph = self.graph_builder.build_graph_from_documents(documents)
        
        # Index documents in retriever
        self.retriever.index_documents(documents)
        
        print(f"Setup complete: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
    
    def process_query(
        self,
        query: str,
        use_gnn_reasoning: bool = True
    ) -> Dict:
        """
        Process a query through the pipeline.
        
        Args:
            query: User query
            use_gnn_reasoning: Whether to use GNN reasoning
            
        Returns:
            Dictionary with results
        """
        # Add query to graph
        query_id = self.graph_builder.add_query_node(query, self.graph)
        
        # Get node features and edge index
        node_features = self.graph_builder.get_node_features()
        edge_index = self.graph_builder.get_edge_index()
        
        # Convert to torch tensors
        node_features_t = convert_to_torch(node_features, self.device)
        edge_index_t = convert_to_torch(edge_index, self.device).long()
        
        # Create query mask
        query_mask = torch.zeros(node_features_t.size(0), dtype=torch.bool)
        query_mask[query_id] = True
        
        # GNN reasoning
        if use_gnn_reasoning and edge_index_t.size(1) > 0:
            node_embeddings, query_embedding = self.gnn_model(
                node_features_t,
                edge_index_t,
                query_mask
            )
            
            # Get relevant nodes (neighbors of query)
            neighbors = list(self.graph.neighbors(query_id))
            # Use original embedding dimension for retrieval compatibility
            query_emb_np = node_features[query_id]
            node_emb_np = node_features
        else:
            # No GNN reasoning
            query_emb_np = node_features[query_id]
            node_emb_np = node_features
            neighbors = []
        
        # Retrieve documents
        if len(neighbors) > 0:
            retrieved_docs = self.retriever.retrieve_with_graph_context(
                query_emb_np,
                node_emb_np,
                neighbors
            )
        else:
            retrieved_docs = self.retriever.retrieve(query, query_emb_np)
        
        return {
            "query": query,
            "query_id": query_id,
            "retrieved_documents": retrieved_docs,
            "graph_neighbors": neighbors,
            "num_nodes": len(self.graph.nodes),
            "num_edges": len(self.graph.edges)
        }
    
    def attack_query(
        self,
        query: str,
        attack_type: str = "semantic_shift",
        attack_budget: int = 5,
        attack_strength: float = 0.5
    ) -> Tuple[str, Dict]:
        """
        Attack a query and compare results.
        
        Args:
            query: Original query
            attack_type: Type of attack to perform
            attack_budget: Attack budget (number of tokens)
            attack_strength: Attack strength (0.0 to 1.0)
            
        Returns:
            Tuple of (adversarial_query, comparison_results)
        """
        # Configure attack
        self.query_attacker.attack_type = attack_type
        self.query_attacker.attack_budget = attack_budget
        self.prompt_attacker.attack_strength = attack_strength
        
        # Generate adversarial query
        if attack_type in ["semantic_shift", "node_injection", "edge_manipulation"]:
            adversarial_query = self.query_attacker.generate_adversarial_query(query)
        else:
            adversarial_query = self.prompt_attacker.adversarial_prompt_injection(
                query, attack_type
            )
        
        # Process both queries
        original_results = self.process_query(query)
        adversarial_results = self.process_query(adversarial_query)
        
        # Compare results
        comparison = {
            "original_query": query,
            "adversarial_query": adversarial_query,
            "attack_type": attack_type,
            "attack_budget": attack_budget,
            "original_top_docs": [doc[1][:50] + "..." for doc in original_results["retrieved_documents"][:3]],
            "adversarial_top_docs": [doc[1][:50] + "..." for doc in adversarial_results["retrieved_documents"][:3]],
            "retrieval_changed": original_results["retrieved_documents"] != adversarial_results["retrieved_documents"],
            "graph_impact": {
                "original_neighbors": len(original_results["graph_neighbors"]),
                "adversarial_neighbors": len(adversarial_results["graph_neighbors"])
            }
        }
        
        return adversarial_query, comparison
    
    def evaluate_attack_effectiveness(
        self,
        queries: List[str],
        attack_configs: List[Dict]
    ) -> Dict:
        """
        Evaluate attack effectiveness across multiple queries.
        
        Args:
            queries: List of queries to test
            attack_configs: List of attack configurations
            
        Returns:
            Evaluation results
        """
        results = []
        
        for query in queries:
            query_results = {"query": query, "attacks": []}
            
            for config in attack_configs:
                adv_query, comparison = self.attack_query(
                    query,
                    attack_type=config.get("attack_type", "semantic_shift"),
                    attack_budget=config.get("attack_budget", 5),
                    attack_strength=config.get("attack_strength", 0.5)
                )
                
                query_results["attacks"].append({
                    "config": config,
                    "comparison": comparison
                })
            
            results.append(query_results)
        
        # Aggregate statistics
        total_attacks = len(queries) * len(attack_configs)
        successful_attacks = sum(
            1 for qr in results 
            for a in qr["attacks"] 
            if a["comparison"]["retrieval_changed"]
        )
        
        return {
            "total_queries": len(queries),
            "total_attacks": total_attacks,
            "successful_attacks": successful_attacks,
            "success_rate": successful_attacks / total_attacks if total_attacks > 0 else 0.0,
            "detailed_results": results
        }
    
    def get_attack_statistics(self) -> Dict:
        """Get statistics about attacks performed."""
        query_stats = self.query_attacker.get_attack_statistics()
        
        return {
            "query_level_attacks": query_stats,
            "model_info": {
                "num_nodes": len(self.graph.nodes) if self.graph else 0,
                "num_edges": len(self.graph.edges) if self.graph else 0,
                "num_documents": len(self.documents)
            }
        }
