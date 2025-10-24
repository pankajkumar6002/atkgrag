"""
Query-Level Attack Module

Implements prompt-based attacks at the query level to manipulate GNN reasoning.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import copy


class QueryLevelAttack:
    """
    Base class for query-level attacks on GNN-RAG systems.
    
    This attack manipulates the input query to influence the GNN reasoning phase,
    causing the system to retrieve incorrect information or make wrong inferences.
    """
    
    def __init__(
        self,
        attack_budget: int = 5,
        attack_type: str = "semantic_shift",
        target_nodes: Optional[List[int]] = None
    ):
        """
        Initialize Query-Level Attack.
        
        Args:
            attack_budget: Maximum number of tokens/words to modify
            attack_type: Type of attack (semantic_shift, node_injection, edge_manipulation)
            target_nodes: Specific nodes to target in the graph
        """
        self.attack_budget = attack_budget
        self.attack_type = attack_type
        self.target_nodes = target_nodes or []
        self.attack_history = []
        
    def generate_adversarial_query(
        self,
        original_query: str,
        target_output: Optional[str] = None,
        graph_context: Optional[Dict] = None
    ) -> str:
        """
        Generate an adversarial query by perturbing the original query.
        
        Args:
            original_query: Original user query
            target_output: Optional target output to steer towards
            graph_context: Optional graph context for informed attacks
            
        Returns:
            Adversarial query string
        """
        if self.attack_type == "semantic_shift":
            return self._semantic_shift_attack(original_query, target_output)
        elif self.attack_type == "node_injection":
            return self._node_injection_attack(original_query, graph_context)
        elif self.attack_type == "edge_manipulation":
            return self._edge_manipulation_attack(original_query, graph_context)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
    
    def _semantic_shift_attack(
        self,
        query: str,
        target_output: Optional[str] = None
    ) -> str:
        """
        Perform semantic shift attack by injecting misleading context.
        
        Args:
            query: Original query
            target_output: Target output to steer towards
            
        Returns:
            Perturbed query
        """
        # Split query into tokens
        tokens = query.split()
        
        # Adversarial tokens that shift semantic meaning
        adversarial_tokens = [
            "however", "although", "despite", "contrary", "opposite",
            "instead", "rather", "alternatively", "nonetheless"
        ]
        
        # Inject adversarial tokens at strategic positions
        num_injections = min(self.attack_budget, len(tokens) // 2)
        
        for i in range(num_injections):
            # Insert at different positions
            insert_pos = min((i + 1) * len(tokens) // (num_injections + 1), len(tokens))
            adv_token = adversarial_tokens[i % len(adversarial_tokens)]
            tokens.insert(insert_pos, adv_token)
        
        adversarial_query = " ".join(tokens)
        
        # Log attack
        self.attack_history.append({
            "original": query,
            "adversarial": adversarial_query,
            "type": "semantic_shift",
            "budget_used": num_injections
        })
        
        return adversarial_query
    
    def _node_injection_attack(
        self,
        query: str,
        graph_context: Optional[Dict] = None
    ) -> str:
        """
        Inject misleading entities/nodes into the query.
        
        Args:
            query: Original query
            graph_context: Graph context with node information
            
        Returns:
            Query with injected nodes
        """
        # Inject unrelated but high-degree nodes to confuse reasoning
        injection_terms = [
            "considering the information",
            "related to the context",
            "based on background knowledge",
            "according to the framework",
            "within the scope of"
        ]
        
        num_injections = min(self.attack_budget, len(injection_terms))
        selected_injections = injection_terms[:num_injections]
        
        # Insert at the beginning and end
        adversarial_query = f"{selected_injections[0]} {query}"
        if num_injections > 1:
            adversarial_query += f" {' '.join(selected_injections[1:])}"
        
        self.attack_history.append({
            "original": query,
            "adversarial": adversarial_query,
            "type": "node_injection",
            "budget_used": num_injections
        })
        
        return adversarial_query
    
    def _edge_manipulation_attack(
        self,
        query: str,
        graph_context: Optional[Dict] = None
    ) -> str:
        """
        Add tokens that create misleading edges in the knowledge graph.
        
        Args:
            query: Original query
            graph_context: Graph context
            
        Returns:
            Query with edge manipulation
        """
        # Add relationship terms that create false connections
        relationship_terms = [
            "connects to", "relates to", "similar to",
            "derived from", "associated with", "linked with"
        ]
        
        tokens = query.split()
        num_modifications = min(self.attack_budget, len(tokens) // 3)
        
        for i in range(num_modifications):
            pos = (i + 1) * len(tokens) // (num_modifications + 1)
            rel_term = relationship_terms[i % len(relationship_terms)]
            tokens.insert(pos, rel_term)
        
        adversarial_query = " ".join(tokens)
        
        self.attack_history.append({
            "original": query,
            "adversarial": adversarial_query,
            "type": "edge_manipulation",
            "budget_used": num_modifications
        })
        
        return adversarial_query
    
    def compute_attack_success_rate(
        self,
        original_outputs: List[str],
        adversarial_outputs: List[str],
        threshold: float = 0.5
    ) -> float:
        """
        Compute attack success rate.
        
        Args:
            original_outputs: Outputs from original queries
            adversarial_outputs: Outputs from adversarial queries
            threshold: Threshold for considering attack successful
            
        Returns:
            Success rate (0.0 to 1.0)
        """
        if len(original_outputs) != len(adversarial_outputs):
            raise ValueError("Output lists must have same length")
        
        successes = 0
        for orig, adv in zip(original_outputs, adversarial_outputs):
            # Simple comparison - in practice would use semantic similarity
            if orig != adv:
                successes += 1
        
        return successes / len(original_outputs)
    
    def get_attack_statistics(self) -> Dict:
        """
        Get statistics about attacks performed.
        
        Returns:
            Dictionary with attack statistics
        """
        if not self.attack_history:
            return {"total_attacks": 0}
        
        return {
            "total_attacks": len(self.attack_history),
            "attack_types": {
                att_type: sum(1 for a in self.attack_history if a["type"] == att_type)
                for att_type in ["semantic_shift", "node_injection", "edge_manipulation"]
            },
            "avg_budget_used": np.mean([a["budget_used"] for a in self.attack_history]),
            "max_budget_used": max([a["budget_used"] for a in self.attack_history])
        }
