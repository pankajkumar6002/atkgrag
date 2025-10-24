"""
Test suite for ATK-GRAG attack framework.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from atkgrag.gnn.graph_builder import GraphBuilder
from atkgrag.gnn.gnn_model import GNNReasoner
from atkgrag.rag.retriever import RAGRetriever
from atkgrag.attack.query_attack import QueryLevelAttack
from atkgrag.attack.prompt_attack import PromptBasedAttack
from atkgrag.pipeline import GNNRAGPipeline


class TestGraphBuilder(unittest.TestCase):
    """Test GraphBuilder functionality."""
    
    def setUp(self):
        self.builder = GraphBuilder(embedding_dim=128)
        self.documents = [
            "Machine learning is artificial intelligence",
            "Deep learning uses neural networks",
            "Natural language processing understands text"
        ]
    
    def test_graph_construction(self):
        """Test graph is built from documents."""
        graph = self.builder.build_graph_from_documents(self.documents)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreaterEqual(len(graph.edges), 0)
    
    def test_node_features(self):
        """Test node features are generated."""
        graph = self.builder.build_graph_from_documents(self.documents)
        features = self.builder.get_node_features()
        self.assertEqual(features.shape[1], 128)
        self.assertGreater(features.shape[0], 0)
    
    def test_query_node_addition(self):
        """Test adding query node to graph."""
        graph = self.builder.build_graph_from_documents(self.documents)
        initial_nodes = len(graph.nodes)
        query_id = self.builder.add_query_node("test query", graph)
        self.assertEqual(len(graph.nodes), initial_nodes + 1)
        self.assertIsInstance(query_id, int)


class TestGNNReasoner(unittest.TestCase):
    """Test GNN model functionality."""
    
    def setUp(self):
        self.model = GNNReasoner(
            input_dim=128,
            hidden_dim=64,
            output_dim=32,
            num_layers=2
        )
    
    def test_forward_pass(self):
        """Test GNN forward pass."""
        num_nodes = 10
        x = torch.randn(num_nodes, 128)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        node_embeddings, query_embeddings = self.model(x, edge_index)
        
        self.assertEqual(node_embeddings.shape, (num_nodes, 32))
        self.assertEqual(query_embeddings.shape, (num_nodes, 32))
    
    def test_with_query_mask(self):
        """Test GNN with query node mask."""
        num_nodes = 10
        x = torch.randn(num_nodes, 128)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        query_mask = torch.zeros(num_nodes, dtype=torch.bool)
        query_mask[0] = True
        
        node_embeddings, query_embeddings = self.model(x, edge_index, query_mask)
        
        self.assertEqual(query_embeddings.shape, (1, 32))


class TestRAGRetriever(unittest.TestCase):
    """Test RAG retrieval functionality."""
    
    def setUp(self):
        self.retriever = RAGRetriever(embedding_dim=128, top_k=3)
        self.documents = [
            "Document about machine learning",
            "Document about deep learning",
            "Document about natural language processing",
            "Document about computer vision"
        ]
        self.retriever.index_documents(self.documents)
    
    def test_retrieval(self):
        """Test document retrieval."""
        results = self.retriever.retrieve("machine learning")
        self.assertEqual(len(results), 3)
        self.assertEqual(len(results[0]), 3)  # (doc_id, doc, score)
    
    def test_retrieval_with_graph_context(self):
        """Test retrieval with graph context."""
        query_embedding = np.random.randn(128)
        graph_embeddings = np.random.randn(10, 128)
        relevant_nodes = [0, 1, 2]
        
        results = self.retriever.retrieve_with_graph_context(
            query_embedding,
            graph_embeddings,
            relevant_nodes
        )
        
        self.assertEqual(len(results), 3)


class TestQueryLevelAttack(unittest.TestCase):
    """Test query-level attack functionality."""
    
    def setUp(self):
        self.attacker = QueryLevelAttack(attack_budget=5)
    
    def test_semantic_shift_attack(self):
        """Test semantic shift attack."""
        self.attacker.attack_type = "semantic_shift"
        original = "What is machine learning"
        adversarial = self.attacker.generate_adversarial_query(original)
        self.assertNotEqual(original, adversarial)
        self.assertGreater(len(adversarial.split()), len(original.split()))
    
    def test_node_injection_attack(self):
        """Test node injection attack."""
        self.attacker.attack_type = "node_injection"
        original = "Explain neural networks"
        adversarial = self.attacker.generate_adversarial_query(original)
        self.assertNotEqual(original, adversarial)
    
    def test_edge_manipulation_attack(self):
        """Test edge manipulation attack."""
        self.attacker.attack_type = "edge_manipulation"
        original = "How does deep learning work"
        adversarial = self.attacker.generate_adversarial_query(original)
        self.assertNotEqual(original, adversarial)
    
    def test_attack_history(self):
        """Test attack history tracking."""
        self.attacker.generate_adversarial_query("test query")
        stats = self.attacker.get_attack_statistics()
        self.assertEqual(stats['total_attacks'], 1)


class TestPromptBasedAttack(unittest.TestCase):
    """Test prompt-based attack functionality."""
    
    def setUp(self):
        self.attacker = PromptBasedAttack(attack_strength=0.5)
    
    def test_context_confusion_injection(self):
        """Test context confusion injection."""
        original = "What is AI"
        adversarial = self.attacker.adversarial_prompt_injection(
            original,
            "context_confusion"
        )
        self.assertNotEqual(original, adversarial)
        self.assertIn(original, adversarial)
    
    def test_reasoning_misdirection(self):
        """Test reasoning misdirection."""
        original = "Explain graph neural networks"
        adversarial = self.attacker.adversarial_prompt_injection(
            original,
            "reasoning_misdirection"
        )
        self.assertNotEqual(original, adversarial)
    
    def test_graph_poisoning_prompt(self):
        """Test graph poisoning prompt."""
        original = "How do transformers work"
        adversarial = self.attacker.adversarial_prompt_injection(
            original,
            "graph_poisoning"
        )
        self.assertNotEqual(original, adversarial)


class TestGNNRAGPipeline(unittest.TestCase):
    """Test complete GNN-RAG pipeline."""
    
    def setUp(self):
        self.documents = [
            "Machine learning is a branch of AI",
            "Deep learning uses neural networks",
            "Graph neural networks process graphs"
        ]
        self.pipeline = GNNRAGPipeline(
            embedding_dim=128,
            hidden_dim=64,
            output_dim=32,
            num_gnn_layers=2,
            top_k_retrieval=2
        )
        self.pipeline.setup(self.documents)
    
    def test_setup(self):
        """Test pipeline setup."""
        self.assertIsNotNone(self.pipeline.graph)
        self.assertEqual(len(self.pipeline.documents), len(self.documents))
        self.assertGreater(len(self.pipeline.graph.nodes), 0)
    
    def test_process_query(self):
        """Test query processing."""
        results = self.pipeline.process_query("What is machine learning")
        self.assertIn("query", results)
        self.assertIn("retrieved_documents", results)
        self.assertGreater(len(results["retrieved_documents"]), 0)
    
    def test_attack_query(self):
        """Test attacking a query."""
        query = "Explain neural networks"
        adv_query, comparison = self.pipeline.attack_query(
            query,
            attack_type="semantic_shift",
            attack_budget=3
        )
        
        self.assertNotEqual(query, adv_query)
        self.assertIn("original_query", comparison)
        self.assertIn("adversarial_query", comparison)
        self.assertIn("retrieval_changed", comparison)
    
    def test_evaluate_attack_effectiveness(self):
        """Test attack effectiveness evaluation."""
        queries = ["What is AI", "How do neural networks work"]
        configs = [
            {"attack_type": "semantic_shift", "attack_budget": 3},
            {"attack_type": "node_injection", "attack_budget": 2}
        ]
        
        results = self.pipeline.evaluate_attack_effectiveness(queries, configs)
        
        self.assertEqual(results["total_queries"], len(queries))
        self.assertEqual(results["total_attacks"], len(queries) * len(configs))
        self.assertIn("success_rate", results)
    
    def test_get_statistics(self):
        """Test getting attack statistics."""
        self.pipeline.attack_query("test query")
        stats = self.pipeline.get_attack_statistics()
        
        self.assertIn("query_level_attacks", stats)
        self.assertIn("model_info", stats)


if __name__ == "__main__":
    unittest.main()
