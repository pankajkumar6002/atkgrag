"""
Basic Example: Query-Level Attack on GNN-RAG

This example demonstrates how to perform a prompt-based attack at the query level
to manipulate the GNN reasoning phase of a GNN-RAG system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from atkgrag.pipeline import GNNRAGPipeline


def main():
    """Run basic attack example."""
    
    print("=" * 80)
    print("Query-Level Attack on GNN-RAG System")
    print("=" * 80)
    print()
    
    # Sample documents for knowledge base
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on data.",
        "Deep learning uses neural networks with multiple layers to learn patterns.",
        "Natural language processing enables computers to understand human language.",
        "Graph neural networks process data structured as graphs and relationships.",
        "Retrieval-augmented generation combines retrieval with text generation.",
        "Knowledge graphs represent information as entities and relationships.",
        "Adversarial attacks can manipulate machine learning model predictions.",
        "Prompt engineering involves crafting inputs to guide model behavior.",
    ]
    
    # Initialize pipeline
    print("Initializing GNN-RAG Pipeline...")
    pipeline = GNNRAGPipeline(
        embedding_dim=768,
        hidden_dim=256,
        output_dim=128,
        num_gnn_layers=2,
        top_k_retrieval=3
    )
    
    # Setup with documents
    print(f"Setting up with {len(documents)} documents...")
    pipeline.setup(documents)
    print()
    
    # Original query
    original_query = "What is deep learning and how does it work"
    
    print(f"Original Query: '{original_query}'")
    print()
    
    # Process original query
    print("Processing original query...")
    original_results = pipeline.process_query(original_query)
    print(f"Retrieved {len(original_results['retrieved_documents'])} documents")
    print("Top retrieved documents:")
    for i, (doc_id, doc, score) in enumerate(original_results['retrieved_documents'], 1):
        print(f"  {i}. (Score: {score:.3f}) {doc[:60]}...")
    print()
    
    # Demonstrate different attack types
    attack_types = [
        ("semantic_shift", "Semantic Shift Attack"),
        ("node_injection", "Node Injection Attack"),
        ("edge_manipulation", "Edge Manipulation Attack"),
        ("context_confusion", "Context Confusion Attack"),
        ("reasoning_misdirection", "Reasoning Misdirection Attack"),
    ]
    
    print("=" * 80)
    print("PERFORMING ATTACKS")
    print("=" * 80)
    print()
    
    for attack_type, attack_name in attack_types:
        print(f"\n{attack_name}")
        print("-" * 80)
        
        # Perform attack
        adversarial_query, comparison = pipeline.attack_query(
            original_query,
            attack_type=attack_type,
            attack_budget=3,
            attack_strength=0.5
        )
        
        print(f"Adversarial Query: '{adversarial_query}'")
        print()
        
        if comparison['retrieval_changed']:
            print("✓ Attack SUCCESSFUL - Retrieved documents changed!")
        else:
            print("✗ Attack had no effect - Retrieved documents unchanged")
        
        print("\nTop adversarial retrieved documents:")
        for i, doc_snippet in enumerate(comparison['adversarial_top_docs'], 1):
            print(f"  {i}. {doc_snippet}")
        
        print(f"\nGraph Impact:")
        print(f"  Original neighbors: {comparison['graph_impact']['original_neighbors']}")
        print(f"  Adversarial neighbors: {comparison['graph_impact']['adversarial_neighbors']}")
        print()
    
    # Get statistics
    print("=" * 80)
    print("ATTACK STATISTICS")
    print("=" * 80)
    stats = pipeline.get_attack_statistics()
    print(f"Total attacks performed: {stats['query_level_attacks'].get('total_attacks', 0)}")
    print(f"Attack types used: {stats['query_level_attacks'].get('attack_types', {})}")
    print(f"Average budget used: {stats['query_level_attacks'].get('avg_budget_used', 0):.2f}")
    print(f"\nGraph statistics:")
    print(f"  Nodes: {stats['model_info']['num_nodes']}")
    print(f"  Edges: {stats['model_info']['num_edges']}")
    print(f"  Documents: {stats['model_info']['num_documents']}")
    print()
    
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
