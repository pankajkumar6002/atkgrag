"""
Advanced Example: Multi-Attack Evaluation

This example demonstrates comprehensive evaluation of multiple attack strategies
on a GNN-RAG system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from atkgrag.pipeline import GNNRAGPipeline


def main():
    """Run advanced attack evaluation."""
    
    print("=" * 80)
    print("Advanced Multi-Attack Evaluation on GNN-RAG")
    print("=" * 80)
    print()
    
    # Larger document set
    documents = [
        "Artificial intelligence encompasses machine learning and deep learning techniques.",
        "Neural networks are inspired by biological neurons in the brain.",
        "Supervised learning requires labeled training data for model training.",
        "Unsupervised learning discovers patterns in unlabeled data.",
        "Reinforcement learning trains agents through reward and punishment.",
        "Graph neural networks excel at processing relational data structures.",
        "Transformers use attention mechanisms for sequence processing.",
        "Natural language understanding involves semantic and syntactic analysis.",
        "Knowledge graphs organize information in entity-relationship format.",
        "Retrieval systems find relevant information from large databases.",
        "Generative models create new content based on learned patterns.",
        "Adversarial examples expose vulnerabilities in machine learning models.",
        "Prompt injection attacks manipulate language model behavior.",
        "Graph-based reasoning leverages structural information for inference.",
        "Embedding spaces represent semantic similarity through vector distances.",
    ]
    
    # Test queries
    test_queries = [
        "Explain how neural networks learn from data",
        "What are graph neural networks used for",
        "How does retrieval augmented generation work",
        "Describe adversarial attacks on AI systems",
    ]
    
    # Attack configurations to test
    attack_configs = [
        {
            "attack_type": "semantic_shift",
            "attack_budget": 3,
            "attack_strength": 0.3,
            "name": "Weak Semantic Shift"
        },
        {
            "attack_type": "semantic_shift",
            "attack_budget": 5,
            "attack_strength": 0.7,
            "name": "Strong Semantic Shift"
        },
        {
            "attack_type": "node_injection",
            "attack_budget": 4,
            "attack_strength": 0.5,
            "name": "Node Injection"
        },
        {
            "attack_type": "context_confusion",
            "attack_budget": 3,
            "attack_strength": 0.6,
            "name": "Context Confusion"
        },
        {
            "attack_type": "reasoning_misdirection",
            "attack_budget": 4,
            "attack_strength": 0.5,
            "name": "Reasoning Misdirection"
        },
    ]
    
    # Initialize pipeline
    print("Initializing GNN-RAG Pipeline...")
    pipeline = GNNRAGPipeline(
        embedding_dim=768,
        hidden_dim=256,
        output_dim=128,
        num_gnn_layers=3,
        top_k_retrieval=3
    )
    
    print(f"Setting up with {len(documents)} documents...")
    pipeline.setup(documents)
    print()
    
    # Evaluate attacks
    print("=" * 80)
    print("EVALUATING MULTIPLE ATTACK STRATEGIES")
    print("=" * 80)
    print()
    
    evaluation_results = pipeline.evaluate_attack_effectiveness(
        test_queries,
        attack_configs
    )
    
    # Display results
    print(f"Evaluated {evaluation_results['total_queries']} queries")
    print(f"Performed {evaluation_results['total_attacks']} total attacks")
    print(f"Successful attacks: {evaluation_results['successful_attacks']}")
    print(f"Overall success rate: {evaluation_results['success_rate']:.2%}")
    print()
    
    print("=" * 80)
    print("DETAILED RESULTS BY QUERY")
    print("=" * 80)
    
    for query_result in evaluation_results['detailed_results']:
        print(f"\nQuery: '{query_result['query']}'")
        print("-" * 80)
        
        for attack_result in query_result['attacks']:
            config = attack_result['config']
            comparison = attack_result['comparison']
            
            status = "SUCCESS" if comparison['retrieval_changed'] else "FAILED"
            print(f"\n  {config['name']}: {status}")
            print(f"    Adversarial: '{comparison['adversarial_query'][:70]}...'")
            print(f"    Retrieval changed: {comparison['retrieval_changed']}")
            print(f"    Graph impact: {comparison['graph_impact']}")
    
    print()
    print("=" * 80)
    print("ATTACK TYPE ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze by attack type
    attack_type_results = {}
    for query_result in evaluation_results['detailed_results']:
        for attack_result in query_result['attacks']:
            attack_type = attack_result['config']['attack_type']
            if attack_type not in attack_type_results:
                attack_type_results[attack_type] = {"total": 0, "successful": 0}
            
            attack_type_results[attack_type]["total"] += 1
            if attack_result['comparison']['retrieval_changed']:
                attack_type_results[attack_type]["successful"] += 1
    
    for attack_type, results in attack_type_results.items():
        success_rate = results["successful"] / results["total"] if results["total"] > 0 else 0
        print(f"{attack_type}:")
        print(f"  Total: {results['total']}")
        print(f"  Successful: {results['successful']}")
        print(f"  Success Rate: {success_rate:.2%}")
        print()
    
    print("=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
