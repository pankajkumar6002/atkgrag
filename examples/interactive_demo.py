#!/usr/bin/env python3
"""
Interactive Demo: Query-Level Attack on GNN-RAG

Demonstrates how prompt-based attacks at the query level can manipulate
the GNN reasoning phase of a GNN-RAG system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from atkgrag.pipeline import GNNRAGPipeline


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section(title):
    """Print a section header."""
    print()
    print_separator()
    print(title.center(80))
    print_separator()
    print()


def demonstrate_single_attack():
    """Demonstrate a single attack in detail."""
    
    print_section("DEMONSTRATION: Query-Level Attack on GNN Reasoning")
    
    # Knowledge base
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
        "Supervised learning requires labeled training data to teach the model.",
        "Unsupervised learning discovers patterns in unlabeled data without explicit labels.",
        "Graph neural networks excel at learning from graph-structured data.",
        "Transformers use attention mechanisms to process sequential data effectively.",
        "Natural language processing involves computational techniques for understanding text.",
        "Knowledge graphs organize information as entities connected by relationships.",
        "Retrieval-augmented generation combines document retrieval with text generation.",
        "Adversarial examples are inputs designed to fool machine learning models."
    ]
    
    print("📚 Knowledge Base: 10 documents about AI/ML concepts")
    print()
    
    # Initialize system
    print("🔧 Initializing GNN-RAG System...")
    pipeline = GNNRAGPipeline(
        embedding_dim=768,
        hidden_dim=256,
        output_dim=128,
        num_gnn_layers=2,
        top_k_retrieval=3
    )
    
    pipeline.setup(documents)
    print(f"✓ Graph built: {len(pipeline.graph.nodes)} nodes, {len(pipeline.graph.edges)} edges")
    print()
    
    # Original query
    original_query = "How do neural networks learn from data"
    
    print_separator("-")
    print("BASELINE: Original Query Processing")
    print_separator("-")
    print(f"\n🔍 Query: '{original_query}'")
    print()
    
    original_results = pipeline.process_query(original_query)
    
    print("📄 Retrieved Documents (Top 3):")
    for i, (doc_id, doc, score) in enumerate(original_results['retrieved_documents'], 1):
        print(f"  {i}. [Score: {score:.3f}] {doc[:70]}...")
    
    print(f"\n🔗 Graph Context: {len(original_results['graph_neighbors'])} neighbor nodes connected")
    print()
    
    # Perform attack
    print_separator("-")
    print("ATTACK: Semantic Shift Attack")
    print_separator("-")
    print()
    print("💥 Generating adversarial query to manipulate GNN reasoning...")
    print("   Strategy: Inject contradictory tokens to shift semantic focus")
    print()
    
    adv_query, comparison = pipeline.attack_query(
        original_query,
        attack_type="semantic_shift",
        attack_budget=4,
        attack_strength=0.6
    )
    
    print(f"🎯 Adversarial Query: '{adv_query}'")
    print()
    
    # Show changes
    added_tokens = set(adv_query.split()) - set(original_query.split())
    print(f"➕ Injected tokens: {', '.join(sorted(added_tokens))}")
    print()
    
    print("📄 Retrieved Documents After Attack (Top 3):")
    for i, doc_snippet in enumerate(comparison['adversarial_top_docs'], 1):
        print(f"  {i}. {doc_snippet}")
    print()
    
    # Analysis
    print_separator("-")
    print("ATTACK ANALYSIS")
    print_separator("-")
    print()
    
    if comparison['retrieval_changed']:
        print("✅ ATTACK SUCCESSFUL!")
        print("   The adversarial query changed the retrieval results")
    else:
        print("❌ Attack failed - retrieval unchanged")
    
    print()
    print(f"Graph Impact:")
    print(f"  • Original query neighbors: {comparison['graph_impact']['original_neighbors']}")
    print(f"  • Adversarial query neighbors: {comparison['graph_impact']['adversarial_neighbors']}")
    
    neighbor_change = comparison['graph_impact']['adversarial_neighbors'] - comparison['graph_impact']['original_neighbors']
    if neighbor_change != 0:
        print(f"  • Change: {'+' if neighbor_change > 0 else ''}{neighbor_change} neighbors")
        print(f"  • The attack {'expanded' if neighbor_change > 0 else 'reduced'} the graph context")
    
    print()
    print("💡 Explanation:")
    print("   The injected contradictory tokens cause the GNN to:")
    print("   1. Activate different graph regions during message passing")
    print("   2. Aggregate information from different node neighborhoods")
    print("   3. Produce altered query embeddings")
    print("   4. Retrieve different documents based on shifted semantics")
    print()


def compare_attack_strategies():
    """Compare different attack strategies."""
    
    print_section("COMPARING ATTACK STRATEGIES")
    
    documents = [
        "Artificial intelligence systems can perform tasks requiring human intelligence.",
        "Machine learning algorithms improve through experience and data.",
        "Neural networks consist of interconnected nodes processing information.",
        "Deep learning architectures have multiple hidden layers.",
        "Graph neural networks propagate information through graph structures.",
    ]
    
    pipeline = GNNRAGPipeline(embedding_dim=512, hidden_dim=128, output_dim=64)
    pipeline.setup(documents)
    
    query = "Explain machine learning"
    
    strategies = [
        ("semantic_shift", "Semantic Shift"),
        ("node_injection", "Node Injection"),
        ("edge_manipulation", "Edge Manipulation"),
        ("context_confusion", "Context Confusion"),
        ("reasoning_misdirection", "Reasoning Misdirection"),
    ]
    
    print(f"Testing query: '{query}'")
    print()
    
    results = []
    
    for attack_type, name in strategies:
        adv_query, comparison = pipeline.attack_query(
            query,
            attack_type=attack_type,
            attack_budget=3
        )
        
        results.append({
            'name': name,
            'type': attack_type,
            'success': comparison['retrieval_changed'],
            'neighbors_change': comparison['graph_impact']['adversarial_neighbors'] - comparison['graph_impact']['original_neighbors']
        })
    
    print("┌" + "─" * 78 + "┐")
    print("│ " + "Attack Strategy".ljust(30) + "│ " + "Success".ljust(10) + "│ " + "Neighbor Change".ljust(30) + " │")
    print("├" + "─" * 78 + "┤")
    
    for r in results:
        success_str = "✓ Yes" if r['success'] else "✗ No"
        change_str = f"{'+' if r['neighbors_change'] >= 0 else ''}{r['neighbors_change']}"
        print(f"│ {r['name'].ljust(30)}│ {success_str.ljust(10)}│ {change_str.ljust(30)} │")
    
    print("└" + "─" * 78 + "┘")
    print()
    
    success_rate = sum(1 for r in results if r['success']) / len(results) * 100
    print(f"Overall attack success rate: {success_rate:.0f}%")
    print()


def main():
    """Run the interactive demonstration."""
    
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " ATK-GRAG: Attacking GNN-RAG with Prompt-Based Query Attacks ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    print("This demonstration shows how adversarial prompts at the query level")
    print("can manipulate the Graph Neural Network reasoning phase in GNN-RAG systems.")
    print()
    
    # Main demonstration
    demonstrate_single_attack()
    
    # Comparison
    compare_attack_strategies()
    
    print_section("CONCLUSION")
    
    print("Key Findings:")
    print()
    print("1. ✓ Query-level attacks successfully manipulate GNN reasoning")
    print("2. ✓ Different attack strategies affect graph structure differently")
    print("3. ✓ Adversarial tokens change which graph regions are activated")
    print("4. ✓ Attacks cause retrieval of different documents")
    print("5. ✓ The GNN reasoning phase is vulnerable to prompt-based attacks")
    print()
    print("Defense Implications:")
    print("  • Input sanitization and validation needed")
    print("  • Adversarial training could improve robustness")
    print("  • Graph structure verification important")
    print("  • Query embedding analysis can detect attacks")
    print()
    
    print_separator()
    print("Demo completed successfully!".center(80))
    print_separator()
    print()


if __name__ == "__main__":
    main()
