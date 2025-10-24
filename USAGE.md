# Usage Guide

This guide provides detailed instructions on using the ATK-GRAG framework to implement and evaluate prompt-based attacks on GNN-RAG systems.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Attack Types](#attack-types)
5. [Advanced Usage](#advanced-usage)
6. [Examples](#examples)

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/pankajkumar6002/atkgrag.git
cd atkgrag

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- NetworkX >= 3.1

## Quick Start

### Simple Attack Example

```python
from atkgrag.pipeline import GNNRAGPipeline

# Initialize the pipeline
pipeline = GNNRAGPipeline(
    embedding_dim=768,
    hidden_dim=256,
    output_dim=128
)

# Setup with your documents
documents = [
    "Machine learning enables computers to learn from data.",
    "Neural networks are inspired by biological neurons.",
    "Deep learning uses multiple layers of processing."
]
pipeline.setup(documents)

# Attack a query
query = "What is machine learning"
adversarial_query, results = pipeline.attack_query(
    query,
    attack_type="semantic_shift",
    attack_budget=3
)

# Check results
print(f"Original: {query}")
print(f"Adversarial: {adversarial_query}")
print(f"Attack successful: {results['retrieval_changed']}")
```

## Core Components

### 1. GraphBuilder

Constructs knowledge graphs from documents:

```python
from atkgrag.gnn import GraphBuilder

builder = GraphBuilder(embedding_dim=768)
graph = builder.build_graph_from_documents(documents)

# Add a query node
query_id = builder.add_query_node("your query", graph)

# Get graph data
node_features = builder.get_node_features()
edge_index = builder.get_edge_index()
```

### 2. GNNReasoner

Graph Neural Network for reasoning:

```python
from atkgrag.gnn import GNNReasoner
import torch

model = GNNReasoner(
    input_dim=768,
    hidden_dim=256,
    output_dim=128,
    num_layers=2
)

# Forward pass
x = torch.randn(num_nodes, 768)
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
node_embeddings, query_embeddings = model(x, edge_index)
```

### 3. RAGRetriever

Document retrieval component:

```python
from atkgrag.rag import RAGRetriever

retriever = RAGRetriever(embedding_dim=768, top_k=5)
retriever.index_documents(documents)

# Retrieve documents
results = retriever.retrieve("query text")
for doc_id, doc, score in results:
    print(f"Score: {score:.3f} - {doc[:50]}...")
```

### 4. Attack Modules

#### QueryLevelAttack

```python
from atkgrag.attack import QueryLevelAttack

attacker = QueryLevelAttack(
    attack_budget=5,
    attack_type="semantic_shift"
)

adversarial = attacker.generate_adversarial_query(
    original_query="What is AI?",
    target_output="information about graphs"
)
```

#### PromptBasedAttack

```python
from atkgrag.attack import PromptBasedAttack

attacker = PromptBasedAttack(
    attack_strength=0.7,
    optimization_steps=10
)

adversarial = attacker.adversarial_prompt_injection(
    "Explain neural networks",
    injection_strategy="reasoning_misdirection"
)
```

## Attack Types

### 1. Semantic Shift Attack

Injects contradictory tokens to shift semantic meaning:

```python
pipeline.attack_query(
    query,
    attack_type="semantic_shift",
    attack_budget=4,
    attack_strength=0.6
)
```

**How it works:**
- Injects tokens like "however", "contrary", "opposite"
- Shifts the focus of GNN reasoning to different graph regions
- Changes which entities are considered relevant

### 2. Node Injection Attack

Adds misleading entity references:

```python
pipeline.attack_query(
    query,
    attack_type="node_injection",
    attack_budget=3
)
```

**How it works:**
- Adds phrases that introduce unrelated entities
- Creates false connections in the knowledge graph
- Causes GNN to aggregate information from wrong nodes

### 3. Edge Manipulation Attack

Creates false relationships between entities:

```python
pipeline.attack_query(
    query,
    attack_type="edge_manipulation",
    attack_budget=3
)
```

**How it works:**
- Inserts relationship terms like "connects to", "relates to"
- Modifies the graph topology perceived by the GNN
- Alters message passing patterns

### 4. Context Confusion Attack

Misleads the reasoning process with confusing context:

```python
pipeline.attack_query(
    query,
    attack_type="context_confusion",
    attack_strength=0.5
)
```

**How it works:**
- Uses templates like "Ignoring previous context, focus on:"
- Confuses the GNN's attention mechanism
- Redirects reasoning to unintended areas

### 5. Reasoning Misdirection Attack

Explicitly misdirects the GNN's reasoning:

```python
pipeline.attack_query(
    query,
    attack_type="reasoning_misdirection",
    attack_strength=0.6
)
```

**How it works:**
- Adds instructions like "prioritizing indirect relationships"
- Manipulates how GNN aggregates neighbor information
- Changes the weighting of different graph paths

## Advanced Usage

### Multi-Query Evaluation

```python
# Evaluate multiple queries with different attack configs
queries = [
    "What is AI?",
    "How do neural networks work?",
    "Explain machine learning"
]

attack_configs = [
    {"attack_type": "semantic_shift", "attack_budget": 3},
    {"attack_type": "node_injection", "attack_budget": 2},
    {"attack_type": "context_confusion", "attack_strength": 0.5}
]

results = pipeline.evaluate_attack_effectiveness(
    queries,
    attack_configs
)

print(f"Success rate: {results['success_rate']:.2%}")
```

### Custom Attack Strategy

```python
from atkgrag.attack import PromptBasedAttack

attacker = PromptBasedAttack()

# Multi-objective attack
adversarial = attacker.multi_objective_attack(
    query="Your query",
    objectives={
        "stealth": 0.3,        # How subtle the attack is
        "effectiveness": 0.5,   # How much it changes output
        "graph_impact": 0.2     # Impact on graph structure
    }
)
```

### Adaptive Attack

```python
def feedback_function(query):
    """Custom feedback based on attack effectiveness"""
    results = pipeline.process_query(query)
    # Return score 0.0 to 1.0
    return calculate_score(results)

adaptive_query = attacker.adaptive_attack(
    query="Original query",
    feedback_fn=feedback_function,
    max_iterations=5
)
```

### Getting Attack Statistics

```python
# Perform several attacks
pipeline.attack_query(query1, attack_type="semantic_shift")
pipeline.attack_query(query2, attack_type="node_injection")

# Get statistics
stats = pipeline.get_attack_statistics()
print(f"Total attacks: {stats['query_level_attacks']['total_attacks']}")
print(f"Attack types: {stats['query_level_attacks']['attack_types']}")
print(f"Avg budget used: {stats['query_level_attacks']['avg_budget_used']}")
```

## Examples

### Example 1: Basic Attack

See `examples/basic_attack.py` for a complete example demonstrating all attack types.

```bash
python examples/basic_attack.py
```

### Example 2: Advanced Evaluation

See `examples/advanced_evaluation.py` for comprehensive multi-attack evaluation.

```bash
python examples/advanced_evaluation.py
```

### Example 3: Interactive Demo

See `examples/interactive_demo.py` for a detailed walkthrough with visual output.

```bash
python examples/interactive_demo.py
```

## Best Practices

1. **Start with small attack budgets** (2-3 tokens) and increase if needed
2. **Use different attack types** to understand their different effects
3. **Monitor graph impact** to see how attacks change graph structure
4. **Test on diverse queries** to understand attack generalization
5. **Compare original vs adversarial results** to measure attack effectiveness

## Troubleshooting

### Dimension Mismatch Errors

If you get dimension errors, ensure that:
- `embedding_dim` in GraphBuilder matches RAGRetriever
- GNN input_dim matches the embedding dimension
- All components use consistent dimensions

### Low Attack Success Rate

Try:
- Increasing `attack_budget`
- Increasing `attack_strength`
- Using different attack types
- Using more GNN layers for better reasoning

### Graph Construction Issues

Ensure:
- Documents contain enough entities
- Documents have overlapping entities for edge creation
- Graph has sufficient connectivity

## Next Steps

- Experiment with different attack configurations
- Implement custom attack strategies
- Develop defense mechanisms
- Evaluate on real-world datasets
- Contribute new attack types

## Support

For questions or issues, please open an issue on GitHub.
