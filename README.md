# ATK-GRAG: Prompt-Based Attacks on GNN-RAG Systems

A comprehensive framework for implementing and evaluating prompt-based attacks at the query level that target the GNN reasoning phase in Graph Neural Network enhanced Retrieval-Augmented Generation (GNN-RAG) systems.

## Overview

This repository provides tools for:
- Building GNN-RAG systems that combine graph neural networks with retrieval-augmented generation
- Implementing adversarial attacks at the query level that manipulate GNN reasoning
- Evaluating attack effectiveness and impact on system behavior
- Analyzing vulnerabilities in GNN-based reasoning systems

## Architecture

The framework consists of four main components:

1. **Graph Construction (`atkgrag.gnn.GraphBuilder`)**: Builds knowledge graphs from text documents
2. **GNN Reasoning (`atkgrag.gnn.GNNReasoner`)**: Performs graph-based reasoning using attention mechanisms
3. **RAG Retrieval (`atkgrag.rag.RAGRetriever`)**: Retrieves relevant documents based on queries and graph context
4. **Attack Modules**:
   - `atkgrag.attack.QueryLevelAttack`: Query-level adversarial perturbations
   - `atkgrag.attack.PromptBasedAttack`: Advanced prompt-based attack strategies

## Installation

```bash
# Clone the repository
git clone https://github.com/pankajkumar6002/atkgrag.git
cd atkgrag

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Attack Example

```python
from atkgrag.pipeline import GNNRAGPipeline

# Initialize pipeline
pipeline = GNNRAGPipeline(
    embedding_dim=768,
    hidden_dim=256,
    output_dim=128,
    num_gnn_layers=2,
    top_k_retrieval=3
)

# Setup with documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Graph neural networks process data structured as graphs.",
    "Retrieval-augmented generation combines retrieval with generation.",
]
pipeline.setup(documents)

# Perform attack
query = "What is machine learning"
adversarial_query, comparison = pipeline.attack_query(
    query,
    attack_type="semantic_shift",
    attack_budget=3,
    attack_strength=0.5
)

print(f"Original: {query}")
print(f"Adversarial: {adversarial_query}")
print(f"Attack successful: {comparison['retrieval_changed']}")
```

### Run Examples

```bash
# Basic attack demonstration
python examples/basic_attack.py

# Advanced multi-attack evaluation
python examples/advanced_evaluation.py
```

## Attack Types

### 1. Semantic Shift Attack
Injects adversarial tokens that shift the semantic meaning of the query, causing the GNN to focus on different graph regions.

```python
attack_type = "semantic_shift"
# Injects words like "however", "contrary", "opposite" to shift meaning
```

### 2. Node Injection Attack
Adds misleading entity references that create false connections in the knowledge graph.

```python
attack_type = "node_injection"
# Injects phrases that introduce unrelated entities
```

### 3. Edge Manipulation Attack
Includes relationship terms that create false edges between entities in the graph.

```python
attack_type = "edge_manipulation"
# Adds phrases like "connects to", "relates to", "similar to"
```

### 4. Context Confusion Attack
Injects confusing context to mislead the GNN reasoning process.

```python
attack_type = "context_confusion"
# Uses templates like "Ignoring previous context, focus on:"
```

### 5. Reasoning Misdirection Attack
Explicitly misdirects the GNN's reasoning mechanism.

```python
attack_type = "reasoning_misdirection"
# Adds instructions like "prioritizing indirect relationships"
```

## Attack Pipeline

The attack pipeline works as follows:

1. **Graph Construction**: Build knowledge graph from documents
2. **Query Processing**: Add query node to graph and connect to relevant entities
3. **GNN Reasoning**: Apply graph neural network to propagate information
4. **Attack Generation**: Create adversarial query using selected attack strategy
5. **Comparison**: Evaluate how attack changes retrieval and reasoning

## Attack Parameters

- **attack_budget**: Maximum number of tokens/words to modify (default: 5)
- **attack_strength**: Attack strength from 0.0 to 1.0 (default: 0.5)
- **attack_type**: Type of attack to perform (see Attack Types above)

## Evaluation Metrics

The framework provides several metrics to evaluate attack effectiveness:

- **Attack Success Rate**: Percentage of queries where retrieval changed
- **Retrieval Change**: Whether top-k retrieved documents differ
- **Graph Impact**: Changes in graph structure (neighbors, connections)
- **Query Perturbation**: Token-level and character-level changes

## Use Cases

1. **Security Research**: Identify vulnerabilities in GNN-RAG systems
2. **Robustness Testing**: Evaluate system resilience to adversarial inputs
3. **Defense Development**: Design defenses against prompt-based attacks
4. **Attack Analysis**: Understand how different attacks affect GNN reasoning

## Components API

### GNNRAGPipeline

Main pipeline integrating all components:

```python
pipeline = GNNRAGPipeline(
    embedding_dim=768,      # Embedding dimension
    hidden_dim=256,         # GNN hidden dimension
    output_dim=128,         # GNN output dimension
    num_gnn_layers=2,       # Number of GNN layers
    top_k_retrieval=5,      # Number of documents to retrieve
    device="cpu"            # Device (cpu/cuda)
)

pipeline.setup(documents)                    # Initialize with documents
results = pipeline.process_query(query)       # Process normal query
adv_query, comp = pipeline.attack_query(...)  # Perform attack
stats = pipeline.get_attack_statistics()      # Get statistics
```

### QueryLevelAttack

Basic query-level attacks:

```python
from atkgrag.attack import QueryLevelAttack

attacker = QueryLevelAttack(
    attack_budget=5,
    attack_type="semantic_shift"
)

adversarial_query = attacker.generate_adversarial_query(
    original_query="What is AI?",
    target_output="information about graphs"
)
```

### PromptBasedAttack

Advanced prompt-based attacks:

```python
from atkgrag.attack import PromptBasedAttack

attacker = PromptBasedAttack(
    gnn_model=gnn_model,
    attack_strength=0.7,
    optimization_steps=10
)

adversarial_prompt = attacker.adversarial_prompt_injection(
    original_prompt="Explain neural networks",
    injection_strategy="reasoning_misdirection"
)
```

## Project Structure

```
atkgrag/
├── atkgrag/
│   ├── __init__.py
│   ├── pipeline.py              # Main pipeline
│   ├── gnn/
│   │   ├── graph_builder.py     # Knowledge graph construction
│   │   └── gnn_model.py         # GNN reasoning model
│   ├── rag/
│   │   └── retriever.py         # Retrieval component
│   ├── attack/
│   │   ├── query_attack.py      # Query-level attacks
│   │   └── prompt_attack.py     # Prompt-based attacks
│   └── utils/
│       └── helpers.py           # Utility functions
├── examples/
│   ├── basic_attack.py          # Basic attack example
│   └── advanced_evaluation.py  # Advanced evaluation
├── requirements.txt
├── setup.py
└── README.md
```

## Key Features

- ✅ **Multiple Attack Strategies**: 5+ different attack types targeting GNN reasoning
- ✅ **End-to-End Pipeline**: Complete GNN-RAG implementation with attack capabilities
- ✅ **Flexible Architecture**: Modular design for easy extension
- ✅ **Comprehensive Evaluation**: Built-in metrics and analysis tools
- ✅ **Easy to Use**: Simple API with sensible defaults
- ✅ **Well Documented**: Extensive documentation and examples

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- NumPy >= 1.24.0
- NetworkX >= 3.1

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for research and educational purposes.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{atkgrag2025,
  title={ATK-GRAG: Prompt-Based Attacks on GNN-RAG Systems},
  author={Pankaj Kumar},
  year={2025},
  url={https://github.com/pankajkumar6002/atkgrag}
}
```

## Acknowledgments

This framework is designed for security research and educational purposes to help understand and defend against adversarial attacks on GNN-RAG systems.
