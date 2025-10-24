from setuptools import setup, find_packages

setup(
    name="atkgrag",
    version="0.1.0",
    description="Prompt-based attacks on GNN-RAG systems",
    author="Pankaj Kumar",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
        "networkx>=3.1",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
    ],
)
