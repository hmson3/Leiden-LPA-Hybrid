# LLAMA: Leiden-LPA Approach for Massive-community Analysis

**LLAMA** combines the **accuracy of the Leiden algorithm** with the **scalability of Label Propagation Algorithm (LPA)** for efficient community detection in large networks.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```python
import networkx as nx
from src.llama import llama

# Load graph
G = nx.karate_club_graph()

# Run hybrid algorithm
labels = llama(G, core_ratio=0.3, seed=42)

print(f"Number of communities: {len(set(labels.values()))}")
```

### 3. Specify Centrality Method

```python
from src.centrality_llama import llama

# PageRank-based core node selection
labels = llama(G, core_ratio=0.4, centrality_method='pagerank', seed=42)

# Degree Centrality-based
labels = llama(G, core_ratio=0.4, centrality_method='degree', seed=42)
```

## 📊 Running Experiments

### 1. Basic Performance Comparison (EQ1)

```bash
cd experiments/eq1
python run_eq1.py
```

Compares the hybrid algorithm with baseline algorithms on various real-world networks.

### 2. Core Ratio & Centrality Method Analysis (EQ2-3)

```bash
cd experiments/eq2_eq3
python run_eq2_eq3.py
```

- **EQ2**: Performance analysis across core ratio variations (0.1 ~ 0.9)
- **EQ3**: Performance comparison across centrality methods (PageRank, Degree, Eigenvector, Betweenness, Closeness)

### 3. Scalability Experiments (EQ4)

```bash
cd experiments/eq4
python run_eq4.py
```

Measures runtime and quality across different network sizes using LFR benchmark datasets.
