# Leiden-LPA Hybrid

A hybrid community detection algorithm that combines the modularity optimization of **Leiden** with the scalability of **Label Propagation**.
## 🔍 Overview

This repository provides:

- An implementation of the **Leiden-LPA Hybrid** algorithm.
- Scripts to run experiments on both **synthetic** and **real-world** datasets.
- Evaluation code for **modularity**, **NMI**, and **runtime**.
- Data generators and plotting tools for visual analysis.


## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Datasets (Optional)

```bash
python3 data_generate/generate_dataset.py
```

### 3. Run Experiments

#### (a) Synthetic Graphs with Core Ratio Variation

```bash
python3 ratio/ratiorunner.py
python3 ratio/ratiosummarize.py 
```

#### (b) Real-world Datasets

```bash
python3 real/realrunner.py
python3 real/realsummarize.py
```

### 4. Visualize Results

```bash
# For ratio-based experiments
python3 ratio/plot.py

# For real-world datasets
python3 real/plot.py
```

## 📈 Evaluation Metrics

- **Modularity**: Measures the quality of the detected communities based on intra-cluster density.
- **Normalized Mutual Information (NMI)**: Compares detected labels against ground truth.
- **Runtime**: Execution time measured for performance comparison.
