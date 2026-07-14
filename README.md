# Graph-Enhanced Causal Reinforcement Learning for Proactive Customer Retention: A Comparative Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This repository contains the complete, reproducible source code, dataset preprocessing pipelines, and detailed hyperparameter configurations for the paper **"Graph-Enhanced Causal Reinforcement Learning for Proactive Customer Retention: A Comparative Benchmark"**.
## Abstract
Customer retention in digital ecosystems has traditionally relied on reactive, predictive models that fail to capture the true causal impact of interventions. This paper benchmarks an end-to-end framework integrating **Graph Neural Networks (GNNs)**, **Causal Inference (Uplift Modeling)**, and **Offline Reinforcement Learning (RL)** to shift retention strategies from reactive prediction to proactive, causally-driven intervention.

## Repository Architecture
To facilitate reproducibility, the repository is modularized into distinct analytical phases:
```text
graph-causal-rl-benchmark/
├── configs/
│   └── hyperparameters.yaml       # Hyperparameter configurations
├── data_processing/               
│   ├── obd_preprocessing.py       # Open Bandit Dataset pipeline
│   └── kuairec_preprocessing.py   # KuaiRec dataset pipeline
├── scripts/
│   ├── train_gnn_embeddings.py    # GNN (LightGCN/GraphSAGE) Training
│   ├── train_causal_uplift.py     # X-Learner / T-Learner CATE estimators
│   └── train_offline_rl.py        # BCQ, CQL, IQL, LinUCB pipeline
├── models/
│   ├── baselines_gnn.py           # GNN architectures
│   ├── baselines_causal.py        # Causal modeling architectures
│   └── baselines_rl.py            # d3rlpy RL network definitions
├── evaluation/
│   └── run_ope_evaluation.py      # SNIPW and Hit Rate evaluation
├── visualization/                 # Plotting scripts for benchmark figures
│   ├── plot_results.py
│   ├── plot_ope_metrics.py
│   ├── plot_ablations.py
│   └── plot_integrations.py
└── README.md
```

## Dataset Access

This benchmark utilizes two large-scale public datasets. Please download and extract them to your local `data/` directory before running the preprocessing pipelines:
- **KuaiRec:** Available at [KuaiRec GitHub Repository](https://github.com/chongminggao/KuaiRec)
- **Open Bandit Dataset (OBD):** Available at [OBD GitHub Repository](https://github.com/st-tech/zr-obp)

## Setup & Installation

All experiments were conducted on an **AMD Ryzen 7 5700G** with **32 GB RAM** and an **NVIDIA RTX 5060 Ti (16 GB VRAM)**.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TasfinMahmud/graph-causal-rl-benchmark.git
   cd graph-causal-rl-benchmark
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Benchmark

The framework processes data out-of-core using `chunksize=1000000` to manage memory efficiently on large-scale datasets like KuaiRec (12.5M+ interactions).

### 1. Data Preprocessing
```bash
python data_processing/obd_preprocessing.py
python data_processing/kuairec_preprocessing.py
```

### 2. Graph Neural Network (GNN) Embeddings
```bash
python scripts/train_gnn_embeddings.py
```

### 3. Causal Uplift Modeling
To run the CATE estimations using the EconML framework:
```bash
python scripts/train_causal_uplift.py
```

### 4. Offline RL Training
To train the offline reinforcement learning agents (BCQ, CQL, IQL):
```bash
python scripts/train_offline_rl.py
```
*Note: This script dynamically computes the Self-Normalized Inverse Propensity Weighting (SNIPW) metrics directly from the testing set to ensure robust Offline Policy Evaluation.*

### 5. Evaluation & Visualization
```bash
python evaluation/run_ope_evaluation.py
python visualization/plot_results.py
```

## Citation
Citation details will be provided upon publication.
