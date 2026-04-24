# RCP-DTA: Representation-Aware Conformal Prediction for Reliable Uncertainty Quantification in Drug-Target Affinity

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper: **"Representation-Aware Conformal Prediction for Reliable Uncertainty Quantification in Drug-Target Affinity"**.

---

## 📖 Introduction

Predicting Drug-Target Affinity  is a cornerstone of computer-aided drug discovery. However, standard deep learning models often provide point estimates without a measure of confidence. 

**RCP-DTA** addresses this by integrating multi-modal representation learning with **Conformal Prediction**. Our framework provides:

* **Point Predictions**: Highly accurate affinity scores.
* **Uncertainty Intervals**: Mathematically guaranteed prediction intervals with user-defined confidence levels (e.g., 95%).

### Key Methodology

1. **Hybrid Feature Extraction**: Combines **Dual-stream GNNs** (`MolGNNEncoder` and `ProGNNEncoder`) for structural information with **Pre-trained Models** (**Uni-mol** for SMILES and **ESM-2** for protein sequences) to capture deep semantic representations.
2. **Representation-Aware Fusion**: Utilizes `UniCrossAttention` and `SSFusion` (Gate Fusion) to align and merge drug-target features effectively.
3. **Adaptive Prediction**: Employs **Local Quantile Mapping** within the Conformal Prediction framework to generate sample-specific uncertainty estimates.

---

## 🏗 Framework Architecture

The architecture consists of three main modules: Representation Learning, Feature Fusion, and Adaptive Uncertainty Prediction.

![Model Architecture](./Figure%201.jpg)

---

## 🛠 Installation

### Prerequisites

* **OS**: Linux  or Windows 10/11
* **Python**: 3.7.16 (Ensures compatibility with PyTorch 1.13.1 and specific GNN libraries)
* **GPU**: NVIDIA GPU with CUDA 11.7 support is highly recommended for GNN training.

### Core Environment Dependencies
The project relies on the following key frameworks:
  * `torch==1.13.1+cu117`
  * `torch-geometric==2.3.1` 
  * `pytorch-lightning==1.9.5`
  * `rdkit==2023.3.2`
  * `biopython==1.81`
  * `dgl==1.1.3`
  * `networkx==2.6.3`
  * `transformers==4.21.3`
  * `fair-esm==2.0.0`
### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Lee515628/RCP-DTA.git
cd RCP-DTA

# 2. Create a new conda environment
conda create -n RCPDTA python=3.7
conda activate RCPDTA

# 3. Install required Python packages
pip install -r requirements.txt
```

## 🚀 Usage Guide

### 1. Feature Extraction
The framework utilizes specialized featurizers to convert raw data into graph representations:
* **Protein Graphs**: Processed via `src/featurizers/proteingraph.py`.
* **Molecular Graphs**: Processed via `src/featurizers/molgraph.py`.

### 2. Training the Model
You can train the model on specific datasets using the `main.py` script. The `--r` flag specifies the run index or configuration.

**For Davis Dataset:**
```bash
python main.py --task Davis --r 0
```
**For KIBA Dataset:**
```bash
python main.py --task KIBA --r 0
```

**Running Multi-Seed Experiments**
```bash
bash training.sh
```

**Running RCP**
```bash
bash runcp.sh
```

---

## 📂 Directory Structure
```text
RCP-DTA/
├── dataset/
│   ├── Davis/          
│   └── KIBA/            
├── src/
│   ├── featurizers/
│   │   ├── proteingraph.py
│   │   ├── molgraph.py
│   │   ├── molecule.py
│   │   ├── protein.py  
│   │   └── molgraph.py
│   ├── model.py          
│   ├── main.py
│   ├── data.py
│   ├── moleculegnn.py     
│   └── proteingnn.py
├── training.sh             
├── Radam.py
├── dynamicCP.py
├── Lookahead.py
├── train_test.py
├── utils_dta.py
├── runcp.sh                
└── requirements.txt     
```
