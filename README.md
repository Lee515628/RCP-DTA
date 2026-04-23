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

1. **Feature Extraction**: Dual-stream GNNs (`MolGNN` and `ProGNN`) capture complex structural information from SMILES and protein sequences.
2. **Representation-Aware Fusion**: Utilizes `UniCrossAttention` and `SSFusion` (Gate Fusion) to align and merge drug-target features effectively.
3. **Adaptive Prediction**: Employs **Local Quantile Mapping** within the Conformal Prediction framework to generate sample-specific uncertainty estimates.

---

## 🏗 Framework Architecture

The architecture consists of three main modules: Representation Learning, Feature Fusion, and Adaptive Uncertainty Prediction.

![Model Architecture](./Figure%201.jpg)

---

## 🛠 Installation

### Prerequisites

* **OS**: Windows 10/11 or Linux
* **Environment**: Python 3.7+ 

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

### 🚀 Getting Started
