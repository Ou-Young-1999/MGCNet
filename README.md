# [MGCNet]

> **One-sentence summary**: Synergizing Time and Frequency: A Cross-modal Deep Learning Approach for Atrial Fibrillation Detection

This repository contains the official implementation of the paper  

---

## ✨ Features

- ✅ Full reproduction of the proposed method in the paper  
- 🧪 Supports training and test on **MIT-BIH-AF**  
- 📦 Pre-trained models available  
- ⚙️ Clean, modular, and easy-to-extend codebase  

---

## 📁 Repository Structure
.
├── afdb_dataset/         # Dataset directory
├── checkpoint/           # Trained model weights
├── config/               # Configuration files
├── data/                 # Data loading and data augmentation
├── model/                # Model architecture definitions
├── preprocess/           # preprocessing
├── results/              # Test results or logs
├── requirements.txt      # Python dependencies
├── run_experiments.py    # One-click test
├── train.py              # Training script
├── test.py               # Test script
└── README.md

---

## 🛠️ Environment Setup

We recommend using Conda or a virtual environment:

```bash
conda create -n myenv python=3.13.9
conda activate myenv
pip install -r requirements.txt
💡 Tip: For full reproducibility, specify exact versions (e.g., torch==2.7.1+cu126).

📥 Data Preparation
Download the preprocessed dataset MIT-BIH-AF:

▶️ Quick Start
Train the model
bash
编辑
python train.py --config ./config/bgm/bgm_afdb_1.yaml
Test the model
bash
编辑
python test.py --config ./config/bgm/bgm_afdb_1.yaml
