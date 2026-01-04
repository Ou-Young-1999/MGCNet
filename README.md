# [MGCNet]

Robust and Generalizable Atrial Fibrillation Detection from ECG Using Time-Frequency Fusion and Supervised Contrastive Learningion

This repository contains the official implementation of the paper  

---

## ✨ Features

- ✅ Full reproduction of the proposed method in the paper  
- 🧪 Supports training and test on **MIT-BIH-AF**  (https://www.physionet.org/content/afdb/1.0.0/)
- 📦 Pre-trained models available  
- ⚙️ Clean, modular, and easy-to-extend codebase  

---

## 📁 Repository Structure

```bash
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
```


## 🛠️ Environment Setup

We recommend using Conda or a virtual environment:

```bash
conda create -n myenv python=3.13.9
conda activate myenv
pip install -r requirements.txt
💡 Tip: For full reproducibility, specify exact versions (e.g., torch==2.7.1+cu126).
```

📥 Data Preparation

Download the preprocessed dataset MIT-BIH-AF, checkpoints and results:
(https://pan.baidu.com/s/1GuOvJJgD3hEXahxgcvr-Eg?pwd=x2ic)

▶️ Quick Start

Train the model
```bash
python train.py --config ./config/bgm/bgm_afdb_1.yaml
```
Test the model
```bash
python test.py --config ./config/bgm/bgm_afdb_1.yaml
```

