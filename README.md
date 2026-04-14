# 🏭 Few-Shot Parameter-Efficient Synthesis of Industrial Textural Anomalies via Localized Diffusion and DoRA

[![Hugging Face Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue.svg)](https://huggingface.co/akarshkunwar/PEFT-Industrial-defect-synthesis)
[![Hugging Face Datasets](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-blue.svg)](https://huggingface.co/datasets/akarshkunwar/synthetic-dataset-industrial-defect-synthesis)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Welcome to the official repository for **Few-Shot Parameter-Efficient Synthesis of Industrial Textural Anomalies via Localized Diffusion and DoRA**. 

This repository contains the codebase for training and evaluating Weight-Decomposed Low-Rank Adaptation (DoRA) models utilizing Stable Diffusion v1.5 Inpainting. Our localized diffusion pipeline generates high-fidelity industrial **textural anomalies** (e.g., surface stains, discoloration, fabric defects) using only a few training examples. These synthetic defects act as a powerful data augmentation strategy to improve downstream anomaly detection models like YOLO and patchcore.

---

## 🤗 Pre-Trained Models & Weights
To keep this Git repository lightweight and clean, **all trained DoRA adapters, model checkpoints, and our synthetized datasets are hosted on Hugging Face.**

You can download our parameter-efficient weights here:
👉 **[Hugging Face Model Hub: PEFT-Industrial-defect-synthesis](https://huggingface.co/akarshkunwar/PEFT-Industrial-defect-synthesis)**

You can view our synthesized data here:
👉 **[Hugging Face Dataset Hub: synthetic-dataset-industrial-defect-synthesis](https://huggingface.co/datasets/akarshkunwar/synthetic-dataset-industrial-defect-synthesis)**

---

## 📢 Updates & Ongoing Research
**Current Focus:** The current pipeline is optimized exclusively for **textural defects**. 

**Future Work:** This repository is actively maintained. Our research is rapidly expanding to include structural anomalies, new material surfaces, and advanced generation techniques. Star and watch this repository to stay updated as new models and defect classes are added!

---

## ⚙️ Quickstart & Usage

### 1. Installation
Clone the repository and set up your Python environment:
```bash
git clone [https://github.com/akarshkunwar/Industrial-defect-synthesis.git](https://github.com/akarshkunwar/Industrial-defect-synthesis.git)
cd Industrial-defect-synthesis
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
