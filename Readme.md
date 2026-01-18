# Morphology-Aware Self-Supervised Vision Transformer for Breast Cancer Diagnosis 🔬

![Python](https://img.shields.io/badge/Python-3.9-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![Accuracy](https://img.shields.io/badge/Accuracy-94.5%25-green)

## 📌 Project Overview
This project introduces a **Morphology-Aware Self-Supervised Learning (SSL)** framework for classifying Breast Cancer Histopathology images. Unlike traditional CNNs that require massive labeled datasets, this system uses a **Vision Transformer (ViT-Tiny)** pre-trained with a novel **Pseudo-Mask Guidance** mechanism to learn cellular structures without human annotation.

## 🚀 The Novelty (Key Innovation)
Standard SSL (like SimCLR) learns texture. Our approach forces the model to learn **Morphology**:
1.  **Unsupervised Mask Generation:** We use Color Deconvolution (Otsu Thresholding) to separate nuclei from the background automatically.
2.  **Morphology-Decoder:** The Transformer is penalized if it cannot reconstruct the nuclei mask, forcing it to focus on biological structures.

## 📊 Results
*   **Dataset:** BreaKHis 400X (Histopathology)
*   **Accuracy:** **94.5%**
*   **Recall (Malignant):** **98%** (Highly sensitive to cancer detection)
*   **Technique:** ViT-Tiny + DINO-style SSL + Morphological Reconstruction

## 🛠️ Installation
1. Clone the repo:
   git clone https://github.com/sinchanarv/Morphology-Aware-Self-Supervised-Vision-Transformer-for-Breast-Cancer-Diagnosis-.git
   
2. Install dependencies:
    pip install -r requirements.txt

## 💻 Usage
Run the Diagnostic GUI:
    python -m streamlit run src/app.py

This will launch a web interface where you can upload histopathology slides (PNG/JPG) and get a real-time diagnosis with Explainable AI heatmaps.