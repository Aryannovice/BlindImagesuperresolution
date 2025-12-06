# Blind Image Super-Resolution (BISR)

This repository contains the implementation of a **Blind Image Super-Resolution** framework based on a **dual-path architecture** that combines **U-Net, CNN, and ResNet** components. The model is designed to handle unknown degradations (blur, noise, compression artifacts) and reconstruct high-quality HR (high-resolution) images from LR (low-resolution) inputs.

> 📝 **Note:**  
> The **project report / IEEE conference paper submission** has been uploaded to this repository.  
> The **codebase is being gradually cleaned and uploaded**. Expect incremental updates to the `src/` and `experiments/` folders.

For any issues, questions, or collaboration requests, feel free to reach out at:  
📧 **ayushpandey1177@gmail.com**

---

## 🔍 Project Overview

Traditional super-resolution models assume that the degradation process (downsampling kernel, noise level, compression) is known or fixed. In practical scenarios, this assumption fails, leading to poor performance on real-world images.

This project tackles **blind image super-resolution**, where:

- The degradation type and intensity are **not known in advance**.
- The model must learn to **infer and compensate for degradations** implicitly.
- The architecture uses a **dual-path design**:
  - One path focuses on **structure and content reconstruction** (U-Net / CNN branch).
  - The other emphasizes **high-frequency detail and texture recovery** (ResNet-style branch).

An additional **unlearning-based agentic parameter** is introduced to adaptively suppress artifact-prone features and improve generalization across diverse degradation settings.

---

## 🏗️ Architecture Highlights

- **Dual-Path Network**
  - **Content Path**: U-Net style encoder–decoder with skip connections to preserve global structure.
  - **Detail Path**: Residual blocks for fine-grained texture and high-frequency detail reconstruction.
- **Unlearning-Based Agentic Component**
  - Learns to reduce hallucinated patterns and artifacts via an auxiliary loss.
  - Encourages robust reconstruction under unknown degradations.
- **Multi-Scale Fusion**
  - Combines features from different resolutions.
  - Balances sharpness and stability in reconstructed images.
- **Loss Functions (planned/implemented)**
  - L1 / L2 Reconstruction Loss  
  - Perceptual Loss (VGG-based)  
  - SSIM-based Structural Loss  
  - Regularization for the unlearning/agentic term

---

## 📂 Repository Structure

> ⚠️ **Work in Progress:** Code is being pushed in stages. Some folders may be placeholders until the corresponding modules are added.

```text
blind-image-super-resolution/
├── README.md                  # Project description and usage
├── report/                    # Project report / IEEE conference submission (PDF, figures, etc.)
├── src/                       # Core model and training code (WIP)
│   ├── models/                # Model definitions (U-Net, ResNet blocks, dual-path wrapper)
│   ├── datasets/              # Dataset loaders and preprocessing utilities
│   ├── utils/                 # Helper functions (metrics, logging, config)
│   ├── train.py               # Training pipeline (coming soon)
│   └── infer.py               # Inference script / demo (coming soon)
├── experiments/               # Configs, logs, and experiment scripts (WIP)
├── requirements.txt           # Python dependencies (will be updated)
└── LICENSE                    # License (to be added if applicable)
