# 🧠 NeuroNet-AD: A Multimodal Deep Learning Framework for Multiclass Alzheimer’s Disease Diagnosis

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview
Alzheimer’s Disease (AD) is the most common form of dementia, severely impacting cognitive functions and quality of life. Early and accurate diagnosis, especially at the stage of *Mild Cognitive Impairment (MCI)*, is crucial for patient care and treatment development.

**NeuroNet-AD** is a novel multimodal deep learning framework that integrates:
- **MRI images** (structural brain scans)  
- **Clinical metadata** (psychological test scores, demographics, genetic biomarkers)  

to achieve **multiclass classification** of Normal Control (NC), MCI, and AD with high accuracy:contentReference[oaicite:1]{index=1}.

---

## Key Features
- **Convolutional Block Attention Module (CBAM):** Enhances ResNet-18 image backbone by focusing on informative spatial & channel-wise features.  
- **Meta Guided Cross Attention (MGCA):** Novel attention mechanism for effective cross-modal fusion of imaging & clinical metadata.  
- **Ensemble-based Feature Selection:** Combines Random Forest, XGBoost, LightGBM, ExtraTrees, and AdaBoost to select the most discriminative clinical features.  
- **Comprehensive Evaluation:**  
  - Subject-level **5-fold cross-validation** and **independent held-out test set** on ADNI dataset.  
  - **External validation** on OASIS-3 dataset for generalizability.  

---

## Performance Highlights
- **ADNI1 Dataset:**  
  - 98.68% accuracy (NC vs. MCI vs. AD)  
  - Outperforms state-of-the-art models such as MADDi, MedBLIP, and hybrid CNN approaches:contentReference[oaicite:2]{index=2}  
- **External OASIS-3 Validation:**  
  - 94.10% accuracy despite demographic and acquisition variability  
- **Ablation Studies:** Each component (CBAM, MGCA, feature selection, text encoder) demonstrated significant improvements in model performance.

---

## Architecture
The NeuroNet-AD pipeline consists of four main modules:
1. **ResNet-18 + CBAM** → Extracts refined image features.  
2. **Text Encoder (BERT)** → Generates embeddings for clinical metadata.  
3. **MGCA (Meta Guided Cross Attention)** → Aligns image and text features via multi-head attention.  
4. **Classifier Layer** → Produces multiclass predictions (NC, MCI, AD).  

---

## Dataset
- **Training/Validation:** [ADNI1 Dataset](https://adni.loni.usc.edu/) – 200 subjects (2000 MRI slices + clinical metadata).  
- **External Validation:** [OASIS-3 Dataset](https://www.oasis-brains.org/) – 921 images (704 NC, 19 MCI, 198 AD).  

*Note: Access to ADNI/OASIS data requires registration and approval.*  

---

## Implementation
- **Framework:** PyTorch  
- **Optimizer:** Adam (lr=0.001, weight decay=1e-5)  
- **Loss Function:** Cross-Entropy (multiclass)  
- **Regularization:** Dropout (p=0.5), early stopping (patience=20)  
- **Augmentation:** Random flips, ±10° rotations, intensity scaling  

---

## Results
| Model         | Dataset | Modality | Accuracy (%) |
|---------------|---------|----------|--------------|
| MedBLIP       | OASIS   | 3D Images + Text | 85.3 |
| MADDi         | ADNI    | Imaging + Clinical + Genetic | 96.88 |
| Hybrid CNN    | ADNI    | MRI + Clinical | 98.4 (binary) |
| **NeuroNet-AD** | ADNI    | MRI + Clinical Metadata | **98.68 (multiclass)** |
| **NeuroNet-AD** | OASIS-3 | MRI + Clinical Metadata | **94.10 (multiclass)** |

---

## Citation
If you use this work, please cite:  

```bibtex
@article{rahman2025neuronetad,
  title   = {NeuroNet-AD: A Multimodal Deep Learning Framework for Multiclass Alzheimer’s Disease Diagnosis},
  author  = {Rahman, Saeka and Rahman, Md Motiur and Bhatt, Smriti and Sundararajan, Raji and Faezipour, Miad},
  journal = {Bioengineering},
  year    = {2025},
  doi     = {10.3390/bioengineering1010000}
}
