# Robust Medical Vision: Enhancing Skin Lesion Classification with Uncertainty Estimation and Conformal Prediction

## Project Overview
Deep learning models have shown remarkable accuracy in medical image classification; however, standard neural networks are often overconfident and lack the ability to express uncertainty or identify out-of-distribution (OOD) inputs. 

This repository implements a **clinical-grade, uncertainty-aware diagnostic system** (named **DermaSense AI**) for skin lesion classification. Using a three-stage ablation framework, the project demonstrates how to transition a standard "black-box" classifier into a safe, reliable, and calibrated clinical assistant that "knows what it does not know."

---

## Dataset: HAM10000
The pipeline operates on the **HAM10000** ("Human Against Machine") dataset, which contains **10,015 multi-source dermatoscopic images** categorized into 7 classes:
1. **nv** - Melanocytic Nevus (Benign)
2. **mel** - Melanoma (Malignant)
3. **bkl** - Benign Keratosis (Benign)
4. **bcc** - Basal Cell Carcinoma (Malignant)
5. **akiec** - Actinic Keratosis / Bowen's Disease (Precancerous)
6. **vasc** - Vascular Lesion (Benign)
7. **df** - Dermatofibroma (Benign)

### Class Imbalance & Data Leakage Protection
* **Imbalance:** The dataset exhibits a severe 58:1 ratio between the dominant class (`nv`) and the rarest classes.
* **Leakage Prevention:** A patient-grouped split (`GroupShuffleSplit`) ensures that different images from the same patient do not leak across the train (6,959), validation (1,529), and test (1,527) sets.

---

## The Three-Phase Ablation Framework

### Phase 1 (Model A): Classical ML Baseline
* **Feature Extraction:** Extracts 104 hand-crafted textural and color features:
  * **GLCM** (Gray-Level Co-occurrence Matrix) for macroscopic texture.
  * **LBP** (Local Binary Patterns) for micro-texture.
  * **HSV/RGB Histograms & Moments** for color distribution.
* **Classifier:** Dimensionality reduction via PCA (50 components retaining 98.24% variance) followed by a **Gaussian Process (GP) Classifier** (One-vs-Rest strategy).
* **OOD Detection:** A baseline **Isolation Forest** trained on the hand-crafted feature space.

### Phase 2 (Model B): Deep Learning Pipeline
* **Backbone:** ImageNet-pretrained **EfficientNet-B3** backbone.
* **Training Protocol:** Two-stage training (frozen backbone first, then full unfreezing) optimizing a **Combined Loss** integrating Focal Loss, Label Smoothing, and Class Weights to mitigate imbalance.
* **Uncertainty Mechanisms:**
  * **Monte Carlo (MC) Dropout:** Dropout layers stay active during inference. 20 forward passes yield predictive variance representing epistemic uncertainty.
  * **Evidential Deep Learning:** The model outputs parameters of a Dirichlet distribution rather than raw softmax, explicitly quantifying vacuity (lack of evidence) for OOD samples.
* **Post-hoc Calibration:** Validated using **Temperature Scaling** (calibrated to $T = 1.500$), reducing Expected Calibration Error (ECE) from 0.220 to 0.089.

### Phase 3 (Model C): Hybrid Safe AI
* **Dual OOD Detection:** Integrates Model B's Mahalanobis distance detector (calculating distances in deep feature space) with Model A's GP Isolation Forest via a union rule.
* **Conformal Prediction:** Employs Regularized Adaptive Prediction Sets (RAPS) to calibrate a non-conformity threshold ($\hat{q} = 0.984$). Rather than a single prediction, it outputs a set of classes mathematically guaranteed to contain the true diagnosis at a target coverage level (e.g., $\ge 95\%$).

---

## Technical Results

| Model | F1 Macro | AUROC | ECE | OOD Flagged |
| :--- | :---: | :---: | :---: | :---: |
| **Model A (ML Baseline)** | 0.3488 | 0.8616 | -- | 4.58% |
| **Model B (DL Pipeline)** | 0.5687 | 0.9165 | 0.0890 | 5.50% |
| **Model C (Hybrid System)** | **0.5687** | **0.9165** | **0.0890** | **10.09%** |

* **Accuracy Boost:** 63% relative F1-Macro improvement from classical ML to DL. Model A completely failed on rare classes (`vasc`, `df`), whereas Model B successfully resolved them.
* **Uncertainty Correlation:** Wrong predictions are **1.79×** more uncertain on average than correct ones, enabling a reliable safety-deferral threshold.
* **Statistical Guarantees:** Model C achieved an actual overall test set coverage of **95.81%** (achieving **100%** on high-risk Melanoma and Basal Cell Carcinoma cases) with an average prediction set size of 3.69 classes.
* **Dual OOD:** Combining Mahalanobis and Isolation Forest doubled the detection of abnormal/out-of-distribution inputs to **10.09%**.

---

## Project Directory Structure
```
Robust-Medical-Vision/
├── Final/              # Deep Learning & Calibration Experiments
│   ├── data/           # Split dataset loaders and patient groups
│   ├── models/         # PyTorch architectures (evidential, dropout)
│   ├── notebooks/      # Chronological training, calibration, and conformal notebooks
│   ├── outputs/        # Saved checkpoints, scalers, and result plots
│   └── report/         # IEEE Conference Report source code (phase3_report.tex)
├── ML/                 # Classical ML Experiments
│   ├── data/           # Data loader for classical features
│   ├── utils/          # GLCM, LBP, and HSV feature extractors
│   └── notebooks/      # Exploratory Data Analysis & baseline modeling
├── web/                # Local Web Application
│   ├── backend/        # FastAPI application (main.py, loads models for inference)
│   └── frontend/       # React + Vite client (interactive panels, Grad-CAM, & metrics)
```

---

## Local Setup & Quick Start

Both backend and frontend dependencies are pre-installed in the workspace. Follow these steps to run the interactive dashboard.

### 1. Run the FastAPI Backend
Open a terminal and execute:
```bash
cd web/backend
source .venv/bin/activate
python main.py
```
*The backend runs at `http://localhost:8000`. Note that loading the GP model (`model_a_gp.pkl` ~229MB) and PyTorch models takes 30-60s on first startup.*

### 2. Run the React Frontend
Open a second terminal window and execute:
```bash
cd web/frontend
npm run dev
```
*The frontend runs at `http://localhost:5173`. Open this URL in your web browser.*

### Interactive Features to Explore
* **Analyze Tab:** Upload skin images, select a Safety Phase (1, 2, or 3), and view predictions, Grad-CAM attention maps, conformal prediction sets, and OOD warnings.
* **Research Tab:** Explore live-rendering interactive charts for Temperature Scaling calibration, ablation metrics, and safety correlations.
* **Pipeline Tab:** Walk through the HAM10000 data specs and technical milestones.