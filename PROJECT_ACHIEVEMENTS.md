# Robust Medical Vision — Project Achievements & Resume Guide

> **Uncertainty-Aware Deep Learning for Safe Medical Image Diagnosis**
> A full-stack, research-grade system that doesn't just classify skin lesions — it knows *when it doesn't know*, gives statistically-guaranteed answers, and explains its reasoning.

**Product name:** DermaSense AI
**Domain:** Medical Computer Vision · Trustworthy / Safety-Critical AI
**Scope:** End-to-end — data engineering → classical ML → deep learning → uncertainty quantification → explainability → deployed full-stack web app

---

## 1. Executive Summary — What We Built

A **7-class skin-cancer classifier** built on the **HAM10000** dermatoscopy dataset that is engineered around a single clinical principle: *a medical AI must broadcast its uncertainty instead of being confidently wrong.*

Rather than shipping one black-box model, the project delivers a **3-model ablation pipeline** that progressively layers safety mechanisms, plus a **deployed web application (DermaSense AI)** that runs live inference with visual explanations and statistically-guaranteed prediction sets.

| Phase | Model | What It Adds | F1 (macro) | AUROC | ECE |
|-------|-------|--------------|:---------:|:-----:|:---:|
| **A** | Classical ML (Gaussian Process + Isolation Forest) | Interpretable baseline + OOD rejection | 0.349 | 0.862 | — |
| **B** | Deep Learning (EfficientNet-B3) | MC-Dropout + Evidential uncertainty + calibration | **0.569** | **0.916** | **0.089** |
| **C** | Hybrid "Safe AI" | + Conformal Prediction (95% coverage guarantee) | 0.569 | 0.916 | 0.089 |

**Headline results**
- **Cut calibration error (ECE) by ~60%** — from ~0.22 (uncalibrated baseline) to **0.089** via temperature scaling + evidential training.
- **AUROC 0.92** on a brutal **58:1 class-imbalanced**, 7-class medical problem.
- **Formal 95% coverage guarantee** through distribution-free Conformal Prediction (avg. set size 3.67, 5.3% confident singletons).
- **Validated uncertainty signal** — the model's uncertainty is ~2× higher on *wrong* predictions than on *correct* ones (0.0035 vs 0.0019), proving the uncertainty is meaningful and not noise.
- **4 complementary uncertainty methods** fused into one OOD-aware decision system.

---

## 2. Tech Stack

### Machine Learning & Deep Learning
- **Languages:** Python 3
- **Deep Learning:** PyTorch, torchvision (transfer learning with **EfficientNet-B1 / B3**, ImageNet-pretrained)
- **Classical ML:** scikit-learn — **Gaussian Process Classifier**, **Isolation Forest** (OOD), PCA, StandardScaler, Empirical Covariance
- **Computer Vision / Feature Engineering:** scikit-image (**HOG, LBP, GLCM**), OpenCV (color spaces, Grad-CAM overlays)
- **Numerics & Data:** NumPy, Pandas, SciPy

### Training Infrastructure & Techniques
- **Optimizer / Schedule:** AdamW, Cosine Annealing with Warm Restarts, discriminative (layer-wise) learning rates
- **Strategy:** two-stage transfer learning (frozen → fine-tuned backbone), gradient accumulation, gradient clipping, early stopping
- **Imbalance handling:** Focal Loss, Weighted Random Sampler, inverse-frequency class weighting
- **Hardware:** Apple **MPS (M2 GPU)** acceleration; cloud training via **Lightning AI**

### Uncertainty Quantification & Calibration (the core differentiator)
- **Monte Carlo Dropout** — epistemic uncertainty via 20-pass stochastic inference
- **Evidential Deep Learning** — Dirichlet head decomposing *aleatoric* vs *epistemic* uncertainty in a single pass
- **Temperature Scaling** — post-hoc confidence calibration (LBFGS-optimized)
- **Mahalanobis-distance OOD detection** — class-conditional Gaussians in deep feature space
- **Conformal Prediction (RAPS)** — distribution-free prediction sets with coverage guarantees
- **Metrics:** Expected Calibration Error (ECE), reliability diagrams, macro-F1, one-vs-rest AUROC

### Explainability
- **Grad-CAM** — visual heatmaps showing which lesion regions drove each prediction

### Full-Stack Web Application (DermaSense AI)
- **Backend:** FastAPI + Uvicorn — loads all 5 model artifacts at startup, serves real-time inference
- **Frontend:** React 19, Vite 8, React Router 7, Recharts (interactive charts), react-dropzone (image upload), lucide-react
- **API design:** REST endpoints for health, pre-computed metrics, and live multi-phase analysis

### Research & Reporting
- **Jupyter Notebooks** — reproducible, stage-by-stage workflows (EDA → training → calibration → conformal → ablation)
- **LaTeX (IEEE conference format)** — formal research report
- Git version control

---

## 3. Dataset

**HAM10000** ("Human Against Machine with 10,000 training images") — the benchmark dermatoscopy dataset from the **ISIC 2018** challenge.

- **10,015 dermatoscopic images** across **7 diagnostic classes**
- **Severe class imbalance: 58:1** (most common vs rarest class)

| Code | Diagnosis | Count | Clinical Risk |
|------|-----------|------:|---------------|
| `nv` | Melanocytic Nevus | 6,705 | Benign |
| `mel` | **Melanoma** | 1,113 | **Malignant** |
| `bkl` | Benign Keratosis | 1,099 | Benign |
| `bcc` | **Basal Cell Carcinoma** | 514 | **Malignant** |
| `akiec` | Actinic Keratosis | 327 | Pre-cancerous |
| `vasc` | Vascular Lesion | 142 | Benign |
| `df` | Dermatofibroma | 115 | Benign |

**Honest, leakage-free split** (train 6,959 / val 1,529 / test 1,527):
- Split by **`lesion_id`, not by image**, using scikit-learn `GroupShuffleSplit`. HAM10000 contains multiple photos of the *same* physical lesion — a naïve random split leaks the same lesion into both train and test and inflates results. Group splitting guarantees **zero lesion overlap** across train/val/test (asserted programmatically).

> **Note on project framing:** The methodology was first prototyped on a **Chest X-Ray Pneumonia** classical-vision formulation (see the IEEE report and `ML/report/`, using HOG + LBP features and ECE calibration). The final, fully-realized and deployed system is the **HAM10000 7-class dermatology** pipeline described here.

---

## 4. Data Preprocessing & Preparation

### Deep Learning pipeline
- **Resize** to 224×224 (EfficientNet input resolution)
- **ImageNet normalization** (mean/std matched to pretrained backbone statistics — critical for transfer learning)
- **Training augmentation** (each with a clinical rationale): horizontal + vertical flips, ±30° rotation, color jitter (brightness/contrast/saturation/hue), random resized crop (scale 0.8–1.0), random grayscale — forcing the model to learn lesion *shape and texture* rather than scanner-specific color artifacts
- **Clean eval transforms** — no augmentation on val/test, so metrics reflect real clinical conditions
- **Class imbalance** — `WeightedRandomSampler` with inverse-frequency weights so rare classes (e.g., 115 dermatofibroma) are seen as often as common ones
- **Metadata hygiene** — median-imputation of missing `age`, `unknown` fill for `sex`/`localization`, on-demand (lazy) image loading to respect an 8 GB RAM constraint

### Classical ML pipeline (Model A)
- Hand-crafted **104-dim** feature vector: **GLCM** (texture co-occurrence) + **LBP** (micro-texture) + **HSV/RGB color moments**
- `StandardScaler` → **PCA to 50 components** retaining **98.2% of variance**
- Feeds a Gaussian Process (One-vs-Rest) classifier with an Isolation Forest OOD gate

---

## 5. Methodology & Architecture

### Model A — Interpretable Classical Baseline
`GLCM + LBP + HSV (104-d)` → `StandardScaler` → `PCA(50)` → **Gaussian Process (OVR)** for calibrated probabilities + **Isolation Forest** for out-of-distribution rejection. Establishes an interpretable, uncertainty-aware floor without deep learning.

### Model B — Uncertainty-Aware Deep Network
A three-layer hybrid head on an **EfficientNet-B3** backbone:
- **Backbone:** ImageNet-pretrained EfficientNet (chosen over ResNet/VGG for compound-scaling parameter efficiency)
- **Uncertainty head:** custom always-on **MC Dropout** (epistemic uncertainty from 20-pass variance)
- **Evidential head:** **Dirichlet** output via softplus, decomposing aleatoric vs epistemic uncertainty (Subjective Logic framework)
- **Loss:** `Focal Loss + λ · Evidential Loss` — Focal handles imbalance (down-weights easy nevus, focuses on rare melanoma); Evidential penalizes *confidently wrong* predictions with KL annealing
- **Post-hoc:** Temperature Scaling for calibration + **Mahalanobis** OOD detector on 256-d features

### Model C — Hybrid "Safe AI"
Wraps Model B with a **union OOD signal** (deep + GP) and **Conformal Prediction (RAPS)**, converting point predictions into **prediction sets with a provable ≥95% coverage guarantee** — clinically actionable ("the lesion is melanoma *or* nevus → biopsy to disambiguate") and statistically defensible.

### DermaSense AI — Deployed Web App
Four-page React app served by a FastAPI backend:
- **Home** — landing page with live metrics and the 3-phase story
- **Analyze** — upload an image → real-time inference, top-k probabilities, uncertainty scores, OOD flag, **Grad-CAM overlay**, and conformal prediction set
- **Research** — interactive ablation, per-class F1, and safety charts (Recharts)
- **Pipeline** — dataset, architecture timeline, and key engineering decisions

---

## 6. Literature Review (Foundations Implemented)

This project is a faithful, hands-on implementation of landmark papers across uncertainty, calibration, and OOD detection:

| Area | Paper | Used For |
|------|-------|----------|
| Evidential uncertainty | **Sensoy et al., 2018** — *Evidential Deep Learning to Quantify Classification Uncertainty* (NeurIPS) | Dirichlet head + evidential loss |
| Calibration | **Guo et al., 2017** — *On Calibration of Modern Neural Networks* (ICML) | Temperature scaling, ECE |
| Conformal prediction | **Angelopoulos et al., 2021** — *Uncertainty Sets for Image Classifiers using Conformal Prediction* (ICLR) | RAPS prediction sets |
| OOD detection | **Lee et al., 2018** — *A Simple Unified Framework for Detecting OOD Samples* | Mahalanobis-distance detector |
| Class imbalance | **Lin et al., 2017** — *Focal Loss for Dense Object Detection* | Focal loss |
| Bayesian DL | **Gal & Ghahramani, 2016** — *Dropout as a Bayesian Approximation* | MC Dropout |
| Backbone | **Tan & Le, 2019** — *EfficientNet: Rethinking Model Scaling* | Compound-scaled backbone |
| Clinical AI safety | **Begoli et al., 2019**; **Kompa et al., 2021** — uncertainty in clinical ML | Problem motivation |
| Feature engineering | **Dalal & Triggs, 2005** (HOG); **Ojala et al., 2002** (LBP) | Classical features |

---

## 7. Key Engineering Decisions (Talking Points for Interviews)

- **Why group-based splitting?** Prevents the most common HAM10000 evaluation mistake — the same lesion leaking into train *and* test.
- **Why two-stage training?** Frozen-backbone warm-up lets the new head adapt before fine-tuning, avoiding catastrophic forgetting of ImageNet features.
- **Why macro-F1 and AUROC, not accuracy?** With 67% nevus, "always predict nevus" scores 67% accuracy while being clinically useless.
- **Why four uncertainty methods?** Each catches a different failure mode — MC-Dropout (model ignorance), Evidential (data ambiguity), Mahalanobis (feature-space novelty), Conformal (statistical coverage).
- **Why conformal prediction matters clinically?** A *set* prediction ("melanoma or nevus") is legally and clinically actionable; a bare "82% melanoma" is not.

---

## 8. Resume Bullet Points (Copy-Paste Ready)

**Pick the framing that fits the role. Numbers are taken directly from the project's result files.**

### Option A — ML / Deep Learning Engineer focus
- Built an **uncertainty-aware medical image classifier** (HAM10000, 7-class skin cancer, 10K images) on a fine-tuned **EfficientNet-B3**, reaching **0.92 AUROC** on a **58:1 class-imbalanced** dataset.
- **Reduced model calibration error (ECE) by ~60%** (0.22 → 0.089) by combining temperature scaling with an evidential (Dirichlet) training objective, making predicted confidences trustworthy.
- Implemented **four complementary uncertainty methods** — MC Dropout, Evidential Deep Learning, Mahalanobis OOD detection, and **Conformal Prediction with a provable 95% coverage guarantee**.
- Engineered a leakage-free `GroupShuffleSplit` (by lesion ID) and an imbalance-aware training stack (Focal Loss + weighted sampling + discriminative LRs + 2-stage transfer learning).

### Option B — Full-Stack / Applied AI focus
- Designed and shipped **DermaSense AI**, a full-stack diagnostic web app (**FastAPI + React 19/Vite**) serving real-time skin-lesion inference with **Grad-CAM explanations** and statistically-guaranteed prediction sets.
- Delivered a **3-model ablation pipeline** (classical ML → deep learning → hybrid safe-AI), improving macro-F1 by **+0.22** from the GP baseline to the deep model while adding formal safety guarantees.
- Productionized **5 model artifacts** behind a REST API (EfficientNet-B3, temperature scaler, Mahalanobis OOD, conformal predictor, Gaussian Process) with sub-second multi-phase inference.

### Option C — Research / single high-impact bullet
- Researched and implemented a **trustworthy clinical-AI pipeline** translating 8+ landmark papers (Evidential DL, Conformal Prediction, Temperature Scaling, Mahalanobis OOD) into a deployed system that **validated its uncertainty signal** (2× higher uncertainty on errors) and provided **distribution-free 95% coverage guarantees**.

### Skills to list (keywords)
`PyTorch` · `Deep Learning` · `Computer Vision` · `Transfer Learning (EfficientNet)` · `Uncertainty Quantification` · `Model Calibration` · `Conformal Prediction` · `Out-of-Distribution Detection` · `Gaussian Processes` · `scikit-learn` · `Grad-CAM / Explainable AI` · `FastAPI` · `React` · `Medical Imaging` · `Imbalanced Classification`

---

## 9. Project Structure

```
Robust-Medical-Vision/
├── ML/        # Phase A — classical ML (GLCM/LBP/HSV → PCA → GP + IsoForest) + IEEE LaTeX report
├── DL/        # Phase B — EfficientNet + MC-Dropout/Evidential training pipeline & notebooks
├── Final/     # Phase B/C — full pipeline: calibration, OOD, conformal prediction, ablation, artifacts
└── web/       # DermaSense AI — FastAPI backend + React/Vite frontend
```

---

*Generated as a portfolio/resume reference. All metrics are sourced from the project's own result files (`Final/outputs/*.json`, `phase3_complete_results.txt`).*
