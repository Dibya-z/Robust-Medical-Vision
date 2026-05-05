# Robust Medical Vision: Uncertainty-Aware Classifiers for Pulmonary Diagnosis

## Project Overview
This project establishes a foundational methodology for an AI product designed to diagnose chest X-rays (NORMAL vs. PNEUMONIA) safely. Instead of relying on typical "black-box" neural networks, this project emphasizes **uncertainty-aware modeling**. The goal is to build a robust pipeline that not only predicts accurately but can handle anomalous inputs, rare conditions, and out-of-distribution (OOD) cases by understanding and broadcasting its lack of confidence.

## Dataset
This pipeline operates on the **Chest X-Ray Pneumonia** dataset (widely available on Kaggle). The data contains lung radiographs categorized into Normal and Pneumonia classes.

## Features & Methodology
1. **Preprocessing (`ML/data/data_loader.py`):**
   - Radiographs are loaded in grayscale.
   - Images are resized to a static $128 \times 128$ spatial frame.
   - Pixel intensities are normalized to `[0.0, 1.0]`.

2. **Feature Engineering (`ML/utils/features.py`):**
   - **Histograms of Oriented Gradients (HOG):** Maps structural gradient geometry, capturing the abnormal internal borders formed by fluid consolidations.
   - **Local Binary Patterns (LBP):** Describes micro-textural patterns. It distinguishes between the uniform texture of healthy lungs and the irregular textures caused by infections.
   - The features are concatenated to form a dense representation of the X-ray characteristics.

3. **Modeling & Confidence Calibration (`ML/utils/calibration.py`):**
   - Rather than deep learning, this project implements interpretable classical machine learning algorithms (Support Vector Machines, Logistic Regression, K-Nearest Neighbors).
   - A major component is **Calibration**: We compute the Expected Calibration Error (ECE) and plot Reliability Diagrams. This ensures that the model's predicted probability directly correlates with the true likelihood of the prediction being correct, minimizing overconfidence.

## Project Structure
```
Robust-Medical-Vision/
├── ML/                 # Classical machine learning pipeline
│   ├── data/           # Data loading and processing scripts (e.g., data_loader.py)
│   ├── dataset/        # Directory for the chest X-ray image dataset
│   ├── models/         # Exported machine learning models
│   ├── notebooks/      # Exploratory Data Analysis and modeling workflows
│   ├── outputs/        # Generated plots and evaluation metrics
│   ├── report/         # LaTeX source code for the project report
│   ├── utils/          # Core processing scripts (calibration.py, features.py)
│   └── requirements.txt# Python dependencies for ML pipeline
├── DL/                 # Deep learning pipeline
│   ├── data/           # Data processing and EDA scripts
│   ├── dataset/        # Deep learning datasets and metadata
│   ├── models/         # Exported PyTorch models
│   ├── notebooks/      # Deep learning training and evaluation workflows
│   ├── outputs/        # Generated plots and evaluation metrics
│   ├── utils/          # Helper scripts (evaluation.py)
│   └── requirements.txt# Python dependencies for DL pipeline
```

## Setup and Installation
1. Clone the repository.
2. (Optional) Create a virtual environment.
3. Install the required dependencies:
   ```bash
   pip install -r ML/requirements.txt
   ```
4. Run the notebooks in the `ML/notebooks/` directory sequentially to see the EDA, feature extraction, and calibration implementations in action.
