# Robust Medical Vision — Phase 2

Uncertainty-aware skin lesion classifier built on HAM10000.

## Project Structure

```
robust_medical_vision/
├── dataset/             # The actual medical images and CSV (git-ignored)
├── DL/
│   ├── data/            # dataset.py (splitting, augmentation, PyTorch Dataset)
│   ├── models/          # architecture.py, losses.py, trainer.py
│   ├── utils/           # evaluation.py (Grad-CAM, calibration, uncertainty)
│   ├── notebooks/       # The core execution pipeline:
│   │   ├── 00_setup.ipynb
│   │   ├── 01_eda.ipynb
│   │   ├── 02_split_and_dataloader.ipynb
│   │   ├── 03_build_model.ipynb
│   │   ├── 04_loss_functions.ipynb
│   │   ├── 05_train.ipynb
│   │   ├── 06_evaluate.ipynb
│   │   └── 07_ablation.ipynb
│   ├── outputs/         # Generated plots and checkpoints (auto-created)
│   ├── requirements.txt
│   └── README.md
```

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

1. Download HAM10000 from Kaggle:
   https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection

2. Merge both image folders into one inside the `DL/dataset/images` directory:
```bash
mkdir -p DL/dataset/images
cp HAM10000_images_part_1/*.jpg DL/dataset/images/
cp HAM10000_images_part_2/*.jpg DL/dataset/images/
```

3. (Optional) Download NIH CXR14 chest X-rays for OOD testing:
   https://nihcc.app.box.com/v/ChestXray-NIHCC

## Running

This project is executed sequentially via Jupyter Notebooks to allow for deep interactive analysis at every step of the pipeline.

Simply launch Jupyter and run the notebooks in order:
```bash
jupyter notebook
```

**Workflow:**
1. `00_setup.ipynb` - Validates your dataset and environment.
2. `01_eda.ipynb` - Generates data visualizations.
3. `02_...` to `04_...` - Prepares loaders, architectures, and losses.
4. `05_train.ipynb` - The main training loop (set `QUICK_MODE = False` for full 30-epoch training).
5. `06_evaluate.ipynb` - Generates the final metrics, confusion matrix, calibration curve, and Grad-CAMs.
6. `07_ablation.ipynb` - Advanced tests.

## Output Files

| File | What it shows |
|------|--------------|
| `eda_1_class_distribution.png` | Class imbalance — justifies F1 metric |
| `eda_2_metadata_analysis.png` | Age/sex/localization patterns |
| `eda_3_sample_images.png` | Visual similarity between classes |
| `eda_4_pixel_statistics.png` | ImageNet normalization validity |
| `eda_5_lesion_duplicates.png` | Data leakage justification |
| `training_history.png` | Train/val loss, F1, AUROC curves |
| `gradcam_per_class.png` | Where the model looks — interpretability |
| `calibration_curves.png` | Are confidence scores trustworthy? |
| `uncertainty_analysis.png` | Does uncertainty track difficulty? |
| `ood_detection.png` | Clinical safety evaluation |
| `confusion_matrix.png` | Per-class performance breakdown |
| `best_model.pth` | Best model checkpoint |
| `test_metrics.json` | Final F1, AUROC numbers |

## Architecture Summary

```
Input (224×224×3)
    ↓
EfficientNet-B1 [pretrained, fine-tuned]
    ↓
Dense(512) + BatchNorm + GELU
    ↓
MCDropout(p=0.4)  ← always active → epistemic uncertainty
    ↓
Dense(256) + GELU
    ↓
MCDropout(p=0.3)  ← always active
    ↓
┌────────────────────────┐
│ Standard Head (7 logits)│  → Focal Loss
└────────────────────────┘
┌────────────────────────┐
│ Evidential Head (7 α)   │  → Evidential Loss → aleatoric + epistemic uncertainty
└────────────────────────┘
```

## Key Design Decisions

| Decision | Why |
|----------|-----|
| EfficientNet-B1 not B3 | Fits M2 8GB RAM; B3 would crash |
| Batch size 16 + gradient accumulation | Memory-safe, mathematically equiv to 32 |
| MC Dropout (20 passes) | Bayesian approximation of epistemic uncertainty |
| Evidential head | Separates aleatoric vs epistemic uncertainty |
| Focal Loss γ=2 | Down-weights easy nevus samples, focuses on rare melanoma |
| GroupShuffleSplit | Prevents same lesion in train+test (data leakage) |
| Two-stage training | Prevents catastrophic forgetting of ImageNet features |
| WeightedRandomSampler | Ensures rare classes seen equally during training |
