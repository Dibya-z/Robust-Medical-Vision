#!/bin/bash
# =============================================================================
# LIGHTNING AI SETUP SCRIPT
# =============================================================================
# Run this ONCE at the start of every Lightning AI session:
#   bash setup_lightning.sh
#
# WHY: Lightning AI resets pip packages when the session pauses.
# This script reinstalls everything and verifies the environment.
# Takes ~3-4 minutes on first run, ~1 minute on subsequent runs.
# =============================================================================

set -e   # exit on first error

echo "============================================"
echo " Robust Medical Vision — Lightning AI Setup"
echo "============================================"

# ── 1. PyTorch with CUDA 11.8 ─────────────────────────────────────────
echo ""
echo "[1/6] Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision --index-url \
    https://download.pytorch.org/whl/cu118 -q
python -c "import torch; print(f'  PyTorch {torch.__version__} ✅')"
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ── 2. ML / Vision libraries ──────────────────────────────────────────
echo ""
echo "[2/6] Installing ML libraries..."
pip install \
    scikit-learn \
    scikit-image \
    timm \
    -q
python -c "import sklearn; print(f'  sklearn {sklearn.__version__} ✅')"

# ── 3. Data / Viz libraries ───────────────────────────────────────────
echo ""
echo "[3/6] Installing data libraries..."
pip install \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    opencv-python-headless \
    Pillow \
    -q
python -c "import pandas; print(f'  pandas {pandas.__version__} ✅')"

# ── 4. Utilities ──────────────────────────────────────────────────────
echo ""
echo "[4/6] Installing utilities..."
pip install \
    nbformat \
    ipykernel \
    tqdm \
    -q

# ── 5. Verify GPU ─────────────────────────────────────────────────────
echo ""
echo "[5/6] Verifying GPU..."
python3 - << 'EOF'
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / 1e9
    print(f"  GPU:  {props.name}")
    print(f"  VRAM: {vram:.1f} GB")
    if vram >= 14:
        print("  ✅ T4 GPU confirmed — batch_size=64 is safe")
    else:
        print(f"  ⚠️  Only {vram:.1f}GB VRAM — reduce batch_size if OOM")
else:
    print("  ❌ No GPU found!")
    print("  Go to Lightning AI Studio settings → Machine → T4 GPU")
EOF

# ── 6. Create output directories ──────────────────────────────────────
echo ""
echo "[6/6] Creating output directories..."
mkdir -p /teamspace/studios/this_studio/outputs/checkpoints
mkdir -p /teamspace/studios/this_studio/dataset/images
echo "  /teamspace/studios/this_studio/outputs/          ✅"
echo "  /teamspace/studios/this_studio/outputs/checkpoints/ ✅"
echo "  /teamspace/studios/this_studio/dataset/images/   ✅"

echo ""
echo "============================================"
echo " Setup complete ✅"
echo "============================================"
echo ""
echo " Next steps:"
echo "   1. Upload dataset (or run Kaggle download):"
echo "      kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection"
echo ""
echo "   2. Upload your project code to /teamspace/studios/this_studio/robust_medical_vision/"
echo ""
echo "   3. Open notebooks in order:"
echo "      phase3_notebooks/01_train_model_b.ipynb"
echo "      phase3_notebooks/02_temperature_and_ood.ipynb"
echo "      phase3_notebooks/03_model_a_gp_baseline.ipynb"
echo "      phase3_notebooks/04_conformal_prediction.ipynb"
echo "      phase3_notebooks/05_ablation_study.ipynb"
echo "      phase3_notebooks/06_architecture_diagram.ipynb"
echo "      phase3_notebooks/07_generate_report_data.ipynb"
echo ""
echo " GPU hours budget:"
echo "   Notebook 1: ~3-4h  (training)"
echo "   Notebook 2: ~0.5h  (calibration)"
echo "   Notebook 3: ~0.5h  (GP baseline)"
echo "   Notebook 4: ~0.2h  (conformal)"
echo "   Notebook 5: ~0.5h  (ablation)"
echo "   Notebook 6: ~0.1h  (diagram)"
echo "   Notebook 7: ~0.3h  (report data)"
echo "   ─────────────────"
echo "   Total:       ~5.5-6h  (well within 22h free tier)"
echo ""
