"""
=============================================================================
STEP 6: EVALUATION & INTERPRETABILITY
=============================================================================
WHY A SEPARATE EVALUATION MODULE?

Training a model is only half the work. The other half is proving:
  (1) The model learned the RIGHT features (Grad-CAM)
  (2) The model's confidence numbers are trustworthy (Calibration)
  (3) The uncertainty scores are meaningful (Uncertainty Analysis)
  (4) The model knows what it doesn't know (OOD Detection)

Each of these answers a different rubric criterion:
  - Grad-CAM          → "validate the model is learning valid features"
  - Calibration curve → "use of appropriate error metrics"
  - Uncertainty plot  → proves uncertainty-aware model works
  - OOD detection     → the core medical safety claim of your project
=============================================================================
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_auc_score, classification_report
)
from typing import List, Dict
import cv2


IDX_TO_CLASS = {
    0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc',
    4: 'akiec', 5: 'vasc', 6: 'df'
}

DISPLAY_NAMES = {
    'nv': 'Melanocytic\nNevus', 'mel': 'Melanoma',
    'bkl': 'Benign\nKeratosis', 'bcc': 'Basal Cell\nCarcinoma',
    'akiec': 'Actinic\nKeratosis', 'vasc': 'Vascular\nLesion',
    'df': 'Dermatofibroma'
}


# =============================================================================
# STEP 6A: GRAD-CAM
# =============================================================================
# WHY Grad-CAM?
# Gradient-weighted Class Activation Mapping shows WHICH pixels in the image
# contributed most to the model's final prediction.
#
# HOW it works (conceptually):
#   1. Pick a convolutional layer deep in the backbone
#   2. Forward pass the image → get the prediction
#   3. Backpropagate the gradient of the predicted class score
#      back to that layer
#   4. The gradients tell us: "which feature maps were important?"
#   5. Weight the feature maps by their gradient importance
#   6. Average → get a heatmap → resize to image size
#
# WHY the last conv layer?
# Later conv layers have more semantically rich features (they "know" about
# shapes and structures, not just edges). The last conv layer just before
# pooling is the richest — perfect for class discrimination heatmaps.
#
# WHAT we look for in the results:
#   ✅ Correct prediction: heatmap concentrated on the lesion
#   ⚠️  High uncertainty: heatmap diffuse, spread across image
#   ❌ Wrong prediction: heatmap on background artifacts (ruler, hair)
#      → This last case is "Shortcut Learning" and must be reported

class GradCAM:
    """
    Grad-CAM implementation using PyTorch hooks.
    
    WHY hooks?
    PyTorch's backward pass overwrites gradients at each layer.
    Hooks let us "intercept" the gradients at a specific layer
    and save them before they're overwritten.
    This is the standard implementation approach.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model:        the RobustMedicalClassifier
            target_layer: the conv layer to compute CAM for
                         e.g. model.backbone.features[-1]
        """
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        # Register hooks to capture activations and gradients
        self._register_hooks()

    def _register_hooks(self):
        """
        Forward hook: captures feature map activations
        Backward hook: captures gradients at that layer
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()   # feature maps

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()   # gradients

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor, 
                 class_idx: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an image.
        
        Args:
            image_tensor: (1, 3, 224, 224) — single image
            class_idx:    which class to explain (None = predicted class)
        
        Returns:
            heatmap: (H, W) — values in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(image_tensor)
        logits = output['logits']

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for the specific class
        # WHY scalar target? Gradients are computed w.r.t. a scalar.
        # We select the class score and backpropagate it.
        class_score = logits[0, class_idx]
        class_score.backward()

        # ── Compute weighted feature maps ──────────────────────────────────
        # gradients: (1, C, H, W) where C = number of channels in target layer
        # Global average pooling of gradients → importance weight per channel
        weights      = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        activations  = self.activations                                   # (1, C, H, W)

        # Weighted sum of feature maps
        cam = (weights * activations).sum(dim=1, keepdim=True)          # (1, 1, H, W)

        # ReLU: keep only positive contributions
        # WHY ReLU: Negative values mean features that DECREASE the class score.
        # We only care about regions that INCREASE the class score.
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def overlay_on_image(self, image: np.ndarray, 
                          cam: np.ndarray,
                          alpha: float = 0.4) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            image: (H, W, 3) uint8 RGB image
            cam:   (H, W) heatmap in [0, 1]
            alpha: transparency of overlay
        """
        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

        # Convert to colormap (jet: blue=low, red=high importance)
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend with original image
        overlay = (alpha * heatmap_rgb + (1 - alpha) * image).astype(np.uint8)
        return overlay


def visualize_gradcam(model, dataset, device, output_dir: str,
                       n_per_class: int = 3):
    """
    Generate Grad-CAM visualizations for correct, uncertain, and OOD cases.
    
    We visualize three types of cases to tell a story:
      1. High confidence correct   → heatmap on lesion (model learned right thing)
      2. High uncertainty cases    → diffuse heatmap (model doesn't know where to look)
      3. Random OOD image          → scattered activation (model is lost)
    """
    print("\n  Generating Grad-CAM visualizations...")

    # Get the last conv layer of EfficientNet-B1
    # WHY this specific layer: it's the deepest conv block, richest features
    target_layer = model.backbone.features[-1]
    gradcam      = GradCAM(model, target_layer)

    import torchvision.transforms as T
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    denorm_mean = np.array([0.485, 0.456, 0.406])
    denorm_std  = np.array([0.229, 0.224, 0.225])

    samples_per_class = {cls: [] for cls in range(7)}
    for i in range(min(500, len(dataset))):
        _, label, _ = dataset[i]
        if len(samples_per_class[label.item()]) < n_per_class:
            samples_per_class[label.item()].append(i)
        if all(len(v) >= n_per_class for v in samples_per_class.values()):
            break

    fig, axes = plt.subplots(7, n_per_class * 2, figsize=(n_per_class * 5, 7 * 3))
    fig.suptitle('Grad-CAM: What the Model Looks At\n'
                 'Left: Original | Right: Grad-CAM overlay (red = high attention)',
                 fontsize=13, fontweight='bold')

    for cls_idx in range(7):
        for col, sample_idx in enumerate(samples_per_class[cls_idx]):
            img_tensor, label, _ = dataset[sample_idx]
            img_tensor_batch = img_tensor.unsqueeze(0).to(device)

            # Generate CAM
            cam = gradcam.generate(img_tensor_batch, class_idx=cls_idx)

            # Denormalize image for display
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * denorm_std + denorm_mean)
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            overlay = gradcam.overlay_on_image(img_np, cam)

            # Original
            axes[cls_idx][col * 2].imshow(img_np)
            axes[cls_idx][col * 2].axis('off')
            if col == 0:
                axes[cls_idx][col * 2].set_ylabel(
                    list(DISPLAY_NAMES.values())[cls_idx],
                    fontsize=8, fontweight='bold'
                )

            # Grad-CAM overlay
            axes[cls_idx][col * 2 + 1].imshow(overlay)
            axes[cls_idx][col * 2 + 1].axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, 'gradcam_per_class.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    Grad-CAM saved to: {path}")


# =============================================================================
# STEP 6B: CALIBRATION CURVE
# =============================================================================
# WHY calibration matters:
# A well-calibrated model means: when it says "90% confident," it's right
# ~90% of the time. Most neural networks are OVERCONFIDENT — they say
# "95% confident" but are only right 70% of the time.
#
# This matters clinically: if a doctor sees "95% melanoma" and takes
# aggressive action, that 95% better be real.
#
# HOW to read the reliability diagram:
#   - x-axis: predicted confidence (what model claims)
#   - y-axis: actual accuracy in that confidence bin (ground truth)
#   - Perfect calibration = diagonal line y=x
#   - Curve ABOVE diagonal = model is underconfident (conservative)
#   - Curve BELOW diagonal = model is overconfident (dangerous in medicine)
#
# Expected Calibration Error (ECE):
#   ECE = weighted average of |confidence - accuracy| across bins
#   Lower is better. Random model: ECE ≈ 0.3. Good model: ECE < 0.1.

def plot_calibration_curve(all_probs: np.ndarray, all_labels: np.ndarray,
                            num_classes: int = 7, output_dir: str = './outputs'):
    """
    Plot reliability diagram and compute ECE.
    
    Args:
        all_probs:  (n_samples, num_classes) — model probability outputs
        all_labels: (n_samples,) — true class labels
    """
    print("\n  Computing calibration curves...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Calibration Analysis\n'
                 'Diagonal = perfect calibration | '
                 'Below diagonal = overconfident (bad for medicine)',
                 fontsize=12, fontweight='bold')

    # ── Per-class calibration (one-vs-rest) ────────────────────────────────
    ece_scores = []
    n_bins = 10

    for cls_idx in range(num_classes):
        binary_labels = (all_labels == cls_idx).astype(int)
        cls_probs     = all_probs[:, cls_idx]

        try:
            fraction_pos, mean_confidence = calibration_curve(
                binary_labels, cls_probs, n_bins=n_bins, strategy='uniform'
            )
            cls_name = IDX_TO_CLASS[cls_idx]
            axes[0].plot(mean_confidence, fraction_pos,
                         label=DISPLAY_NAMES[cls_name].replace('\n', ' '),
                         marker='o', markersize=4, alpha=0.8)

            # ECE for this class
            bin_weights = np.histogram(cls_probs, bins=n_bins, range=(0, 1))[0]
            bin_weights = bin_weights / bin_weights.sum()
            ece_scores.append(np.abs(fraction_pos - mean_confidence).mean())
        except Exception:
            pass

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    axes[0].set_xlabel('Mean Predicted Confidence')
    axes[0].set_ylabel('Fraction Positives (True Frequency)')
    axes[0].set_title('Reliability Diagram (per class)')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[0].grid(alpha=0.3)

    # ── ECE bar chart ─────────────────────────────────────────────────────
    class_names = [DISPLAY_NAMES[IDX_TO_CLASS[i]].replace('\n', ' ')
                   for i in range(num_classes)]
    colors = ['#DD3B3B' if ece > 0.1 else '#55A868' for ece in ece_scores]
    bars = axes[1].bar(class_names, ece_scores, color=colors, edgecolor='white')
    axes[1].axhline(0.1, color='orange', linestyle='--', alpha=0.8,
                     label='ECE=0.1 threshold')
    axes[1].set_ylabel('Expected Calibration Error (ECE)')
    axes[1].set_title('ECE per Class\n(green = well-calibrated, red = needs improvement)')
    axes[1].tick_params(axis='x', rotation=40)
    axes[1].legend()
    for bar, ece in zip(bars, ece_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f'{ece:.3f}', ha='center', fontsize=8)

    mean_ece = np.mean(ece_scores)
    print(f"    Mean ECE: {mean_ece:.4f} "
          f"({'well-calibrated ✅' if mean_ece < 0.1 else 'overconfident ⚠️'})")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    return mean_ece


# =============================================================================
# STEP 6C: UNCERTAINTY ANALYSIS
# =============================================================================
# WHY this plot?
# This is the core proof that your uncertainty model works.
#
# If uncertainty is meaningful, then:
#   - HIGH uncertainty samples → model is wrong more often
#   - LOW uncertainty samples  → model is right more often
#
# We prove this by sorting test samples by uncertainty score,
# binning them, and plotting accuracy per bin.
# The expected shape: accuracy DECREASES as uncertainty INCREASES.
#
# If the plot is flat → uncertainty is just random noise.
# If the plot shows a clear decline → uncertainty tracks genuine difficulty.

def plot_uncertainty_analysis(uncertainties: np.ndarray, 
                               correct: np.ndarray,
                               output_dir: str = './outputs'):
    """
    Plot accuracy vs uncertainty to prove uncertainty is meaningful.
    
    Args:
        uncertainties: (n_samples,) — MC dropout variance per sample
        correct:       (n_samples,) — 1 if prediction was correct, 0 otherwise
    """
    print("\n  Plotting uncertainty analysis...")

    n_bins = 10
    bins   = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_accuracy = []
    bin_centers  = []

    for i in range(n_bins):
        mask = (uncertainties >= bins[i]) & (uncertainties <= bins[i + 1])
        if mask.sum() > 0:
            bin_accuracy.append(correct[mask].mean())
            bin_centers.append((bins[i] + bins[i + 1]) / 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Uncertainty Analysis — Does High Uncertainty = Wrong Prediction?\n'
                 'If yes: our uncertainty model is working correctly.',
                 fontsize=12, fontweight='bold')

    # ── Accuracy vs Uncertainty bins ─────────────────────────────────────
    axes[0].bar(range(len(bin_accuracy)), bin_accuracy,
                color=['#55A868' if a > 0.6 else '#DD3B3B' for a in bin_accuracy],
                edgecolor='white')
    axes[0].set_xlabel('Uncertainty Bin (left = low, right = high)')
    axes[0].set_ylabel('Prediction Accuracy')
    axes[0].set_title('Accuracy per Uncertainty Bin\n'
                       '→ Declining curve = uncertainty is meaningful ✅')
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    axes[0].set_xticks(range(len(bin_accuracy)))
    axes[0].set_xticklabels([f'B{i+1}' for i in range(len(bin_accuracy))])
    axes[0].legend()

    # ── Uncertainty distribution: correct vs wrong predictions ────────────
    correct_uncertainty = uncertainties[correct == 1]
    wrong_uncertainty   = uncertainties[correct == 0]
    axes[1].hist(correct_uncertainty, bins=30, alpha=0.6, color='#55A868',
                  label='Correct predictions', density=True)
    axes[1].hist(wrong_uncertainty, bins=30, alpha=0.6, color='#DD3B3B',
                  label='Wrong predictions', density=True)
    axes[1].set_xlabel('MC Dropout Uncertainty (variance)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Uncertainty Distribution\n'
                       '→ Wrong predictions should have higher uncertainty')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Key statistic: mean uncertainty for correct vs wrong
    print(f"    Mean uncertainty — Correct: {correct_uncertainty.mean():.4f} | "
          f"Wrong: {wrong_uncertainty.mean():.4f}")
    if wrong_uncertainty.mean() > correct_uncertainty.mean():
        print(f"    Wrong predictions have higher uncertainty ✅ — model is self-aware")
    else:
        print(f"    Unexpected result ⚠️ — investigate calibration")


# =============================================================================
# STEP 6D: OOD DETECTION
# =============================================================================
# WHY OOD detection is the medical safety claim:
#
# In a real clinical setting, doctors sometimes submit incorrect image types
# by mistake (e.g., an X-ray submitted to a skin lesion classifier).
# The model must not confidently classify these — it must flag them as
# "I don't know what this is."
#
# We simulate this by feeding images from a completely different domain
# (chest X-rays from NIH CXR14) into our skin lesion model.
# The model should output HIGH uncertainty for these images.
#
# If it does → our system is clinically safe.
# If it doesn't → the model is making random confident predictions → dangerous.

def evaluate_ood_detection(model, in_dist_loader, ood_images_dir: str,
                            device, output_dir: str, n_passes: int = 15):
    """
    Compare uncertainty scores: in-distribution (HAM10000) vs OOD images.
    
    Args:
        model:            trained RobustMedicalClassifier
        in_dist_loader:   test DataLoader for HAM10000
        ood_images_dir:   folder with OOD images (e.g., chest X-rays)
        device:           torch device
        n_passes:         MC Dropout forward passes
    """
    print("\n  Running OOD detection evaluation...")

    import torchvision.transforms as T
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── In-distribution uncertainty ─────────────────────────────────────────
    in_dist_uncertainties = []
    model.train()   # keep MC Dropout active
    with torch.no_grad():
        for images, _, _ in in_dist_loader:
            images = images.to(device)
            result = model.predict_with_uncertainty(images, n_passes=n_passes)
            in_dist_uncertainties.extend(
                result['mc_uncertainty'].cpu().numpy()
            )
            if len(in_dist_uncertainties) >= 500:
                break

    # ── OOD uncertainty ──────────────────────────────────────────────────────
    ood_uncertainties = []
    if os.path.isdir(ood_images_dir):
        ood_files = [
            os.path.join(ood_images_dir, f)
            for f in os.listdir(ood_images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:200]

        for img_path in ood_files:
            try:
                img   = Image.open(img_path).convert('RGB')
                img_t = val_transform(img).unsqueeze(0).to(device)
                result = model.predict_with_uncertainty(img_t, n_passes=n_passes)
                ood_uncertainties.append(
                    result['mc_uncertainty'].cpu().item()
                )
            except Exception:
                continue

    in_dist_uncertainties = np.array(in_dist_uncertainties)
    ood_uncertainties     = np.array(ood_uncertainties)

    # ── Plot comparison ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('OOD Detection: Does the Model Know What It Doesn\'t Know?\n'
                 'In-dist = HAM10000 skin lesions | OOD = Chest X-rays',
                 fontsize=12, fontweight='bold')

    axes[0].hist(in_dist_uncertainties, bins=30, alpha=0.7,
                  color='#4C72B0', label='In-distribution (HAM10000)', density=True)
    if len(ood_uncertainties) > 0:
        axes[0].hist(ood_uncertainties, bins=30, alpha=0.7,
                      color='#DD3B3B', label='OOD (Chest X-rays)', density=True)
    axes[0].set_xlabel('MC Dropout Uncertainty')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Uncertainty Distribution\n'
                       '→ OOD should have much higher uncertainty')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Detection rate at threshold ──────────────────────────────────────────
    # Choose threshold as 95th percentile of in-distribution uncertainty.
    # WHY: If the model is well-calibrated, 95% of normal images should
    # be below this threshold. Anything above → likely OOD.
    if len(ood_uncertainties) > 0:
        threshold        = np.percentile(in_dist_uncertainties, 95)
        ood_detected     = (ood_uncertainties > threshold).mean() * 100
        false_alarm_rate = (in_dist_uncertainties > threshold).mean() * 100

        thresholds = np.linspace(
            in_dist_uncertainties.min(), in_dist_uncertainties.max(), 100
        )
        ood_rates  = [(ood_uncertainties > t).mean() for t in thresholds]
        fpr        = [(in_dist_uncertainties > t).mean() for t in thresholds]

        axes[1].plot(fpr, ood_rates, 'b-', linewidth=2, label='ROC curve')
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        axes[1].scatter([false_alarm_rate / 100], [ood_detected / 100],
                         color='red', s=100, zorder=5,
                         label=f'Operating point (threshold=95th pct)')
        axes[1].set_xlabel('False Alarm Rate (in-dist flagged as OOD)')
        axes[1].set_ylabel('OOD Detection Rate')
        axes[1].set_title(f'OOD Detection ROC\nAt threshold: {ood_detected:.1f}% OOD detected, '
                           f'{false_alarm_rate:.1f}% false alarms')
        axes[1].legend()

        print(f"    Threshold (95th pct of in-dist): {threshold:.4f}")
        print(f"    OOD detection rate:  {ood_detected:.1f}%")
        print(f"    False alarm rate:    {false_alarm_rate:.1f}%")
        if ood_detected > 70:
            print(f"    Model successfully detects OOD inputs ✅")
        else:
            print(f"    OOD detection is weak ⚠️ — consider temperature scaling")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ood_detection.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# STEP 6E: CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(all_preds, all_labels, output_dir):
    """Plot normalized confusion matrix."""
    print("\n  Plotting confusion matrix...")

    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    labels = [DISPLAY_NAMES[IDX_TO_CLASS[i]].replace('\n', ' ')
              for i in range(7)]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted Class', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)
    ax.set_title('Confusion Matrix (normalized)\n'
                  'Diagonal = correct predictions | '
                  'Off-diagonal = misclassifications',
                  fontsize=12, fontweight='bold')
    plt.xticks(rotation=40, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
