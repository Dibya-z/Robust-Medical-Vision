"""
=============================================================================
TEMPERATURE SCALING
=============================================================================
WHY THIS EXISTS:
  Phase 2 produced ECE ~0.22 — the model is overconfident.
  When it says "90% melanoma" it is only correct ~68% of the time.

  Temperature Scaling is the simplest, most effective fix.
  It learns ONE parameter T (the temperature) that divides all
  logits before softmax:

      calibrated_probs = softmax(logits / T)

  T > 1 → spreads probabilities → reduces overconfidence
  T < 1 → sharpens probabilities → increases confidence
  T = 1 → no change (identity)

  Critically:
  - Does NOT change model weights
  - Does NOT change predictions (argmax is unchanged)
  - Does NOT change accuracy or F1
  - ONLY changes the probability magnitudes
  - Learned on validation set in ~30 seconds

  This is post-hoc calibration — the most pragmatic fix.
  Expected result: ECE drops from ~0.22 to ~0.05-0.08.

PAPER: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.calibration import calibration_curve


class TemperatureScaling(nn.Module):
    """
    Single-parameter post-hoc calibration.

    Usage:
        1. Train your model fully (all epochs done)
        2. Freeze model weights
        3. Create TemperatureScaling()
        4. Optimize T on validation set
        5. Use calibrated model for all future inference

    The temperature T is the only learnable parameter.
    """

    def __init__(self):
        super().__init__()
        # Initialize T=1.5 (slightly above 1)
        # WHY 1.5 and not 1.0:
        # Neural networks are almost always overconfident (T > 1 needed).
        # Starting at 1.5 puts us in the right ballpark immediately,
        # making optimization faster and more stable.
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        Returns calibrated probabilities.
        """
        return F.softmax(self._scale(logits), dim=1)

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        # Clamp temperature to [0.05, 5.0] for numerical stability
        # WHY: T near 0 causes softmax to explode (division by tiny number)
        # T > 5 makes everything nearly uniform — useless
        T = self.temperature.clamp(min=0.05, max=5.0)
        return logits / T

    @property
    def T(self) -> float:
        """Current temperature value."""
        return self.temperature.item()


def fit_temperature(model: nn.Module,
                    val_loader,
                    device: torch.device,
                    max_iter: int = 50,
                    lr: float = 0.01) -> TemperatureScaling:
    """
    Learn the optimal temperature T on the validation set.

    WHY validation set (not test set):
    Using test set for calibration would be data leakage —
    the calibration would be overfit to test data.
    Validation set is held-out from training and separate from test.

    Args:
        model:       trained RobustMedicalClassifier (weights frozen)
        val_loader:  validation DataLoader
        device:      torch device
        max_iter:    optimization steps
        lr:          learning rate for T

    Returns:
        Fitted TemperatureScaling module
    """
    print("\n" + "=" * 60)
    print("TEMPERATURE SCALING CALIBRATION")
    print("=" * 60)
    print(f"  Goal: minimize NLL on val set by learning T")
    print(f"  Model weights: FROZEN (T is the only parameter)")

    # ── Collect all logits and labels from val set ────────────────────
    # WHY collect first then optimize:
    # We don't need gradients through the model — only through T.
    # Pre-collecting is faster and uses less memory.
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            output = model(images)
            all_logits.append(output['logits'].cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)   # (N, 7)
    all_labels = torch.cat(all_labels, dim=0)   # (N,)

    # ── Compute pre-calibration ECE ───────────────────────────────────
    pre_probs = F.softmax(all_logits, dim=1).numpy()
    pre_preds = pre_probs.argmax(axis=1)
    pre_ece   = compute_ece(pre_probs, all_labels.numpy())
    print(f"\n  Pre-calibration:")
    print(f"    Temperature T = 1.000")
    print(f"    ECE           = {pre_ece:.4f}")

    # ── Optimize T ────────────────────────────────────────────────────
    # WHY NLL (cross-entropy) as the calibration objective:
    # NLL penalizes confident wrong predictions more than uncertain ones.
    # Minimizing NLL on the val set forces T to be the value that makes
    # confidence numbers match accuracy — exactly what calibration means.
    temp_scaler = TemperatureScaling().to(device)
    optimizer   = torch.optim.LBFGS(
        [temp_scaler.temperature],
        lr=lr, max_iter=max_iter
    )
    # WHY LBFGS and not Adam:
    # LBFGS is a quasi-Newton optimizer — uses curvature information.
    # For a 1-parameter optimization problem like this, it converges
    # in very few steps. Adam would work but takes 10× more iterations.

    nll_criterion = nn.CrossEntropyLoss()
    logits_gpu    = all_logits.to(device)
    labels_gpu    = all_labels.to(device)

    def eval_step():
        optimizer.zero_grad()
        scaled_logits = temp_scaler._scale(logits_gpu)
        loss = nll_criterion(scaled_logits, labels_gpu)
        loss.backward()
        return loss

    optimizer.step(eval_step)

    # ── Compute post-calibration ECE ──────────────────────────────────
    with torch.no_grad():
        post_probs = temp_scaler(all_logits.to(device)).cpu().numpy()

    post_ece = compute_ece(post_probs, all_labels.numpy())

    print(f"\n  Post-calibration:")
    print(f"    Temperature T = {temp_scaler.T:.4f}")
    print(f"    ECE           = {post_ece:.4f}")
    print(f"\n  ECE improvement: {pre_ece:.4f} → {post_ece:.4f} "
          f"({((pre_ece - post_ece)/pre_ece)*100:.1f}% reduction) ✅")

    if temp_scaler.T > 1.0:
        print(f"  T > 1: model was overconfident → probabilities spread out")
    else:
        print(f"  T < 1: model was underconfident → probabilities sharpened")

    return temp_scaler


def compute_ece(probs: np.ndarray,
                labels: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Expected Calibration Error — one-vs-rest, averaged across classes.

    ECE = Σ (bin_size/N) × |accuracy_in_bin - confidence_in_bin|

    Lower is better. Target: < 0.10 for clinical use.
    """
    num_classes = probs.shape[1]
    ece_per_class = []

    for cls in range(num_classes):
        binary_labels = (labels == cls).astype(int)
        cls_probs     = probs[:, cls]

        # Skip classes with no positive examples
        if binary_labels.sum() == 0:
            continue

        try:
            fraction_pos, mean_conf = calibration_curve(
                binary_labels, cls_probs,
                n_bins=n_bins, strategy='uniform'
            )
            # Weighted absolute calibration error for this class
            counts = np.histogram(cls_probs, bins=n_bins, range=(0,1))[0]
            weights = counts / counts.sum()
            # Only weight bins that actually have data
            n_bins_actual = len(fraction_pos)
            cls_ece = np.sum(
                np.abs(fraction_pos - mean_conf) *
                weights[:n_bins_actual]
            )
            ece_per_class.append(cls_ece)
        except Exception:
            pass

    return float(np.mean(ece_per_class)) if ece_per_class else 0.0
