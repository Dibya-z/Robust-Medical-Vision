"""
=============================================================================
MAHALANOBIS DISTANCE OOD DETECTION
=============================================================================
WHY THIS EXISTS:
  MC Dropout detects OOD through variance — it needs 30 forward passes.
  The Evidential Head detects OOD through low total evidence S.
  Both are good but neither is specifically designed for OOD detection.

  Mahalanobis Distance is a dedicated OOD detector that operates on
  the deep feature space (the 256-dim vector before the output heads).

  CORE IDEA:
  During training, for each class, compute the mean feature vector
  and the covariance matrix of all training samples of that class.
  This defines a multivariate Gaussian per class in feature space.

  At inference time, compute how far the new sample's feature vector
  is from the nearest class Gaussian. Far = OOD. Close = in-distribution.

  WHY Mahalanobis over Euclidean distance:
  Euclidean distance treats all feature dimensions equally.
  Mahalanobis accounts for the covariance structure — if features
  are correlated or have different scales, Mahalanobis handles it.

  EMPIRICALLY:
  Lee et al. (2018) "A Simple Unified Framework for Detecting
  Out-of-Distribution Samples" — Mahalanobis outperforms MC Dropout,
  ODIN, and softmax confidence on most OOD benchmarks.
  This makes it the right tool for clinical OOD detection.

  HOW IT FITS IN THE PIPELINE:
  1. After training: fit Gaussians on training features (once)
  2. At inference: compute Mahalanobis score in <1ms (single pass)
  3. If score > threshold: flag as OOD before classification
  4. If score <= threshold: classify normally + report uncertainty
=============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from sklearn.covariance import EmpiricalCovariance


class MahalanobisOODDetector:
    """
    Fits a class-conditional Gaussian on training features.
    At inference, computes Mahalanobis distance to nearest class.

    HIGH distance = OOD (never seen anything like this)
    LOW  distance = In-distribution (familiar input)
    """

    def __init__(self, num_classes: int = 7):
        self.num_classes   = num_classes
        self.class_means   = {}    # {class_idx: mean_vector (D,)}
        self.precision_matrix = None  # shared precision (inv covariance)
        self.threshold     = None  # set during calibration
        self.feature_dim   = None
        self._is_fitted    = False

    def fit(self,
            model: nn.Module,
            train_loader,
            device: torch.device):
        """
        Compute per-class means and shared covariance from training data.

        WHY shared covariance (tied) and not per-class covariance:
        Per-class covariance needs enough samples per class to be
        well-estimated. Dermatofibroma has only 80-90 training images.
        A 256×256 covariance matrix needs >256 samples to be full rank.
        Tied (shared) covariance pools all classes — much more stable.
        This is also what the original paper recommends.

        Args:
            model:        trained RobustMedicalClassifier
            train_loader: training DataLoader
            device:       torch device
        """
        print("\n" + "=" * 60)
        print("MAHALANOBIS OOD DETECTOR — FITTING")
        print("=" * 60)
        print("  Step 1: Extracting features from training set...")

        model.eval()
        # Collect features per class
        class_features: Dict[int, list] = {i: [] for i in range(self.num_classes)}

        with torch.no_grad():
            for images, labels, _ in train_loader:
                images = images.to(device)
                output = model(images)
                # features: (batch, 256) — the 256-dim head output
                feats  = output['features'].cpu().numpy()
                lbls   = labels.numpy()

                for feat, lbl in zip(feats, lbls):
                    class_features[lbl].append(feat)

        self.feature_dim = class_features[0][0].shape[0]

        # ── Per-class means ───────────────────────────────────────────
        print("  Step 2: Computing per-class means...")
        for cls_idx in range(self.num_classes):
            feats_cls = np.array(class_features[cls_idx])
            self.class_means[cls_idx] = feats_cls.mean(axis=0)
            print(f"    Class {cls_idx}: {len(feats_cls)} samples, "
                  f"mean norm = {np.linalg.norm(self.class_means[cls_idx]):.3f}")

        # ── Tied covariance (pooled across all classes) ───────────────
        # WHY pooled: each class has insufficient samples for a
        # reliable full-rank 256×256 covariance estimate.
        print("  Step 3: Computing tied (pooled) precision matrix...")
        all_centered = []
        for cls_idx in range(self.num_classes):
            feats_cls = np.array(class_features[cls_idx])
            centered  = feats_cls - self.class_means[cls_idx]
            all_centered.append(centered)

        all_centered = np.vstack(all_centered)   # (N_total, 256)

        # EmpiricalCovariance with shrinkage for numerical stability
        # WHY shrinkage (regularization):
        # Raw sample covariance can be singular if features are
        # correlated or if N < D. Shrinkage adds a small identity
        # component to guarantee invertibility.
        cov_estimator = EmpiricalCovariance(assume_centered=True)
        cov_estimator.fit(all_centered)

        # Precision = inverse covariance
        self.precision_matrix = cov_estimator.precision_

        print(f"  Precision matrix shape: {self.precision_matrix.shape}")
        print(f"  Condition number: "
              f"{np.linalg.cond(self.precision_matrix):.2e}")
        self._is_fitted = True
        print("  Fitting complete ✅")

    def mahalanobis_score(self,
                          features: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance to nearest class Gaussian.

        M(x) = min_c sqrt((x - μ_c)^T Σ^{-1} (x - μ_c))

        Args:
            features: (N, D) feature vectors
        Returns:
            scores: (N,) — lower = more in-distribution
        """
        assert self._is_fitted, "Call fit() before score()"

        scores = []
        for feat in features:
            # Distance to each class mean
            class_dists = []
            for cls_idx in range(self.num_classes):
                diff = feat - self.class_means[cls_idx]   # (D,)
                # Mahalanobis: diff^T @ precision @ diff
                dist = np.dot(diff, np.dot(self.precision_matrix, diff))
                class_dists.append(dist)
            # Take MINIMUM distance (closest class)
            # WHY minimum: we want to know how close this sample is
            # to ANY known class. Far from all = OOD.
            scores.append(min(class_dists))

        return np.array(scores)

    def calibrate_threshold(self,
                            model: nn.Module,
                            val_loader,
                            device: torch.device,
                            fpr_target: float = 0.05):
        """
        Set OOD threshold so that fpr_target fraction of in-distribution
        val images are falsely flagged as OOD.

        WHY calibrate on val set:
        We want the threshold to reflect real in-distribution spread,
        not the training set (which the model has memorized).

        fpr_target=0.05 means: at most 5% of normal images get flagged.
        This is the clinical operating point — low false alarm rate.
        """
        print("\n  Calibrating OOD threshold on validation set...")
        model.eval()
        val_scores = []

        with torch.no_grad():
            for images, _, _ in val_loader:
                images = images.to(device)
                output = model(images)
                feats  = output['features'].cpu().numpy()
                scores = self.mahalanobis_score(feats)
                val_scores.extend(scores.tolist())

        val_scores = np.array(val_scores)
        # Threshold at (1 - fpr_target) percentile of in-dist scores
        # WHY: 95th percentile means 95% of in-dist samples score below
        # this — so only 5% are falsely flagged
        self.threshold = np.percentile(val_scores, (1 - fpr_target) * 100)

        print(f"  Threshold set at {(1-fpr_target)*100:.0f}th percentile: "
              f"{self.threshold:.4f}")
        print(f"  False alarm rate on val: {fpr_target*100:.0f}% "
              f"(clinically acceptable)")

    def predict(self,
                features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict OOD status for a batch of feature vectors.

        Returns:
            scores:   (N,) Mahalanobis distances
            is_ood:   (N,) boolean — True = OOD
        """
        assert self.threshold is not None, \
            "Call calibrate_threshold() before predict()"
        scores = self.mahalanobis_score(features)
        is_ood = scores > self.threshold
        return scores, is_ood

    def save(self, path: str):
        """Save detector state."""
        import pickle
        state = {
            'class_means':      self.class_means,
            'precision_matrix': self.precision_matrix,
            'threshold':        self.threshold,
            'feature_dim':      self.feature_dim,
            'num_classes':      self.num_classes,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"  OOD detector saved: {path}")

    def load(self, path: str):
        """Load detector state."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.class_means      = state['class_means']
        self.precision_matrix = state['precision_matrix']
        self.threshold        = state['threshold']
        self.feature_dim      = state['feature_dim']
        self.num_classes      = state['num_classes']
        self._is_fitted       = True
        print(f"  OOD detector loaded: {path}")
