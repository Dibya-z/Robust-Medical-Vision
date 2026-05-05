"""
=============================================================================
CONFORMAL PREDICTION
=============================================================================
WHY THIS EXISTS:
  Both MC Dropout and the Evidential Head produce uncertainty scores,
  but these scores have no statistical guarantee. When the model says
  "uncertainty = 0.003", that number has no formal interpretation.

  Conformal Prediction (CP) wraps the final model and produces
  PREDICTION SETS with a GUARANTEED coverage probability.

  Example output with CP at 95% coverage:
    Standard model: "Melanoma: 82%"
    With CP:        "{Melanoma, Nevus} — true class is in this set
                     with 95% guaranteed probability"

  The guarantee is distribution-free — it holds regardless of the
  true data distribution, without any assumptions about the model.

  WHY THIS MATTERS CLINICALLY:
  A doctor can act on a set prediction: "the answer is either melanoma
  or nevus — biopsy to distinguish." This is legally defensible and
  clinically actionable in a way that "82% melanoma" is not.

  HOW IT WORKS (RAPS variant — Regularized Adaptive Prediction Sets):
  1. On calibration set (part of val set): compute nonconformity scores
     s_i = nonconformity(model_output_i, true_label_i)
  2. Find the (1-α) quantile of these scores: q_hat
  3. At inference: include class k in prediction set if
     nonconformity(output, k) <= q_hat
  4. The resulting set contains the true class with probability >= 1-α

  We use RAPS (Angelopoulos et al. 2020) which produces smaller,
  more informative sets than naive CP while keeping the coverage guarantee.

PAPER: Angelopoulos et al. "Uncertainty Sets for Image Classifiers
       using Conformal Prediction" (ICLR 2021)
=============================================================================
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple


class ConformalPredictor:
    """
    RAPS Conformal Predictor.

    Produces adaptive prediction sets with guaranteed coverage.
    Coverage guarantee: P(true_label ∈ prediction_set) >= 1 - alpha.

    Args:
        alpha:     miscoverage rate (default 0.05 = 95% coverage)
        lambda_:   RAPS regularization (penalizes large sets)
        k_reg:     RAPS — start penalizing after k_reg classes in set
    """

    def __init__(self,
                 alpha:    float = 0.05,
                 lambda_:  float = 0.01,
                 k_reg:    int   = 5):
        self.alpha    = alpha
        self.lambda_  = lambda_
        self.k_reg    = k_reg
        self.q_hat    = None   # the calibration quantile
        self._is_fitted = False

    def _nonconformity_score(self,
                              softmax_probs: np.ndarray,
                              labels: np.ndarray = None,
                              return_all: bool = False) -> np.ndarray:
        """
        RAPS nonconformity score.

        For each sample, sort classes by probability (descending).
        Accumulate probabilities until we reach the true class.
        The score = cumulative probability up to and including true class
                  + regularization penalty for large sets.

        Higher score = model was more uncertain about the true class.
        Lower score  = model correctly assigned high probability to true class.

        Args:
            softmax_probs: (N, K) probabilities
            labels:        (N,) true labels (None for inference)
            return_all:    if True, return scores for ALL classes (inference)

        Returns:
            scores: (N,) calibration scores or (N, K) inference scores
        """
        N, K = softmax_probs.shape

        # Sort classes by probability descending for each sample
        # WHY sort: we want to know "how many classes does the model
        # need to consider before reaching the true class?"
        sorted_idx  = np.argsort(-softmax_probs, axis=1)   # (N, K)
        sorted_prob = np.take_along_axis(softmax_probs, sorted_idx, axis=1)

        # Cumulative sum of sorted probabilities
        cum_probs = np.cumsum(sorted_prob, axis=1)          # (N, K)

        # RAPS regularization: penalize sets larger than k_reg
        # L[i,j] = max(0, j+1 - k_reg) * lambda_
        # j+1 = number of classes included so far
        reg = np.maximum(
            0,
            np.arange(1, K + 1)[None, :] - self.k_reg
        ) * self.lambda_   # (1, K) broadcast to (N, K)

        scores_sorted = cum_probs + reg   # (N, K)

        if return_all:
            # For inference: return score for each class in original order
            # Unsort back to original class ordering
            scores_original = np.zeros_like(scores_sorted)
            np.put_along_axis(scores_original, sorted_idx, scores_sorted, axis=1)
            return scores_original   # (N, K)

        else:
            # For calibration: return score at the TRUE class position
            assert labels is not None
            scores = np.zeros(N)
            for i in range(N):
                # Find where the true label lands in the sorted order
                true_rank = np.where(sorted_idx[i] == labels[i])[0][0]
                scores[i] = scores_sorted[i, true_rank]
            return scores   # (N,)

    def calibrate(self,
                  model:      torch.nn.Module,
                  cal_loader,
                  device:     torch.device,
                  temp_scaler=None):
        """
        Fit the conformal quantile q_hat on calibration data.

        WHY separate calibration set:
        We need data the model hasn't trained on AND that is separate
        from the test set. We use a portion of the val set for this.

        Args:
            model:       trained model (weights frozen)
            cal_loader:  calibration DataLoader (subset of val)
            device:      torch device
            temp_scaler: fitted TemperatureScaling (optional but recommended)
        """
        print("\n" + "=" * 60)
        print("CONFORMAL PREDICTION — CALIBRATION")
        print(f"  Coverage target: {(1-self.alpha)*100:.0f}%")
        print("=" * 60)

        model.eval()
        all_probs  = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in cal_loader:
                images = images.to(device)
                output = model(images)
                logits = output['logits']

                # Apply temperature scaling if provided
                if temp_scaler is not None:
                    probs = temp_scaler(logits)
                else:
                    probs = F.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

        all_probs  = np.vstack(all_probs)     # (N, 7)
        all_labels = np.concatenate(all_labels)  # (N,)

        # Compute nonconformity scores on calibration set
        cal_scores = self._nonconformity_score(all_probs, all_labels)

        # The (1-alpha) quantile — adjusted for finite sample
        # WHY n+1 correction: this finite-sample correction guarantees
        # exactly (1-alpha) coverage (not just approximately)
        n     = len(cal_scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_hat      = np.quantile(cal_scores, level)
        self._is_fitted = True

        print(f"  Calibration samples: {n}")
        print(f"  q_hat (threshold):   {self.q_hat:.4f}")
        print(f"  ✅ Conformal predictor ready")

        # Sanity check: coverage on cal set should be ~= (1-alpha)
        covered = (cal_scores <= self.q_hat).mean()
        print(f"  Empirical coverage on cal set: {covered:.3f} "
              f"(target: {1-self.alpha:.3f})")

    def predict(self,
                probs: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
        """
        Generate prediction sets for a batch of probability vectors.

        Args:
            probs: (N, K) softmax probabilities (after temperature scaling)

        Returns:
            prediction_sets: list of N lists, each containing class indices
            set_sizes:       (N,) number of classes in each set
        """
        assert self._is_fitted, "Call calibrate() first"

        # Nonconformity score for every class for every sample
        all_scores = self._nonconformity_score(probs, return_all=True)

        # Include class k if its nonconformity score <= q_hat
        prediction_sets = []
        for i in range(len(probs)):
            pred_set = [k for k in range(probs.shape[1])
                        if all_scores[i, k] <= self.q_hat]
            # Always include at least the top-1 prediction
            if len(pred_set) == 0:
                pred_set = [probs[i].argmax()]
            prediction_sets.append(pred_set)

        set_sizes = np.array([len(s) for s in prediction_sets])
        return prediction_sets, set_sizes

    def evaluate(self,
                 model:      torch.nn.Module,
                 test_loader,
                 device:     torch.device,
                 temp_scaler=None,
                 class_names: list = None) -> dict:
        """
        Evaluate conformal predictor on test set.

        Metrics:
          - Empirical coverage: fraction where true label ∈ prediction set
          - Average set size:   smaller = more informative
          - Coverage per class: should all be >= (1-alpha)

        Args:
            model:       trained model
            test_loader: test DataLoader
            device:      torch device
            temp_scaler: fitted TemperatureScaling (recommended)
            class_names: list of class name strings for reporting

        Returns:
            dict of evaluation metrics
        """
        assert self._is_fitted, "Call calibrate() first"
        print("\n" + "=" * 60)
        print("CONFORMAL PREDICTION — TEST EVALUATION")
        print(f"  Coverage guarantee: >= {(1-self.alpha)*100:.0f}%")
        print("=" * 60)

        model.eval()
        all_probs  = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device)
                output = model(images)
                logits = output['logits']

                if temp_scaler is not None:
                    probs = temp_scaler(logits)
                else:
                    probs = F.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

        all_probs  = np.vstack(all_probs)
        all_labels = np.concatenate(all_labels)

        # Generate prediction sets
        pred_sets, set_sizes = self.predict(all_probs)

        # ── Overall coverage ──────────────────────────────────────────
        covered  = np.array([
            all_labels[i] in pred_sets[i]
            for i in range(len(all_labels))
        ])
        coverage = covered.mean()

        # ── Per-class coverage ────────────────────────────────────────
        num_classes = all_probs.shape[1]
        class_coverage = {}
        for cls in range(num_classes):
            mask = all_labels == cls
            if mask.sum() > 0:
                class_coverage[cls] = covered[mask].mean()

        # ── Singleton rate ────────────────────────────────────────────
        # Sets of size 1 = model is very confident
        singleton_rate = (set_sizes == 1).mean()

        # ── Print results ─────────────────────────────────────────────
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]

        print(f"\n  Overall Coverage:   {coverage:.4f} "
              f"(guarantee: >= {1-self.alpha:.2f}) "
              f"{'✅' if coverage >= 1-self.alpha else '❌'}")
        print(f"  Average Set Size:   {set_sizes.mean():.2f} "
              f"(1 = perfect, 7 = uncertain about everything)")
        print(f"  Singleton Rate:     {singleton_rate:.3f} "
              f"({singleton_rate*100:.1f}% of predictions are confident single-class)")

        print(f"\n  Per-Class Coverage:")
        print(f"  {'Class':<12} {'Coverage':>10} {'Status':>8}")
        print(f"  {'-'*12} {'-'*10} {'-'*8}")
        for cls, cov in class_coverage.items():
            name   = class_names[cls] if cls < len(class_names) else str(cls)
            status = "✅" if cov >= 1 - self.alpha else "⚠️"
            print(f"  {name:<12} {cov:>10.4f} {status:>8}")

        return {
            'coverage':        float(coverage),
            'avg_set_size':    float(set_sizes.mean()),
            'singleton_rate':  float(singleton_rate),
            'class_coverage':  {int(k): float(v)
                                 for k, v in class_coverage.items()},
            'q_hat':           float(self.q_hat),
            'alpha':           self.alpha,
        }
