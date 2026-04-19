"""
=============================================================================
STEP 4: LOSS FUNCTIONS
=============================================================================
WHY CUSTOM LOSS FUNCTIONS?

Standard cross-entropy has two problems for this project:

Problem 1 — Class Imbalance:
  Cross-entropy treats all correct/wrong predictions equally.
  But ~67% of HAM10000 is nevus. A model that ALWAYS predicts nevus
  achieves 67% accuracy while being clinically useless.
  Focal Loss down-weights easy correct predictions (nevus) so the model
  is forced to focus on the rare, hard cases (melanoma, dermatofibroma).

Problem 2 — No uncertainty penalty:
  Cross-entropy penalizes wrong predictions but doesn't care if the
  model was wrongly confident. We want to penalize "confidently wrong"
  more than "uncertainly wrong."
  Evidential loss does this — it penalizes evidence for wrong classes.

COMBINED LOSS = Focal Loss + λ × Evidential Loss

Both losses point toward the same goal: correct predictions with
calibrated confidence. They complement each other.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# STEP 4A: FOCAL LOSS
# =============================================================================
# ORIGINAL PAPER: Lin et al., "Focal Loss for Dense Object Detection" (2017)
# 
# Standard cross-entropy:
#   CE(p, y) = -log(p_y)     where p_y = probability of true class
#
# Focal Loss adds a modulating factor:
#   FL(p, y) = -(1 - p_y)^γ × log(p_y)
#
# WHY this works for class imbalance:
# 
# For an EASY example (nevus, well-classified, p_y = 0.95):
#   (1 - 0.95)^2 = 0.0025   → loss is multiplied by 0.0025 → near zero
#
# For a HARD example (melanoma, rarely seen, p_y = 0.2):
#   (1 - 0.2)^2 = 0.64      → loss is multiplied by 0.64 → substantial
#
# The model's gradient is dominated by hard, rare examples — not by easy,
# common ones. This is exactly what we want.
#
# WHY γ = 2?
# The original paper found γ = 2 to be optimal across many experiments.
# γ = 0 reduces to standard cross-entropy. γ > 5 focuses too hard on
# extreme outliers and becomes unstable. γ = 2 is the standard choice.
#
# WHY alpha weights?
# An additional per-class weighting factor α.
# Typically set to inverse class frequency — rare classes get higher weight.
# This is on top of the focal modulation.

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p, y) = -α_y × (1 - p_y)^γ × log(p_y)
    
    Args:
        gamma:        focusing parameter (default=2, from original paper)
        alpha:        per-class weight tensor or None for uniform weights
        reduction:    'mean' or 'sum'
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha      # (num_classes,) or None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch_size, num_classes) — raw logits (before softmax)
            targets: (batch_size,) — integer class labels
        """
        # Cross-entropy gives log-probabilities: log(p_y)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Gather log probability of the TRUE class for each sample
        # ce_loss[i] = -log(p_{y_i}) for sample i
        ce_loss = F.nll_loss(log_probs, targets, weight=self.alpha, reduction='none')

        # Get probability of true class: p_y = exp(log(p_y))
        probs    = torch.exp(log_probs)
        p_t      = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Apply focal modulation: (1 - p_y)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Focal loss = focal_weight × cross_entropy
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def compute_focal_alpha(class_counts: dict, num_classes: int,
                         device: torch.device) -> torch.Tensor:
    """
    Compute per-class alpha weights as inverse class frequency.
    
    WHY: Even with focal modulation, very rare classes (115 dermatofibroma)
    benefit from an explicit upweighting. The combination of focal + alpha
    is more powerful than either alone.
    
    Args:
        class_counts: {class_idx: count} dictionary
        num_classes:  total number of classes
        device:       torch device
    
    Returns:
        alpha tensor of shape (num_classes,), normalized to sum to 1
    """
    total = sum(class_counts.values())
    alpha = torch.zeros(num_classes)

    for cls_idx, count in class_counts.items():
        alpha[cls_idx] = total / (num_classes * count)

    # Normalize
    alpha = alpha / alpha.sum()
    return alpha.to(device)


# =============================================================================
# STEP 4B: EVIDENTIAL LOSS (Dirichlet Loss)
# =============================================================================
# PAPER: Sensoy et al., "Evidential Deep Learning to Quantify Classification
#        Uncertainty" (NeurIPS 2018)
#
# INTUITION:
# Standard cross-entropy says: "predict class y."
# Evidential loss says: "accumulate evidence for class y AND remove
#                        evidence for wrong classes."
#
# The model is penalized for two things simultaneously:
#   (1) Not predicting the right class
#   (2) Being confident about wrong classes
#
# FORMULA (simplified):
#   L_evidential = sum over classes k of:
#     E[log(S) - log(alpha_k)] for correct class k = y
#   + KL divergence penalty for maintaining uncertainty on wrong classes
#
# WHERE:
#   alpha:  Dirichlet parameters predicted by model
#   S:      sum(alpha) = total evidence
#   KL:     penalty that pushes evidence for wrong classes toward 0
#
# WHY KL divergence penalty?
# Without it, the model might "play it safe" by putting moderate evidence
# everywhere (high alpha for all classes). The KL term penalizes this —
# it says: "if you're wrong about a class, don't accumulate evidence for it."
# This produces SPARSE evidence — concentrated only where the model
# actually has reason to believe a class is correct.
#
# WHY annealing coefficient λ(t)?
# In early training, the KL penalty can be too strong — it overwhelms the
# classification signal. We start with λ = 0 and gradually increase it.
# This lets the model learn class patterns first, then learn to be
# well-calibrated. This is the same intuition as β-VAE training.

class EvidentialLoss(nn.Module):
    """
    Evidential (Dirichlet) loss for uncertainty-aware training.
    
    Combines:
      1. Type II Maximum Likelihood loss (fit the correct class)
      2. KL divergence regularizer (penalize wrong-class evidence)
    
    Args:
        num_classes:  number of output classes
        annealing:    if True, gradually increase KL weight during training
    """

    def __init__(self, num_classes: int = 7, annealing: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.annealing   = annealing

    def forward(self, alpha: torch.Tensor, targets: torch.Tensor,
                epoch: int = 1, max_epochs: int = 30) -> torch.Tensor:
        """
        Args:
            alpha:      (batch, num_classes) — Dirichlet params, all > 0
            targets:    (batch,) — integer class labels
            epoch:      current epoch (for KL annealing)
            max_epochs: total epochs (for KL annealing)
        """
        # One-hot encode targets
        # WHY: Dirichlet loss operates on probability vectors, not integers
        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()

        S = alpha.sum(dim=1, keepdim=True)   # total evidence: (batch, 1)

        # ── Type II Maximum Likelihood Loss ──────────────────────────────────
        # This is the "fit the data" term.
        # It maximizes the probability of generating the true class label
        # under the Dirichlet distribution.
        # Equivalent to: minimize E[log(S) - log(alpha_y)]
        # WHY log: working in log-space is numerically stable
        loss = (y_onehot * (torch.log(S) - torch.log(alpha))).sum(dim=1)

        # ── KL Divergence Regularization ─────────────────────────────────────
        # This penalizes the model for accumulating evidence for WRONG classes.
        # Without this, the model could have high alpha everywhere (hedge).
        # With this, it must focus evidence on the correct class.
        #
        # KL between Dirichlet(alpha_wrong) and Dirichlet(uniform)
        # We only penalize WRONG class evidence:
        # alpha_tilde = 1 + (1 - y_onehot) × (alpha - 1)
        # This removes evidence for the correct class and only looks at
        # how much wrong-class evidence was accumulated.
        alpha_tilde = y_onehot + (1 - y_onehot) * alpha

        kl_loss = self._dirichlet_kl_divergence(alpha_tilde)

        # ── KL Annealing ──────────────────────────────────────────────────────
        # WHY: Early in training, KL penalty destroys the gradient signal.
        # We start at 0 and linearly increase to 1 over training.
        if self.annealing:
            lambda_t = min(1.0, epoch / (max_epochs / 2))
        else:
            lambda_t = 1.0

        total_loss = (loss + lambda_t * kl_loss).mean()
        return total_loss

    def _dirichlet_kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between Dirichlet(alpha) and Dirichlet(1,...,1).
        
        Closed-form solution using the lgamma function.
        This measures how far the predicted distribution is from
        "maximum uncertainty" (uniform Dirichlet).
        """
        K  = self.num_classes
        S  = alpha.sum(dim=1)   # (batch,)

        # Using the closed-form KL for Dirichlet distributions
        # KL = lgamma(S) - lgamma(K) - sum(lgamma(alpha_k))
        #      + sum((alpha_k - 1) * (digamma(alpha_k) - digamma(S)))
        
        ones = torch.ones_like(alpha)   # uniform Dirichlet: all alphas = 1

        kl = (
            torch.lgamma(S)
            - torch.lgamma(torch.tensor(float(K)).to(alpha.device))
            - torch.lgamma(alpha).sum(dim=1)
            + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S.unsqueeze(1)))).sum(dim=1)
        )
        return kl


# =============================================================================
# STEP 4C: COMBINED LOSS
# =============================================================================
# WHY combine both losses?
#
# Focal Loss:     "predict the RIGHT class, with emphasis on rare classes"
# Evidential Loss: "accumulate evidence for right class, penalize wrong-class evidence"
#
# They train complementary aspects of the same goal:
#   Focal     → WHAT to predict (class identity)
#   Evidential → HOW MUCH to believe it (uncertainty calibration)
#
# Without Focal:     model ignores rare classes
# Without Evidential: model is overconfident, uncertainty is uncalibrated
# With both:          correct AND calibrated predictions
#
# λ_evidential = 0.5 by default — a reasonable balance.
# You should tune this: try 0.2, 0.5, 1.0 and compare calibration curves.

class CombinedLoss(nn.Module):
    """
    Focal Loss + weighted Evidential Loss.
    
    L_total = L_focal + lambda_ev × L_evidential
    
    Args:
        num_classes:    number of classes
        gamma:          focal loss focusing parameter
        lambda_ev:      weight for evidential loss term
        class_counts:   {class_idx: count} for computing alpha weights
        device:         torch device
    """

    def __init__(self, num_classes: int = 7, gamma: float = 2.0,
                 lambda_ev: float = 0.5, class_counts: dict = None,
                 device: torch.device = None):
        super().__init__()

        self.lambda_ev = lambda_ev

        # Per-class alpha for focal loss
        alpha = None
        if class_counts and device:
            alpha = compute_focal_alpha(class_counts, num_classes, device)
            print(f"  Focal alpha (per-class weights): {alpha.cpu().numpy().round(3)}")

        self.focal_loss     = FocalLoss(gamma=gamma, alpha=alpha)
        self.evidential_loss = EvidentialLoss(num_classes=num_classes)

        print(f"  Combined Loss: Focal(γ={gamma}) + {lambda_ev}×Evidential")

    def forward(self, output: dict, targets: torch.Tensor,
                epoch: int = 1, max_epochs: int = 30) -> dict:
        """
        Args:
            output:     dict from model.forward() with 'logits' and 'alpha'
            targets:    (batch,) integer labels
            epoch/max_epochs: for KL annealing in evidential loss
        
        Returns:
            dict with individual and total losses for logging
        """
        logits = output['logits']
        alpha  = output['alpha']

        focal_loss = self.focal_loss(logits, targets)
        ev_loss    = self.evidential_loss(alpha, targets, epoch, max_epochs)
        total_loss = focal_loss + self.lambda_ev * ev_loss

        return {
            'total':     total_loss,
            'focal':     focal_loss.item(),
            'evidential': ev_loss.item(),
        }
