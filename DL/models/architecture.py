"""
=============================================================================
STEP 3: MODEL ARCHITECTURE
=============================================================================
WHY THIS ARCHITECTURE?

The model is built in three conceptual layers:

  Layer 1 — BACKBONE (EfficientNet-B1)
    → Extracts rich visual features from skin lesion images
    → Uses ImageNet pretrained weights so we don't train from scratch
    → Chosen over ResNet because it's more parameter-efficient

  Layer 2 — UNCERTAINTY HEAD (MC Dropout)
    → Keeps dropout ON during inference
    → Running 20 forward passes gives a distribution of predictions
    → The VARIANCE of this distribution = epistemic uncertainty
    → Catches inputs the model has never seen (OOD detection)

  Layer 3 — EVIDENTIAL HEAD (Dirichlet output)
    → Instead of outputting raw class scores, outputs Dirichlet parameters
    → From these parameters we separately compute:
        - Aleatoric uncertainty (noise in the image itself)
        - Epistemic uncertainty (model's ignorance)
    → A single forward pass gives both uncertainty types simultaneously

The combination answers: WHAT class is it? + HOW sure are we? + WHY unsure?
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple, Dict


NUM_CLASSES = 7   # HAM10000 has 7 disease classes


# =============================================================================
# STEP 3A: BACKBONE — EfficientNet-B1 with Pretrained Weights
# =============================================================================
# WHY EfficientNet and not ResNet/VGG?
#
# EfficientNet was designed by neural architecture search to find the
# OPTIMAL way to scale a network. It scales depth + width + resolution
# simultaneously with a fixed ratio.
#
# ResNet only scales depth. VGG is fixed. Both waste parameters.
# EfficientNet achieves ~2-4% better accuracy with the same parameter count.
#
# WHY pretrained on ImageNet?
# Training from scratch on 8,000 skin images (after split) would massively
# overfit. ImageNet pretraining gives us a backbone that already detects:
#   - Edges and textures (useful for lesion border analysis)
#   - Color patterns (useful for pigmentation analysis)
#   - Shapes and objects (useful for lesion morphology)
# We're not teaching the model to see — we're teaching it what to look for.
#
# WHY B1 over B3?
# B3 requires ~5-6GB of memory per training step on batch size 16.
# B1 requires ~2-2.5GB. Both give similar performance on 10K-image datasets.
# On your MacBook M2 with 8GB unified memory, B1 is the correct choice.

def build_backbone() -> Tuple[nn.Module, int]:
    """
    Build EfficientNet-B1 backbone with pretrained ImageNet weights.
    Returns: (backbone_module, feature_dimension)
    """
    # Load pretrained model
    backbone = models.efficientnet_b1(
        weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1
    )

    # Feature dimension of EfficientNet-B1's final layer = 1280
    feature_dim = backbone.classifier[1].in_features  # 1280

    # REMOVE the original classification head
    # WHY: The original head was designed for 1000 ImageNet classes.
    # We replace it with our own uncertainty-aware head.
    # We keep only the feature extractor part.
    backbone.classifier = nn.Identity()

    return backbone, feature_dim


# =============================================================================
# STEP 3B: MC DROPOUT LAYERS
# =============================================================================
# WHY we need a custom MC Dropout class:
#
# PyTorch's nn.Dropout has a training flag — it is ONLY active when
# model.train() is called. When model.eval() is called (during inference),
# dropout is automatically disabled.
#
# For MC Dropout, we WANT dropout to remain active during inference.
# This custom class ignores the training flag and is ALWAYS active.
#
# This is the key implementation detail that makes uncertainty estimation work.

class MCDropout(nn.Module):
    """
    Monte Carlo Dropout — active during BOTH training AND inference.
    
    Standard nn.Dropout turns off at model.eval().
    MCDropout stays on always — enabling uncertainty sampling.
    
    WHY: During inference, we run 20 forward passes.
    Each pass drops different neurons → slightly different model.
    The variance across 20 predictions = epistemic uncertainty.
    High variance = model hasn't seen inputs like this before.
    """
    def __init__(self, p: float = 0.4):
        super().__init__()
        self.p = p

    def forward(self, x):
        # training=True forces dropout regardless of model.eval() state
        return F.dropout(x, p=self.p, training=True)

    def __repr__(self):
        return f'MCDropout(p={self.p}) [always_active=True]'


# =============================================================================
# STEP 3C: EVIDENTIAL HEAD
# =============================================================================
# WHY Evidential Deep Learning?
#
# Standard softmax gives you probabilities that sum to 1.
# But these probabilities are RELATIVE — they don't tell you how
# much the model actually knows about this input.
#
# Example:
#   Image A (clear melanoma): softmax → [0.92, 0.05, 0.01, ...]
#   Image B (horse — OOD):    softmax → [0.88, 0.07, 0.03, ...]
#   Both look confident. Only Image A should be.
#
# Evidential DL models uncertainty as a Dirichlet distribution.
# The Dirichlet has parameters α (alpha) — one per class.
# Think of α as "evidence accumulated for each class."
#
# If the model has seen 1000 melanoma images:
#   → High α for melanoma → high confidence → low uncertainty
# If the model has never seen this input:
#   → All α near 1 → spread-out Dirichlet → high uncertainty
#
# From the Dirichlet parameters we can compute:
#   - Class probabilities:       p_k = α_k / sum(α)
#   - Total evidence:            S   = sum(α)
#   - Aleatoric uncertainty:     from the Dirichlet entropy
#   - Epistemic uncertainty:     inversely related to S

class EvidentialHead(nn.Module):
    """
    Outputs Dirichlet distribution parameters for uncertainty decomposition.
    
    Input:  feature vector (batch_size, feature_dim)
    Output: alpha parameters (batch_size, num_classes) — all positive
    
    WHY softplus activation?
    Dirichlet parameters must be > 0 (strictly positive).
    softplus(x) = log(1 + exp(x)) is always positive and smooth.
    ReLU would give exact zeros which cause numerical instability in the
    Dirichlet distribution calculation.
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x) -> torch.Tensor:
        logits = self.fc(x)
        # softplus ensures all alpha > 0 (Dirichlet constraint)
        alpha = F.softplus(logits) + 1   # +1 ensures alpha >= 1
        return alpha


def compute_uncertainty_from_dirichlet(alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Given Dirichlet parameters alpha, compute uncertainty measures.
    
    Args:
        alpha: (batch_size, num_classes) — all positive
    
    Returns dict with:
        - probs:               class probabilities
        - aleatoric:           data uncertainty (inherent ambiguity)
        - epistemic:           model uncertainty (lack of knowledge)
        - total_evidence:      total evidence accumulated
        - vacuity:             1/S — how "empty" the evidence is
    
    WHY these specific formulas?
    These come from the Subjective Logic framework (Sensoy et al., 2018).
    
    Vacuity = K / S where K = num_classes, S = sum(alpha)
    High vacuity means the model has very little evidence → uncertain
    
    Aleatoric = expected entropy of the categorical distribution
    This captures inherent ambiguity that more data won't resolve
    
    Epistemic = total uncertainty - aleatoric
    This is what more training data CAN fix
    """
    S     = alpha.sum(dim=1, keepdim=True)   # total evidence per sample
    probs = alpha / S                          # expected class probabilities

    num_classes = alpha.shape[1]

    # Vacuity: how "empty" is the evidence?
    # High vacuity = model barely knows anything about this input
    vacuity = num_classes / S.squeeze(1)

    # Aleatoric uncertainty: expected entropy of the distribution
    # sum(p_k * (digamma(S) - digamma(alpha_k)))
    # approximated as normalized entropy of probabilities
    aleatoric = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

    # Epistemic uncertainty: total - aleatoric
    # High when model lacks evidence (new input type)
    epistemic = vacuity   # vacuity IS the epistemic component in this framework

    return {
        'probs':          probs,
        'aleatoric':      aleatoric,
        'epistemic':      epistemic,
        'total_evidence': S.squeeze(1),
        'vacuity':        vacuity,
    }


# =============================================================================
# STEP 3D: FULL MODEL — PUTTING IT ALL TOGETHER
# =============================================================================

class RobustMedicalClassifier(nn.Module):
    """
    Full uncertainty-aware medical image classifier.
    
    Architecture Flow:
    
    Input (224×224×3)
        ↓
    [EfficientNet-B1 Backbone]   ← pretrained, fine-tuned
        ↓
    [Global Average Pooling]     ← 1280-dim feature vector
        ↓
    [Dense(512) + BatchNorm + GELU]
        ↓
    [MCDropout(p=0.4)]           ← ALWAYS active
        ↓
    [Dense(256) + GELU]
        ↓
    [MCDropout(p=0.3)]           ← ALWAYS active
        ↓
    ┌──────────┴──────────┐
    ↓                     ↓
    [Standard Head]   [Evidential Head]
    (8 logits)        (8 Dirichlet α)
    ↓                     ↓
    [Cross-entropy]   [Evidential Loss]
                          ↓
              [Uncertainty Decomposition]
    
    WHY BatchNorm after first dense layer?
    - Normalizes activations → stable gradient flow → faster convergence
    - Acts as regularizer → reduces overfitting
    - Especially important when fine-tuning: backbone outputs can have
      very different scale than randomly initialized head weights
    
    WHY GELU over ReLU?
    - GELU (Gaussian Error Linear Unit) is smoother than ReLU
    - Doesn't have "dead neuron" problem (neurons stuck at 0)
    - Empirically better in transformer literature, works well in CNNs too
    - Provides better gradient flow for fine-grained medical features
    
    WHY two dropout layers?
    - More MC sampling points = more stable variance estimate
    - First dropout (p=0.4) on larger layer = more diversity
    - Second dropout (p=0.3) on smaller layer = fine-grained variance
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = True):
        super().__init__()

        self.num_classes = num_classes

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone, feature_dim = build_backbone()

        if freeze_backbone:
            self._freeze_backbone()
            print("  Backbone FROZEN (Stage 1 training)")
        else:
            print("  Backbone UNFROZEN (Stage 2 fine-tuning)")

        # ── Classification Head ───────────────────────────────────────────────
        # WHY this specific architecture?
        # 1280 → 512 → 256 → 7 is a standard funnel design.
        # Each layer compresses features while extracting higher-level patterns.
        # Too shallow → can't combine features. Too deep → overfits.
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            MCDropout(p=0.4),          # uncertainty sampling point 1
            nn.Linear(512, 256),
            nn.GELU(),
            MCDropout(p=0.3),          # uncertainty sampling point 2
        )

        # ── Standard Classification Output ────────────────────────────────────
        self.classifier = nn.Linear(256, num_classes)

        # ── Evidential Output ─────────────────────────────────────────────────
        self.evidential = EvidentialHead(256, num_classes)

        # ── Weight initialization ─────────────────────────────────────────────
        # WHY Kaiming initialization?
        # Randomly initialized weights can cause vanishing/exploding gradients.
        # Kaiming initialization sets variance based on layer size so that
        # the signal magnitude is preserved through forward pass.
        # This is especially important with GELU activation.
        self._initialize_head_weights()

    def _freeze_backbone(self):
        """Internal: called during __init__."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """
        Public method: Freeze all backbone parameters.
        WHY: Pretrained weights are valuable. In Stage 1, we only train
        the head so the head can adapt to medical features without
        destroying the backbone's learned representations.
        This is called 'warm-up' training.
        """
        self._freeze_backbone()
        print("  Backbone FROZEN (Stage 1 training)")

    def unfreeze_backbone(self, lr_multiplier: float = 0.1):
        """
        Unfreeze backbone for Stage 2 fine-tuning.
        
        WHY lr_multiplier=0.1?
        When we unfreeze, we update backbone weights with a MUCH smaller
        learning rate than the head. The head is learning new things —
        the backbone is gently adjusting things it already knows.
        If we use the same LR for both, we'll overwrite the pretrained
        features (catastrophic forgetting).
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"  Backbone UNFROZEN — use {lr_multiplier}× LR for backbone params")

    def _initialize_head_weights(self):
        """Kaiming initialization for all linear layers in the head."""
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, 
                                        nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning both predictions and uncertainty.
        
        Args:
            x: input images (batch_size, 3, 224, 224)
        
        Returns dict with:
            - logits:    raw class scores for cross-entropy loss
            - alpha:     Dirichlet parameters for evidential loss
            - features:  intermediate features for Grad-CAM
        """
        # Extract features from backbone
        features = self.backbone(x)          # (batch, 1280)

        # Pass through head (with MC Dropout active)
        head_out = self.head(features)        # (batch, 256)

        # Standard classification branch
        logits = self.classifier(head_out)    # (batch, 7)

        # Evidential branch
        alpha = self.evidential(head_out)     # (batch, 7) — all positive

        return {
            'logits':   logits,
            'alpha':    alpha,
            'features': head_out,   # for Grad-CAM hook
        }

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor,
                                  n_passes: int = 20) -> Dict[str, torch.Tensor]:
        """
        Run MC Dropout inference: n_passes forward passes, compute statistics.
        
        WHY 20 passes?
        Empirically, variance estimates stabilize at ~15-20 passes.
        Beyond 30 you get diminishing returns but 2× the compute.
        20 is the efficient sweet spot.
        
        Returns:
            - mean_probs:         average prediction across passes
            - mc_uncertainty:     variance across passes (epistemic)
            - evidential_results: uncertainty from Dirichlet head
            - predicted_class:    argmax of mean_probs
            - confidence:         max of mean_probs
        """
        self.eval()   # CRITICAL: keeps MCDropout active
        # Note: we use torch.no_grad() so gradients aren't computed,
        # but model.train() is called to keep dropout ON.
        # This is the standard MC Dropout inference pattern.

        all_probs = []
        all_alpha = []

        for _ in range(n_passes):
            output = self.forward(x)
            probs  = F.softmax(output['logits'], dim=1)
            all_probs.append(probs.unsqueeze(0))   # (1, batch, 7)
            all_alpha.append(output['alpha'].unsqueeze(0))

        # Stack: (n_passes, batch, num_classes)
        all_probs = torch.cat(all_probs, dim=0)
        all_alpha = torch.cat(all_alpha, dim=0)

        # Mean and variance across MC passes
        mean_probs   = all_probs.mean(dim=0)              # (batch, 7)
        mc_variance  = all_probs.var(dim=0)               # (batch, 7)
        mc_uncertainty = mc_variance.mean(dim=1)           # (batch,) — scalar per sample

        # Evidential uncertainty from mean alpha
        mean_alpha = all_alpha.mean(dim=0)
        evidential = compute_uncertainty_from_dirichlet(mean_alpha)

        predicted_class = mean_probs.argmax(dim=1)
        confidence      = mean_probs.max(dim=1).values

        return {
            'mean_probs':      mean_probs,
            'mc_uncertainty':  mc_uncertainty,   # epistemic from MC
            'predicted_class': predicted_class,
            'confidence':      confidence,
            **{f'ev_{k}': v for k, v in evidential.items()},  # evidential results
        }


# =============================================================================
# STEP 3E: MODEL SUMMARY UTILITY
# =============================================================================

def model_summary(model: RobustMedicalClassifier):
    """Print a clean summary of trainable parameters."""
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen       = total_params - trainable

    print(f"  Total parameters:     {total_params:>10,}")
    print(f"  Trainable:            {trainable:>10,}")
    print(f"  Frozen (backbone):    {frozen:>10,}")
    print(f"\n  Architecture:")
    print(f"    Backbone:  EfficientNet-B1 (ImageNet pretrained)")
    print(f"    Head:      Linear(1280→512) → MCDrop(0.4) → Linear(512→256) → MCDrop(0.3)")
    print(f"    Output 1:  Standard classifier → 7 class logits")
    print(f"    Output 2:  Evidential head → 7 Dirichlet α parameters")
    print(f"    MC passes: 20 at inference time")
    print("=" * 60)
