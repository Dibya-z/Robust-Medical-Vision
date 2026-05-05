"""
=============================================================================
PHASE 3 ARCHITECTURE — EfficientNet-B3 Upgrade
=============================================================================
WHAT CHANGES FROM PHASE 2:

  Phase 2: EfficientNet-B1
    - 7.8M parameters
    - 1280-dim feature vector
    - Forced by M2 8GB RAM constraint

  Phase 3: EfficientNet-B3 (Lightning AI T4 — 16GB VRAM)
    - 12M parameters
    - 1536-dim feature vector  ← wider, richer features
    - Native resolution 300×300 (we use 224×224 for training consistency)
    - ~2% better ImageNet accuracy → better transfer learning base
    - Same architectural logic, just bigger and more capable

EVERYTHING ELSE IS IDENTICAL:
  - MCDropout: same design, stays active at inference
  - Evidential Head: same Dirichlet formulation
  - Two-stage training: same frozen→unfrozen strategy
  - Loss functions: same Focal + Evidential

The upgrade from B1 to B3 is the primary reason we expect
F1 to improve from 0.569 to ~0.72-0.76. Not a different approach —
just more capacity with the same principled design.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Dict, Tuple

NUM_CLASSES = 7


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout — always active regardless of model.eval().
    Identical to Phase 2. See Phase 2 architecture for full explanation.
    """
    def __init__(self, p: float = 0.4):
        super().__init__()
        self.p = p

    def forward(self, x):
        # training=True hardcoded — ignores model mode
        return F.dropout(x, p=self.p, training=True)

    def __repr__(self):
        return f'MCDropout(p={self.p}, always_active=True)'


class EvidentialHead(nn.Module):
    """
    Outputs Dirichlet α parameters for uncertainty decomposition.
    Identical to Phase 2. See Phase 2 architecture for full explanation.
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x) -> torch.Tensor:
        logits = self.fc(x)
        # softplus + 1 ensures alpha >= 1 (Dirichlet constraint)
        return F.softplus(logits) + 1


def compute_uncertainty_from_dirichlet(alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Same as Phase 2 — compute uncertainty from Dirichlet parameters.
    """
    S     = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    K     = alpha.shape[1]

    vacuity   = K / S.squeeze(1)
    aleatoric = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    epistemic = vacuity

    return {
        'probs':          probs,
        'aleatoric':      aleatoric,
        'epistemic':      epistemic,
        'total_evidence': S.squeeze(1),
        'vacuity':        vacuity,
    }


class RobustMedicalClassifierV2(nn.Module):
    """
    Phase 3 upgrade: EfficientNet-B3 backbone.

    Key differences from Phase 2 (V1):
      - Backbone: B1 (1280-dim) → B3 (1536-dim)
      - Head input: 1280 → 1536
      - Batch size: 16 → 64 (T4 allows this)
      - MC passes: 15 → 30 (T4 is faster)
      - Everything else identical by design

    WHY identical design for comparable ablation:
    The ablation table compares Model A, B, C on the same dataset.
    Model B (Phase 3) must be directly comparable to the GP baseline.
    Keeping the same architectural logic means improvements in the
    ablation table come from the hybrid, not from architectural changes
    that aren't present in all three models.
    """

    def __init__(self,
                 num_classes:     int  = NUM_CLASSES,
                 freeze_backbone: bool = True,
                 dropout_p1:      float = 0.4,
                 dropout_p2:      float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone: EfficientNet-B3 ─────────────────────────────────
        self.backbone  = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        feature_dim    = self.backbone.classifier[1].in_features  # 1536
        self.backbone.classifier = nn.Identity()

        self.feature_dim = feature_dim   # 1536 for B3

        if freeze_backbone:
            self._freeze_backbone()
            print(f"  Backbone: EfficientNet-B3 — FROZEN (Stage 1)")
        else:
            print(f"  Backbone: EfficientNet-B3 — TRAINABLE (Stage 2)")

        # ── Classification Head ───────────────────────────────────────
        # 1536 → 512 → 256
        # Same funnel design as Phase 2, adjusted for 1536-dim input
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            MCDropout(p=dropout_p1),
            nn.Linear(512, 256),
            nn.GELU(),
            MCDropout(p=dropout_p2),
        )

        # ── Standard classification output ────────────────────────────
        self.classifier = nn.Linear(256, num_classes)

        # ── Evidential output ─────────────────────────────────────────
        self.evidential = EvidentialHead(256, num_classes)

        # ── Weight initialisation ─────────────────────────────────────
        self._init_head_weights()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        self._freeze_backbone()
        print("  Backbone FROZEN")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  Backbone UNFROZEN — use 10× smaller LR for backbone params")

    def _init_head_weights(self):
        """Kaiming initialisation for GELU activation."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            logits:   (B, 7) raw class scores
            alpha:    (B, 7) Dirichlet parameters
            features: (B, 256) intermediate features for Grad-CAM + Mahalanobis
        """
        backbone_feats = self.backbone(x)          # (B, 1536)
        head_out       = self.head(backbone_feats)  # (B, 256)
        logits         = self.classifier(head_out)  # (B, 7)
        alpha          = self.evidential(head_out)   # (B, 7)

        return {
            'logits':   logits,
            'alpha':    alpha,
            'features': head_out,
        }

    @torch.no_grad()
    def predict_with_uncertainty(self,
                                  x:        torch.Tensor,
                                  n_passes: int = 30) -> Dict[str, torch.Tensor]:
        """
        MC Dropout inference: 30 forward passes.

        Phase 3 uses 30 passes (vs 15 in Phase 2).
        T4 GPU makes this fast enough — ~0.3s per image vs ~1.5s on M2.

        Returns comprehensive uncertainty breakdown.
        """
        # model.eval() keeps BatchNorm stable.
        # MCDropout ignores model mode — stays active regardless.
        self.eval()

        all_probs = []
        all_alpha = []

        for _ in range(n_passes):
            out = self.forward(x)
            all_probs.append(F.softmax(out['logits'], dim=1).unsqueeze(0))
            all_alpha.append(out['alpha'].unsqueeze(0))

        all_probs = torch.cat(all_probs, dim=0)   # (n_passes, B, 7)
        all_alpha = torch.cat(all_alpha, dim=0)   # (n_passes, B, 7)

        mean_probs     = all_probs.mean(dim=0)    # (B, 7)
        mc_variance    = all_probs.var(dim=0)     # (B, 7)
        mc_uncertainty = mc_variance.mean(dim=1)  # (B,)

        mean_alpha     = all_alpha.mean(dim=0)
        evidential     = compute_uncertainty_from_dirichlet(mean_alpha)

        return {
            'mean_probs':      mean_probs,
            'mc_uncertainty':  mc_uncertainty,
            'predicted_class': mean_probs.argmax(dim=1),
            'confidence':      mean_probs.max(dim=1).values,
            **{f'ev_{k}': v for k, v in evidential.items()},
        }


def model_summary_v2(model: RobustMedicalClassifierV2):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("MODEL SUMMARY — Phase 3 (EfficientNet-B3)")
    print("=" * 60)
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable:            {trainable:>12,}")
    print(f"  Frozen:               {total-trainable:>12,}")
    print(f"\n  Backbone:  EfficientNet-B3")
    print(f"  Features:  {model.feature_dim}-dim → head → 256-dim")
    print(f"  Head:      Linear(1536→512) + BN + GELU + MCDrop(0.4)")
    print(f"             Linear(512→256)  + GELU + MCDrop(0.3)")
    print(f"  Output 1:  Standard classifier → 7 logits")
    print(f"  Output 2:  Evidential head    → 7 Dirichlet α")
    print(f"  MC passes: 30 at inference (T4 GPU, ~0.3s/image)")
    print("=" * 60)
