"""
=============================================================================
STEP 5: TRAINING LOOP
=============================================================================
WHY A CAREFULLY DESIGNED TRAINING LOOP?

Training is not just "run model.forward() and call loss.backward()."
Four things need to work together correctly:

  1. Two-stage training: frozen backbone → unfrozen fine-tuning
     (prevents catastrophic forgetting of ImageNet features)

  2. Gradient accumulation: simulates larger batch sizes on small RAM
     (batch 16 × 2 accumulation = effective batch 32)

  3. Discriminative learning rates: backbone gets smaller LR than head
     (backbone knows things — update gently; head is new — update boldly)

  4. Early stopping: stop when val F1 stops improving
     (prevents overfitting on 8,000 training images)

Each of these has a specific, justified reason for existing.
=============================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from collections import defaultdict
import json


# =============================================================================
# STEP 5A: DEVICE SETUP
# =============================================================================
# WHY MPS and not CUDA?
# Your MacBook M2 has no NVIDIA GPU. Apple provides the MPS (Metal Performance
# Shaders) backend for PyTorch which routes computation to M2's GPU cores.
# MPS is 5-10× faster than CPU for matrix operations.
#
# Without this, training would take ~40-60 hours. With MPS: 6-12 hours.

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"  Device: Apple M2 GPU (MPS) ✅ — training will use GPU cores")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  Device: CUDA GPU ✅")
    else:
        device = torch.device('cpu')
        print(f"  Device: CPU ⚠️  — training will be slow (~10× slower than MPS)")
    return device


# =============================================================================
# STEP 5B: OPTIMIZER SETUP — DISCRIMINATIVE LEARNING RATES
# =============================================================================
# WHY different learning rates for backbone vs head?
#
# The backbone (EfficientNet-B1) has weights that were trained on 1.2 million
# ImageNet images. These weights encode rich visual knowledge.
#
# If we update backbone weights with the same LR as the head:
#   - Head LR = 1e-3 (needs large steps to learn from scratch)
#   - Backbone with LR 1e-3 → catastrophic forgetting — ImageNet features erased
#
# Instead:
#   - Head:     LR = 1e-3  (learning new medical classification patterns)
#   - Backbone: LR = 1e-4  (gently adapting to skin lesion textures)
#
# This ratio (10×) is standard in transfer learning literature.
# The exact values matter less than the ratio.
#
# WHY AdamW and not SGD?
# AdamW (Adam with decoupled weight decay):
#   - Adaptive learning rates per parameter → faster convergence
#   - Weight decay correctly applied (not mixed with gradient in regular Adam)
#   - Better generalization than vanilla Adam on vision tasks
# SGD with momentum can match AdamW with careful tuning, but takes longer.
# For a project with a deadline, AdamW is the right choice.

def build_optimizer(model, head_lr: float = 1e-3, backbone_lr: float = 1e-4,
                    weight_decay: float = 1e-4):
    """
    Build AdamW optimizer with discriminative learning rates.
    
    Groups:
      - Backbone parameters: lr = backbone_lr (gentle updates)
      - Head + classifier + evidential: lr = head_lr (bold updates)
    """
    backbone_params = list(model.backbone.parameters())
    head_params     = (
        list(model.head.parameters()) +
        list(model.classifier.parameters()) +
        list(model.evidential.parameters())
    )

    optimizer = AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params,     'lr': head_lr},
    ], weight_decay=weight_decay)

    print(f"\n  Optimizer: AdamW")
    print(f"    Backbone LR: {backbone_lr} (frozen in Stage 1, gentle in Stage 2)")
    print(f"    Head LR:     {head_lr}")
    print(f"    Weight decay: {weight_decay}")

    return optimizer


# =============================================================================
# STEP 5C: LEARNING RATE SCHEDULER
# =============================================================================
# WHY CosineAnnealingWarmRestarts?
#
# The loss landscape of a neural network has many local minima and saddle points.
# A learning rate that only decreases monotonically gets trapped in the first
# decent minimum it finds.
#
# Cosine annealing + warm restarts:
#   - LR follows a cosine curve from max → min
#   - Then RESTARTS at a high LR (warm restart)
#   - Allows optimizer to escape local minima periodically
#   - Tends to find flatter minima → better generalization
#
# Flat minima matter for uncertainty estimation:
#   A model in a flat minimum is less sensitive to small input changes
#   → more stable predictions → better-calibrated uncertainty.
#
# T_0 = 10 means: restart every 10 epochs. 
# For a 30-epoch training run, you get ~3 cycles.

def build_scheduler(optimizer, T_0: int = 10, T_mult: int = 1,
                    eta_min: float = 1e-6):
    """
    Cosine annealing with warm restarts.
    
    Args:
        T_0:    epochs for first restart cycle
        T_mult: multiply cycle length after each restart (1 = constant length)
        eta_min: minimum learning rate
    """
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )
    print(f"  Scheduler: CosineAnnealingWarmRestarts(T_0={T_0}, T_mult={T_mult})")
    return scheduler


# =============================================================================
# STEP 5D: METRICS COMPUTATION
# =============================================================================
# WHY these specific metrics and not accuracy?
#
# Accuracy is misleading with class imbalance. If 67% of data is nevus,
# a model that ALWAYS predicts nevus achieves 67% accuracy.
#
# F1-Score (macro): average F1 across all classes, treating each equally.
#   Punishes the model for ignoring rare classes.
#   "Macro" means each class contributes equally regardless of size.
#
# AUROC: Area Under the ROC Curve.
#   Measures how well the model ranks positive vs negative samples.
#   Threshold-independent — works at all operating points.
#   1.0 = perfect, 0.5 = random. Good classifiers: > 0.85.
#
# Both metrics are standard in medical imaging literature.
# You can directly compare your results to published HAM10000 papers.

def compute_metrics(all_preds: list, all_labels: list,
                    all_probs: list, num_classes: int = 7) -> dict:
    """
    Compute F1 (macro), per-class F1, and AUROC.
    
    Args:
        all_preds:  list of predicted class indices
        all_labels: list of true class indices
        all_probs:  list of probability vectors (num_classes,)
        num_classes: total classes
    
    Returns:
        dict with metrics
    """
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    probs  = np.array(all_probs)   # (n_samples, num_classes)

    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    # AUROC: needs probability scores, one-vs-rest
    try:
        auroc = roc_auc_score(
            labels, probs,
            multi_class='ovr', average='macro'
        )
    except ValueError:
        # Happens if a class has no positive samples in the batch
        auroc = 0.0

    return {
        'f1_macro':     f1_macro,
        'f1_per_class': f1_per_class.tolist(),
        'auroc':        auroc,
    }


# =============================================================================
# STEP 5E: SINGLE EPOCH — TRAIN
# =============================================================================
# WHY gradient accumulation?
#
# With batch size 16 (safe for M2 8GB), gradient estimates are noisier
# than batch size 32. Gradient accumulation computes gradients over
# multiple small batches before updating weights.
#
# accumulation_steps = 2 means:
#   Pass 1 (batch 0-15):  compute loss / 2, backward → accumulate gradients
#   Pass 2 (batch 16-31): compute loss / 2, backward → accumulate gradients
#   Then: optimizer.step() → update weights
#   Result: mathematically equivalent to batch size 32
#
# WHY divide loss by accumulation_steps?
# Because the gradient magnitudes would otherwise be 2× what they'd be
# with a real batch size 32. Dividing normalizes them correctly.

def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn,
                    device, epoch, max_epochs,
                    accumulation_steps: int = 2) -> dict:
    """
    Train for one epoch with gradient accumulation.
    
    Returns:
        dict with loss components and basic metrics
    """
    model.train()

    total_loss = 0.0
    focal_loss_sum = 0.0
    ev_loss_sum    = 0.0
    all_preds, all_labels, all_probs = [], [], []
    n_batches = len(dataloader)

    optimizer.zero_grad()   # clear gradients at epoch start

    for batch_idx, (images, labels, _meta) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # ── Forward pass ──────────────────────────────────────────────────────
        output = model(images)

        # ── Compute loss ──────────────────────────────────────────────────────
        loss_dict = loss_fn(output, labels, epoch=epoch, max_epochs=max_epochs)
        
        # Divide by accumulation_steps to normalize gradient magnitude
        loss = loss_dict['total'] / accumulation_steps

        # ── Backward pass ─────────────────────────────────────────────────────
        loss.backward()

        # ── Gradient accumulation step ────────────────────────────────────────
        # Only update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == n_batches:
            # Gradient clipping: prevents exploding gradients
            # WHY max_norm=1.0: standard value. Clips gradient vector norm.
            # Especially important when backbone is unfrozen — gradients
            # from deep layers can become large.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(epoch + batch_idx / n_batches)

        # ── Collect predictions ───────────────────────────────────────────────
        with torch.no_grad():
            probs = torch.softmax(output['logits'], dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        total_loss     += loss_dict['total'].item()
        focal_loss_sum += loss_dict['focal']
        ev_loss_sum    += loss_dict['evidential']

        # Progress print every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{n_batches} | "
                  f"Loss: {loss_dict['total'].item():.4f} | "
                  f"Focal: {loss_dict['focal']:.4f} | "
                  f"Ev: {loss_dict['evidential']:.4f}")

    metrics = compute_metrics(all_preds, all_labels, all_probs)
    metrics.update({
        'loss':          total_loss / n_batches,
        'focal_loss':    focal_loss_sum / n_batches,
        'ev_loss':       ev_loss_sum / n_batches,
    })
    return metrics


# =============================================================================
# STEP 5F: SINGLE EPOCH — VALIDATION
# =============================================================================
# WHY torch.no_grad() during validation?
# During validation, we don't need to compute gradients (we're not updating
# weights). torch.no_grad() disables gradient tracking → uses less memory →
# allows larger batch size during validation.
#
# WHY model.eval() during validation but model.train() for MC Dropout?
# model.eval() is used for standard validation to get clean predictions.
# For uncertainty estimation, we separately call model.train() to keep
# MC Dropout active. These are two different evaluation modes with
# different purposes.

@torch.no_grad()
def validate(model, dataloader, loss_fn, device,
             epoch, max_epochs) -> dict:
    """
    Validate model on val/test set.
    Standard forward pass (no MC Dropout, no grad).
    """
    model.eval()

    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels, _meta in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        output    = model(images)
        loss_dict = loss_fn(output, labels, epoch=epoch, max_epochs=max_epochs)

        probs = torch.softmax(output['logits'], dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        total_loss += loss_dict['total'].item()

    metrics = compute_metrics(all_preds, all_labels, all_probs)
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


# =============================================================================
# STEP 5G: EARLY STOPPING
# =============================================================================
# WHY early stopping?
# With only ~8,000 training images and a model with millions of parameters,
# overfitting is a real risk. After a certain number of epochs, train loss
# keeps falling but val F1 stops improving or starts degrading.
# Training beyond this point just memorizes the training data.
#
# We monitor val F1 (not val loss) because F1 is our actual clinical metric.
# The model should stop when it stops getting better at what we care about.
#
# patience=7: allow 7 epochs of no improvement before stopping.
# This gives the scheduler's warm restarts a chance to escape local minima.

class EarlyStopping:
    """
    Monitor val_f1_macro. Stop if no improvement for 'patience' epochs.
    Save best model checkpoint automatically.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4,
                 checkpoint_path: str = './outputs/best_model.pth'):
        self.patience         = patience
        self.min_delta        = min_delta
        self.checkpoint_path  = checkpoint_path
        self.best_f1          = -np.inf
        self.epochs_no_improve = 0
        self.should_stop      = False

    def step(self, val_f1: float, model, optimizer, epoch: int):
        """
        Call after each epoch. Saves checkpoint if improved.
        Returns True if training should stop.
        """
        if val_f1 > self.best_f1 + self.min_delta:
            self.best_f1           = val_f1
            self.epochs_no_improve = 0
            # Save best model
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1':              val_f1,
            }, self.checkpoint_path)
            print(f"    ✅ New best val F1: {val_f1:.4f} — checkpoint saved")
        else:
            self.epochs_no_improve += 1
            print(f"    No improvement for {self.epochs_no_improve}/{self.patience} epochs "
                  f"(best F1: {self.best_f1:.4f})")
            if self.epochs_no_improve >= self.patience:
                self.should_stop = True
                print(f"    ⏹  Early stopping triggered at epoch {epoch}")

        return self.should_stop


# =============================================================================
# STEP 5H: FULL TRAINING PIPELINE — TWO STAGES
# =============================================================================

def train(model, train_loader, val_loader, loss_fn, device,
          output_dir: str = './outputs',
          stage1_epochs: int = 5,
          stage2_epochs: int = 25,
          batch_size: int = 16):
    """
    Full two-stage training:
    
    STAGE 1 (epochs 1-5): Frozen backbone
      - Only head layers are trained
      - WHY: Let the head adapt to medical domain before touching backbone
      - LR: head=1e-3, backbone=0 (frozen)
      - Fast: ~45 minutes on M2
    
    STAGE 2 (epochs 6-30): Unfrozen fine-tuning
      - Entire network is trained
      - WHY: Now backbone can gently shift its features toward skin textures
      - LR: head=5e-4 (smaller — already partially trained), backbone=5e-5
      - Slower: ~4-5 hours on M2
    """
    os.makedirs(output_dir, exist_ok=True)
    history = defaultdict(list)

    print("\n" + "=" * 60)
    print("STAGE 1: Frozen Backbone Training")
    print(f"  Epochs: 1 to {stage1_epochs}")
    print("  WHY: Head learns medical features; backbone stays intact")
    print("=" * 60)

    # ── Stage 1: Frozen backbone ──────────────────────────────────────────────
    model.freeze_backbone() if hasattr(model, 'freeze_backbone') else None
    optimizer  = build_optimizer(model, head_lr=1e-3, backbone_lr=0.0)
    scheduler  = build_scheduler(optimizer, T_0=stage1_epochs)
    early_stop = EarlyStopping(
        patience=5,
        checkpoint_path=os.path.join(output_dir, 'stage1_best.pth')
    )

    for epoch in range(1, stage1_epochs + 1):
        print(f"\n  Epoch {epoch}/{stage1_epochs}")
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            device, epoch, stage1_epochs
        )
        val_metrics = validate(
            model, val_loader, loss_fn, device, epoch, stage1_epochs
        )

        elapsed = time.time() - t0
        _log_epoch(epoch, train_metrics, val_metrics, elapsed, history, stage='1')
        
        if early_stop.step(val_metrics['f1_macro'], model, optimizer, epoch):
            break

    # ── Stage 2: Unfrozen fine-tuning ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Full Fine-Tuning (Backbone Unfrozen)")
    print(f"  Epochs: {stage1_epochs+1} to {stage1_epochs + stage2_epochs}")
    print("  WHY: Backbone gently shifts toward skin lesion features")
    print("  WHY smaller LR: prevent catastrophic forgetting")
    print("=" * 60)

    # Load best stage 1 weights before starting stage 2
    ckpt = torch.load(os.path.join(output_dir, 'stage1_best.pth'), 
                      map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    model.unfreeze_backbone()
    # Smaller LRs for stage 2 — model already partially trained
    optimizer  = build_optimizer(model, head_lr=5e-4, backbone_lr=5e-5)
    scheduler  = build_scheduler(optimizer, T_0=10)
    early_stop = EarlyStopping(
        patience=7,
        checkpoint_path=os.path.join(output_dir, 'best_model.pth')
    )

    for epoch in range(stage1_epochs + 1, stage1_epochs + stage2_epochs + 1):
        print(f"\n  Epoch {epoch}/{stage1_epochs + stage2_epochs}")
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            device, epoch, stage1_epochs + stage2_epochs
        )
        val_metrics = validate(
            model, val_loader, loss_fn, device,
            epoch, stage1_epochs + stage2_epochs
        )

        elapsed = time.time() - t0
        _log_epoch(epoch, train_metrics, val_metrics, elapsed, history, stage='2')

        if early_stop.step(val_metrics['f1_macro'], model, optimizer, epoch):
            break

    # Save training history for later plotting
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(dict(history), f, indent=2)
    print(f"\n  Training history saved to: {history_path}")
    print(f"  Best model saved to: {output_dir}/best_model.pth")

    return history


def _log_epoch(epoch, train_m, val_m, elapsed, history, stage):
    """Print and record epoch results."""
    print(f"\n  ─── Epoch {epoch} Results (Stage {stage}) ───")
    print(f"  Train  → Loss: {train_m['loss']:.4f} | F1: {train_m['f1_macro']:.4f} | AUROC: {train_m['auroc']:.4f}")
    print(f"  Val    → Loss: {val_m['loss']:.4f}   | F1: {val_m['f1_macro']:.4f} | AUROC: {val_m['auroc']:.4f}")
    print(f"  Time:  {elapsed:.1f}s")

    for key in ['loss', 'f1_macro', 'auroc', 'focal_loss', 'ev_loss']:
        if key in train_m:
            history[f'train_{key}'].append(train_m[key])
        if key in val_m:
            history[f'val_{key}'].append(val_m[key])
