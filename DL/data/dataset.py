"""
=============================================================================
STEP 1: DATA PIPELINE
=============================================================================
WHY THIS FILE EXISTS:
Everything in ML lives or dies by the quality of data preparation.
This file handles:
  - Loading HAM10000 images + metadata
  - Group-based train/val/test split (prevents data leakage)
  - Preprocessing and augmentation
  - Weighted sampling to handle class imbalance

Every decision here has a direct consequence on model performance and
on whether your evaluation is honest or inflated.
=============================================================================
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from collections import Counter


# =============================================================================
# STEP 1A: LABEL ENCODING
# =============================================================================
# WHY: Neural networks work with numbers, not strings.
# We map each disease name to an integer index 0-6.
# We also keep the reverse mapping (index → name) for visualization later.

CLASS_NAMES = {
    'nv':    0,   # Melanocytic Nevus        — most common, benign
    'mel':   1,   # Melanoma                 — malignant, dangerous
    'bkl':   2,   # Benign Keratosis
    'bcc':   3,   # Basal Cell Carcinoma     — malignant
    'akiec': 4,   # Actinic Keratosis        — precancerous
    'vasc':  5,   # Vascular Lesion          — rare, benign
    'df':    6,   # Dermatofibroma           — rarest, benign
}

IDX_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}

# Human-readable names for plots and reports
DISPLAY_NAMES = {
    'nv':    'Melanocytic Nevus',
    'mel':   'Melanoma',
    'bkl':   'Benign Keratosis',
    'bcc':   'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc':  'Vascular Lesion',
    'df':    'Dermatofibroma',
}


# =============================================================================
# STEP 1B: METADATA LOADER
# =============================================================================
# WHY: Before touching images, we load and understand the metadata CSV.
# HAM10000's metadata has: lesion_id, image_id, dx (diagnosis), age, sex,
# localization, dx_type.
#
# lesion_id is the critical column — multiple images can share one lesion_id
# (same physical lesion, different photos). We MUST split by lesion_id,
# not by image, to prevent the same lesion appearing in train AND test.
# Not doing this is the most common mistake in Kaggle HAM10000 kernels.

def load_metadata(metadata_path: str, images_dir: str) -> pd.DataFrame:
    """
    Load and validate HAM10000 metadata.
    
    Args:
        metadata_path: path to HAM10000_metadata.csv
        images_dir: path to folder containing all .jpg images
                    (merge HAM10000_images_part_1 and _part_2 into one folder)
    Returns:
        Clean DataFrame with image paths and labels
    """
    print("=" * 60)
    print("STEP 1B: Loading HAM10000 Metadata")
    print("=" * 60)

    df = pd.read_csv(metadata_path)
    print(f"  Total records in CSV: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Unique lesions (lesion_id): {df['lesion_id'].nunique()}")
    print(f"  Unique images (image_id):   {df['image_id'].nunique()}")

    # Build full image path for each row
    # WHY: We store the path now so the Dataset class just reads from it later
    df['image_path'] = df['image_id'].apply(
        lambda x: os.path.join(images_dir, x + '.jpg')
    )

    # Verify images actually exist on disk — catches missing files early
    missing = df[~df['image_path'].apply(os.path.exists)]
    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} images not found on disk. Dropping them.")
        df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    else:
        print(f"  All {len(df)} images found on disk. ✅")

    # Map string labels to integers
    df['label'] = df['dx'].map(CLASS_NAMES)

    # Handle missing metadata (age has ~57 missing values in HAM10000)
    # WHY: We fill missing age with median — not mean — because age is
    # approximately normally distributed and median is more robust to outliers
    df['age'] = df['age'].fillna(df['age'].median())
    df['sex'] = df['sex'].fillna('unknown')
    df['localization'] = df['localization'].fillna('unknown')

    print(f"\n  Class distribution:")
    for cls, count in sorted(df['dx'].value_counts().items()):
        pct = 100 * count / len(df)
        print(f"    {cls:6s} ({DISPLAY_NAMES[cls]:<25}): {count:5d} ({pct:.1f}%)")

    return df


# =============================================================================
# STEP 1C: TRAIN / VAL / TEST SPLIT — THE RIGHT WAY
# =============================================================================
# WHY GroupShuffleSplit and not random_split?
#
# HAM10000 has ~10,015 images but only ~7,470 unique lesions.
# ~2,500 images are duplicates of existing lesions (different angles/zoom).
#
# If you do a random 80/20 split:
#   - Image A (lesion_xyz, front view) → train set
#   - Image B (lesion_xyz, side view)  → test set
#   The model has effectively "seen" the test lesion. Accuracy is inflated.
#
# GroupShuffleSplit ensures ALL images of lesion_xyz go to the SAME split.
# This is the only honest evaluation for this dataset.

def split_dataset(df: pd.DataFrame, 
                  val_size: float = 0.15,
                  test_size: float = 0.15,
                  random_state: int = 42):
    """
    Split by lesion_id to prevent data leakage.
    Returns three DataFrames: train, val, test
    """
    print("\n" + "=" * 60)
    print("STEP 1C: Group-Based Train/Val/Test Split")
    print("WHY: Splitting by lesion_id prevents same lesion in train+test")
    print("=" * 60)

    groups = df['lesion_id'].values

    # First split off test set
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, 
                                  random_state=random_state)
    train_val_idx, test_idx = next(splitter.split(df, groups=groups))

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test      = df.iloc[test_idx].reset_index(drop=True)

    # Then split train/val from the remaining data
    groups_tv = df_train_val['lesion_id'].values
    val_ratio_adjusted = val_size / (1 - test_size)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio_adjusted,
                                   random_state=random_state)
    train_idx, val_idx = next(splitter2.split(df_train_val, groups=groups_tv))

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val   = df_train_val.iloc[val_idx].reset_index(drop=True)

    # Verify no leakage: no lesion_id should appear in two splits
    train_lesions = set(df_train['lesion_id'])
    val_lesions   = set(df_val['lesion_id'])
    test_lesions  = set(df_test['lesion_id'])

    assert len(train_lesions & val_lesions) == 0,  "LEAKAGE: train/val overlap!"
    assert len(train_lesions & test_lesions) == 0, "LEAKAGE: train/test overlap!"
    assert len(val_lesions & test_lesions) == 0,   "LEAKAGE: val/test overlap!"

    print(f"  Train: {len(df_train):5d} images | {df_train['lesion_id'].nunique()} unique lesions")
    print(f"  Val:   {len(df_val):5d} images | {df_val['lesion_id'].nunique()} unique lesions")
    print(f"  Test:  {len(df_test):5d} images | {df_test['lesion_id'].nunique()} unique lesions")
    print(f"  No data leakage detected ✅")

    return df_train, df_val, df_test


# =============================================================================
# STEP 1D: AUGMENTATION TRANSFORMS
# =============================================================================
# WHY different transforms for train vs val/test?
#
# TRAIN transforms: aggressive augmentation to improve generalization.
#   We want the model to see many variations of each lesion so it learns
#   the UNDERLYING DISEASE PATTERN, not surface-level artifacts like lighting.
#
# VAL/TEST transforms: only normalization + resize.
#   We evaluate on images as they'd appear in a real clinic — no artificial
#   augmentation. If we augmented test images, our metrics wouldn't reflect
#   real-world performance.
#
# WHY these specific ImageNet normalization values?
#   EfficientNet-B1 was pretrained on ImageNet using these exact statistics.
#   If we don't normalize to the same distribution, the pretrained weights
#   "see" a different world than what they were trained on — features break.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224   # EfficientNet-B1 input size


def get_train_transforms():
    """
    Aggressive augmentation for training.
    Each transform has a specific clinical justification.
    """
    return T.Compose([
        # Resize to model input size
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        # WHY RandomHorizontalFlip + RandomVerticalFlip:
        # A melanoma doesn't care about orientation. The dermatoscope can be
        # rotated in any direction. The model shouldn't either.
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),

        # WHY RandomRotation:
        # Same reason — clinical images are taken at various angles.
        T.RandomRotation(degrees=30),

        # WHY ColorJitter:
        # Different dermatoscopes produce different color profiles.
        # Different skin tones change background color. The model should
        # learn melanoma from SHAPE and TEXTURE, not from absolute color values.
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),

        # WHY RandomResizedCrop:
        # Lesions appear at different scales and positions. This teaches
        # the model scale and position invariance.
        T.RandomResizedCrop(
            size=IMAGE_SIZE,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),

        # WHY RandomGrayscale:
        # Forces the model to sometimes learn from texture/shape alone,
        # not relying on color — makes it more robust.
        T.RandomGrayscale(p=0.1),

        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """
    Minimal transforms for validation and test — no augmentation.
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# =============================================================================
# STEP 1E: PYTORCH DATASET CLASS
# =============================================================================
# WHY a custom Dataset class?
# PyTorch's DataLoader needs a Dataset object that defines:
#   __len__  → how many samples
#   __getitem__ → how to load ONE sample given its index
#
# The DataLoader then handles batching, shuffling, and multi-worker loading
# automatically. We never load all images into RAM at once — only one batch
# at a time. Critical for your 8GB RAM constraint.

class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000.
    
    Loads images on-demand (not all into RAM at once).
    Returns image tensor, integer label, and metadata for analysis.
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        """
        Args:
            dataframe: DataFrame with columns: image_path, label, age, sex, localization
            transform: torchvision transforms to apply
        """
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image from disk
        # WHY .convert('RGB'): Some images might be RGBA or grayscale.
        # EfficientNet expects exactly 3 channels. convert('RGB') ensures this.
        image = Image.open(row['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['label'], dtype=torch.long)

        # Return metadata too — needed for uncertainty analysis later
        # (does uncertainty correlate with age? localization?)
        meta = {
            'image_id':     row['image_id'],
            'lesion_id':    row['lesion_id'],
            'age':          row['age'],
            'sex':          row['sex'],
            'localization': row['localization'],
            'dx_type':      row['dx_type'],
        }

        return image, label, meta


# =============================================================================
# STEP 1F: WEIGHTED SAMPLER FOR CLASS IMBALANCE
# =============================================================================
# WHY WeightedRandomSampler and not just training normally?
#
# If you train on the raw distribution:
#   - Model sees ~6700 nevus images per epoch
#   - Model sees ~115 dermatofibroma images per epoch
#   The model will overwhelmingly optimize for nevus because that's where
#   most of its gradient signal comes from.
#
# WeightedRandomSampler assigns higher sampling probability to rare classes
# so effectively every class is seen roughly equally per epoch.
#
# WHY this over just duplicating rare class images?
#   Weighted sampling is cleaner — you're controlling the SAMPLING PROCESS,
#   not artificially inflating the dataset. Metrics remain honest.

def get_weighted_sampler(df_train: pd.DataFrame) -> WeightedRandomSampler:
    """
    Creates a sampler that oversamples rare classes proportionally.
    """
    class_counts = Counter(df_train['label'].values)
    total        = len(df_train)

    # Weight of each class = inverse of its frequency
    # Rare class → high weight → sampled more often
    class_weights = {
        cls: total / count for cls, count in class_counts.items()
    }

    # Assign weight to each individual sample
    sample_weights = [
        class_weights[label] for label in df_train['label'].values
    ]
    sample_weights = torch.FloatTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True   # sampling with replacement is required here
    )

    print("\n  Class weights for sampler:")
    for cls_idx, weight in sorted(class_weights.items()):
        cls_name = IDX_TO_CLASS[cls_idx]
        count    = class_counts[cls_idx]
        print(f"    {cls_name:6s}: {count:5d} samples → weight {weight:.2f}")

    return sampler


# =============================================================================
# STEP 1G: DATALOADER FACTORY
# =============================================================================
# WHY num_workers=2 and not more?
# Each worker is a separate process that loads images in parallel.
# On your M2 MacBook with 8GB RAM, too many workers eat memory.
# 2 workers is the safe sweet spot — fast enough, doesn't crash.
#
# WHY pin_memory=False for MPS?
# pin_memory=True speeds up GPU transfer on CUDA systems.
# On Apple MPS, it's not supported and causes errors. Keep it False.

def get_dataloaders(df_train, df_val, df_test, batch_size=16):
    """
    Create train, val, test DataLoaders.
    
    Batch size 16 is safe for EfficientNet-B1 on M2 8GB.
    Use gradient accumulation (in training loop) to simulate batch size 32.
    """
    print("\n" + "=" * 60)
    print("STEP 1G: Creating DataLoaders")
    print(f"  Batch size: {batch_size} (safe for M2 8GB)")
    print("=" * 60)

    train_dataset = HAM10000Dataset(df_train, transform=get_train_transforms())
    val_dataset   = HAM10000Dataset(df_val,   transform=get_val_transforms())
    test_dataset  = HAM10000Dataset(df_test,  transform=get_val_transforms())

    sampler = get_weighted_sampler(df_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = sampler,      # use weighted sampler instead of shuffle
        num_workers = 2,
        pin_memory  = False,        # False for Apple MPS
        drop_last   = True,         # drop last incomplete batch for stable BN
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size * 2,  # no grad during val, can use bigger batch
        shuffle     = False,
        num_workers = 2,
        pin_memory  = False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = False,
    )

    print(f"  Train batches:      {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches:       {len(test_loader)}")

    return train_loader, val_loader, test_loader
