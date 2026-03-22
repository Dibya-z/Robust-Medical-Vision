import numpy as np
import cv2
from pathlib import Path


def load_images(split, data_root, img_size=(128, 128), max_per_class=500):
    """
    Load chest X-ray images from a given split, preprocess, and return
    as numpy arrays ready for feature extraction.

    Preprocessing applied:
        1. Convert to grayscale
        2. Resize to img_size
        3. Normalize pixel values to [0.0, 1.0]

    Parameters
    ----------
    split         : str — one of 'train', 'val', 'test'
    data_root     : str — path to the chest_xray folder
    img_size      : tuple — (height, width) to resize every image to
    max_per_class : int or None — max images to load per class.
                    If None, loads all available images.

    Returns
    -------
    images : np.ndarray, shape (N, H, W), dtype float32
             Grayscale normalized images
    labels : np.ndarray, shape (N,), dtype int
             0 = NORMAL, 1 = PNEUMONIA
    """
    classes     = ['NORMAL', 'PNEUMONIA']
    label_map   = {'NORMAL': 0, 'PNEUMONIA': 1}
    extensions  = ['*.jpeg', '*.jpg', '*.png']

    all_images = []
    all_labels = []

    for cls in classes:
        folder = Path(data_root) / split / cls

        files = []
        for ext in extensions:
            files.extend(list(folder.glob(ext)))

        if max_per_class is not None:
            files = files[:max_per_class]

        loaded = 0
        for fp in files:
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"  Warning: could not read {fp.name} — skipping")
                continue

            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0

            all_images.append(img)
            all_labels.append(label_map[cls])
            loaded += 1

        print(f"  Loaded {loaded} images from {cls}")

    images = np.array(all_images, dtype=np.float32)
    labels = np.array(all_labels, dtype=int)

    rng         = np.random.default_rng(seed=42)
    shuffle_idx = rng.permutation(len(images))
    images      = images[shuffle_idx]
    labels      = labels[shuffle_idx]

    print(f"  Total: {len(images)} images — shape {images.shape}")
    return images, labels