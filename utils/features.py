import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.feature import graycomatrix, graycoprops


def extract_hog(images):
    """
    Extract HOG (Histogram of Oriented Gradients) features.

    Captures edge directions and gradient structure at lung boundaries.
    Pneumonia consolidation creates internal edges absent in normal tissue.

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W), float32, values in [0, 1]

    Returns
    -------
    features : np.ndarray, shape (N, hog_feature_length)
    """
    orientations    = 9
    pixels_per_cell = (16, 16)
    cells_per_block = (2, 2)

    features = []
    for img in images:
        fd = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            feature_vector=True
        )
        features.append(fd)

    features = np.array(features, dtype=np.float32)
    return features


def extract_hog_single(img):
    """
    Extract HOG feature and visualization from a single image.
    Used only for visualization in notebooks.

    Parameters
    ----------
    img : np.ndarray, shape (H, W), float32

    Returns
    -------
    fd      : np.ndarray — HOG feature vector
    hog_img : np.ndarray — HOG visualization image
    """
    fd, hog_img = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True
    )
    return fd, hog_img


def extract_lbp(images, P=8, R=1):
    """
    Extract LBP (Local Binary Pattern) texture features.

    Compares each pixel to its P neighbors at radius R.
    Normal lung is uniform (consistent codes).
    Pneumonia consolidation is irregular (varied codes).

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W), float32, values in [0, 1]
    P      : int, number of neighbors (default 8)
    R      : float, radius (default 1)

    Returns
    -------
    features : np.ndarray, shape (N, P+2)
    """
    n_bins   = P + 2
    features = []

    for img in images:
        lbp = local_binary_pattern(img, P=P, R=R, method='uniform')
        hist, _ = np.histogram(
            lbp,
            bins=n_bins,
            range=(0, n_bins),
            density=True
        )
        features.append(hist)

    return np.array(features, dtype=np.float32)


def extract_glcm(images):
    """
    Extract GLCM (Gray Level Co-occurrence Matrix) texture features.

    Captures second-order texture statistics at block level.
    Motivated by EDA Finding 2 — visual texture difference between classes.

    Pneumonia consolidation creates:
      - High contrast  (neighboring pixels differ greatly)
      - Low homogeneity (irregular texture)
      - Low energy     (non-repetitive pattern)

    Normal lung creates:
      - Low contrast   (uniform air-filled tissue)
      - High homogeneity
      - High energy

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W), float32, values in [0, 1]

    Returns
    -------
    features : np.ndarray, shape (N, 5)
        [contrast, dissimilarity, homogeneity, energy, correlation]
    """
    properties = ['contrast', 'dissimilarity',
                  'homogeneity', 'energy', 'correlation']
    features   = []

    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)
        glcm      = graycomatrix(
            img_uint8,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )
        props = [graycoprops(glcm, prop).mean() for prop in properties]
        features.append(props)

    return np.array(features, dtype=np.float32)


def extract_histogram(images, n_bins=32):
    """
    Extract pixel intensity histogram features.

    Captures global brightness distribution shape per image.
    Motivated by EDA Finding 3 — PNEUMONIA images are measurably
    brighter due to fluid-filled consolidation appearing white on X-ray.

    A single mean brightness value loses distribution shape.
    The histogram captures the full distribution including bimodality
    (dark air regions + bright consolidation patches together).

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W), float32, values in [0, 1]
    n_bins : int, number of histogram bins (default 32)

    Returns
    -------
    features : np.ndarray, shape (N, n_bins)
    """
    features = []

    for img in images:
        hist, _ = np.histogram(
            img,
            bins=n_bins,
            range=(0.0, 1.0),
            density=True
        )
        features.append(hist)

    return np.array(features, dtype=np.float32)


def extract_region_stats(images, grid_size=4):
    """
    Extract spatial region statistics from a grid of patches.

    Divides each image into a grid_size x grid_size grid.
    Computes mean and std brightness per patch.
    Captures where in the image bright/variable regions are located.

    Motivated by EDA Finding 4 — spatial heatmap showed pathology
    concentrates in lower and middle lung zones. A global mean
    discards this spatial information. Region stats preserve it.

    Parameters
    ----------
    images    : np.ndarray, shape (N, H, W), float32, values in [0, 1]
    grid_size : int, number of grid divisions per axis (default 4)

    Returns
    -------
    features : np.ndarray, shape (N, grid_size * grid_size * 2)
        For each patch: [mean, std] — total 32 values with default grid
    """
    features = []

    for img in images:
        h, w   = img.shape
        ph, pw = h // grid_size, w // grid_size
        patch_features = []

        for i in range(grid_size):
            for j in range(grid_size):
                patch = img[
                    i * ph : (i + 1) * ph,
                    j * pw : (j + 1) * pw
                ]
                patch_features.append(patch.mean())
                patch_features.append(patch.std())

        features.append(patch_features)

    return np.array(features, dtype=np.float32)


def extract_combined_baseline(images):
    """
    Baseline feature vector — HOG + LBP only.
    Used in Notebook 01.

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W), float32

    Returns
    -------
    features : np.ndarray, shape (N, hog_length + lbp_length)
    """
    print("  Extracting HOG...")
    hog_feats = extract_hog(images)
    print(f"  HOG shape: {hog_feats.shape}")

    print("  Extracting LBP...")
    lbp_feats = extract_lbp(images)
    print(f"  LBP shape: {lbp_feats.shape}")

    combined = np.hstack([hog_feats, lbp_feats])
    print(f"  Baseline combined shape: {combined.shape}")
    return combined


def extract_combined_advanced(images):
    """
    Advanced feature vector — HOG + LBP + GLCM + Histogram + Region stats.
    Used in Notebook 02.

    Every feature traces to a specific EDA finding:
      HOG          — EDA Finding 2 (visual texture difference)
      LBP          — EDA Finding 2 (visual texture difference)
      GLCM         — EDA Finding 2 (block-level texture statistics)
      Histogram    — EDA Finding 3 (brightness distribution shift)
      Region stats — EDA Finding 4 (spatial localization of pathology)

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W), float32

    Returns
    -------
    features : np.ndarray, shape (N, total_feature_length)
    """
    print("  Extracting HOG...")
    hog_feats  = extract_hog(images)
    print(f"  HOG shape        : {hog_feats.shape}")

    print("  Extracting LBP...")
    lbp_feats  = extract_lbp(images)
    print(f"  LBP shape        : {lbp_feats.shape}")

    print("  Extracting GLCM...")
    glcm_feats = extract_glcm(images)
    print(f"  GLCM shape       : {glcm_feats.shape}")

    print("  Extracting intensity histogram...")
    hist_feats = extract_histogram(images)
    print(f"  Histogram shape  : {hist_feats.shape}")

    print("  Extracting region statistics...")
    reg_feats  = extract_region_stats(images)
    print(f"  Region stats shape: {reg_feats.shape}")

    combined = np.hstack([
        hog_feats,
        lbp_feats,
        glcm_feats,
        hist_feats,
        reg_feats
    ])

    print()
    print(f"  Advanced combined shape : {combined.shape}")
    print(f"  Breakdown:")
    print(f"    HOG          : {hog_feats.shape[1]} dims")
    print(f"    LBP          : {lbp_feats.shape[1]} dims")
    print(f"    GLCM         : {glcm_feats.shape[1]} dims")
    print(f"    Histogram    : {hist_feats.shape[1]} dims")
    print(f"    Region stats : {reg_feats.shape[1]} dims")
    print(f"    Total        : {combined.shape[1]} dims")

    return combined