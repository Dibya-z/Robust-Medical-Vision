import numpy as np
from skimage.feature import hog, local_binary_pattern


def extract_hog(images):
    """
    Extract HOG (Histogram of Oriented Gradients) features from images.

    HOG captures edge directions and gradient structure — in chest X-rays
    this corresponds to lung boundary patterns and rib cage structure.
    Pneumonia consolidation creates new internal edges that HOG detects.

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W), float32, values in [0, 1]
        Grayscale normalized images from load_images()

    Returns
    -------
    features : np.ndarray, shape (N, hog_feature_length)
        HOG feature vector for each image
    """
    orientations     = 9
    pixels_per_cell  = (16, 16)
    cells_per_block  = (2, 2)

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
    print(f"HOG features extracted — shape: {features.shape}")
    return features


def extract_hog_single(img):
    """
    Extract HOG feature and visualization image from a single image.
    Used only for visualization purposes in the notebook.

    Parameters
    ----------
    img : np.ndarray, shape (H, W), float32, values in [0, 1]

    Returns
    -------
    fd       : np.ndarray — HOG feature vector
    hog_img  : np.ndarray — HOG visualization image
    """
    orientations    = 9
    pixels_per_cell = (16, 16)
    cells_per_block = (2, 2)

    fd, hog_img = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        feature_vector=True
    )
    return fd, hog_img


def extract_lbp(images, P=8, R=1):
    """
    Extract LBP (Local Binary Pattern) features from images.

    LBP captures local texture micropatterns — for each pixel it checks
    whether its 8 neighbors are brighter or darker, producing a binary code.
    Normal lung tissue is uniform (consistent LBP codes). Pneumonia
    consolidation creates irregular texture (varied LBP codes).
    The histogram of these codes is the feature vector.

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W), float32, values in [0, 1]
        Grayscale normalized images from load_images()
    P      : int, number of neighbors to sample (default 8)
    R      : float, radius of the circle of neighbors (default 1)

    Returns
    -------
    features : np.ndarray, shape (N, P+2)
        Normalized LBP histogram for each image
    """
    n_bins = P + 2
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

    features = np.array(features, dtype=np.float32)
    print(f"LBP features extracted — shape: {features.shape}")
    return features


def extract_combined(images):
    """
    Extract combined HOG + LBP feature vector from images.

    Combines two complementary feature types into one vector:
    - HOG: structural gradients at the edge scale
    - LBP: local texture at the pixel-neighbourhood scale

    This multi-scale representation is motivated by EDA Finding 2
    (visual texture difference between classes).

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W), float32, values in [0, 1]

    Returns
    -------
    features : np.ndarray, shape (N, hog_length + lbp_length)
        Concatenated HOG + LBP feature vector per image
    """
    print("Extracting HOG features...")
    hog_features = extract_hog(images)

    print("Extracting LBP features...")
    lbp_features = extract_lbp(images)

    combined = np.hstack([hog_features, lbp_features])
    print(f"Combined features shape: {combined.shape}")
    print(f"  HOG contribution : {hog_features.shape[1]} dimensions")
    print(f"  LBP contribution : {lbp_features.shape[1]} dimensions")
    print(f"  Total            : {combined.shape[1]} dimensions")
    return combined
