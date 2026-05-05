import numpy as np
from numpy.linalg import inv


def fit_mahalanobis(X_train, y_train):
    """
    Fit Mahalanobis distance OOD detector on training data.

    Computes class-conditional means and a shared regularized
    covariance matrix from training embeddings.

    Mahalanobis distance accounts for the covariance structure
    of the feature space — unlike Euclidean distance which treats
    all dimensions equally regardless of their variance or correlation.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, D) — training feature matrix
    y_train : np.ndarray, shape (N,)   — training labels

    Returns
    -------
    detector : dict with keys:
        class_means  — dict mapping label to mean vector
        inv_cov      — inverse of regularized covariance matrix
    """
    classes      = np.unique(y_train)
    class_means  = {}

    for cls in classes:
        class_means[cls] = X_train[y_train == cls].mean(axis=0)

    cov_matrix = np.cov(X_train.T)
    reg_cov    = cov_matrix + 1e-6 * np.eye(X_train.shape[1])
    inv_cov    = inv(reg_cov)

    print(f"Mahalanobis detector fitted.")
    print(f"  Classes         : {list(classes)}")
    print(f"  Feature dims    : {X_train.shape[1]}")
    print(f"  Covariance shape: {cov_matrix.shape}")

    return {'class_means': class_means, 'inv_cov': inv_cov}


def mahalanobis_scores(X, detector):
    """
    Compute Mahalanobis OOD score for each sample.

    For each sample computes distance to the nearest class centroid
    using the Mahalanobis metric. Lower score = in-distribution.
    Higher score = out-of-distribution.

    Parameters
    ----------
    X        : np.ndarray, shape (N, D)
    detector : dict from fit_mahalanobis()

    Returns
    -------
    scores : np.ndarray, shape (N,) — OOD score per sample
    """
    class_means = detector['class_means']
    inv_cov     = detector['inv_cov']
    scores      = []

    for x in X:
        dists = []
        for mu in class_means.values():
            diff = x - mu
            dist = np.sqrt(diff @ inv_cov @ diff)
            dists.append(dist)
        scores.append(min(dists))

    return np.array(scores)


def set_ood_threshold(train_scores, percentile=95):
    """
    Set OOD detection threshold from training score distribution.

    Any test sample with score above this threshold is flagged as OOD.
    Using 95th percentile means we expect 5% of clean training
    samples to be flagged — a conservative but practical threshold.

    Parameters
    ----------
    train_scores : np.ndarray — Mahalanobis scores on training data
    percentile   : float — threshold percentile (default 95)

    Returns
    -------
    threshold : float
    """
    threshold = np.percentile(train_scores, percentile)
    print(f"OOD threshold set at {percentile}th percentile: {threshold:.4f}")
    return threshold


def flag_ood(scores, threshold):
    """
    Flag samples as OOD based on threshold.

    Parameters
    ----------
    scores    : np.ndarray — Mahalanobis scores
    threshold : float

    Returns
    -------
    ood_mask : np.ndarray of bool — True where sample is OOD
    """
    return scores > threshold