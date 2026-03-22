import numpy as np
import matplotlib.pyplot as plt


def compute_ece(y_true, y_proba, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Groups predictions into confidence bins and measures the weighted
    average gap between predicted confidence and actual accuracy.
    ECE = 0 means perfectly calibrated. Higher = worse.

    Parameters
    ----------
    y_true  : np.ndarray, shape (N,) — true binary labels (0 or 1)
    y_proba : np.ndarray, shape (N,) — predicted probabilities for class 1
    n_bins  : int — number of bins to divide confidence range into

    Returns
    -------
    ece : float — Expected Calibration Error
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (y_proba >= low) & (y_proba < high)

        if mask.sum() == 0:
            continue

        avg_confidence = y_proba[mask].mean()
        avg_accuracy   = y_true[mask].mean()
        bin_weight     = mask.sum() / len(y_true)

        ece += bin_weight * abs(avg_confidence - avg_accuracy)

    return ece


def plot_reliability_diagram(y_true, y_proba, model_name,
                              save_path=None, n_bins=10):
    """
    Plot a reliability diagram for a single model.

    X-axis: mean predicted confidence per bin
    Y-axis: actual fraction of positives per bin
    Diagonal line = perfect calibration
    Points below diagonal = overconfident
    Points above diagonal = underconfident

    Parameters
    ----------
    y_true     : np.ndarray — true binary labels
    y_proba    : np.ndarray — predicted probabilities for class 1
    model_name : str — used in plot title
    save_path  : str or None — if provided, saves the plot here
    n_bins     : int — number of bins
    """
    bins           = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers    = []
    bin_accuracies = []

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (y_proba >= low) & (y_proba < high)

        if mask.sum() == 0:
            continue

        bin_centers.append(y_proba[mask].mean())
        bin_accuracies.append(y_true[mask].mean())

    ece = compute_ece(y_true, y_proba, n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.plot(bin_centers, bin_accuracies,
            marker='o', color='#D85A30', linewidth=2,
            markersize=7, label=f'Model (ECE={ece:.4f})')
    ax.fill_between(bin_centers, bin_centers, bin_accuracies,
                    alpha=0.15, color='#D85A30')

    ax.set_xlabel('Mean predicted confidence')
    ax.set_ylabel('Actual fraction of positives')
    ax.set_title(f'Reliability diagram — {model_name}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")

    plt.show()


def plot_reliability_comparison(y_true, y_proba_before, y_proba_after,
                                 model_name, save_path=None, n_bins=10):
    """
    Plot reliability diagram before and after calibration side by side.

    Parameters
    ----------
    y_true          : np.ndarray — true binary labels
    y_proba_before  : np.ndarray — probabilities before calibration
    y_proba_after   : np.ndarray — probabilities after calibration
    model_name      : str — used in plot title
    save_path       : str or None
    n_bins          : int
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    titles = ['Before calibration', 'After calibration']
    probas = [y_proba_before, y_proba_after]
    colors = ['#D85A30', '#1D9E75']

    for ax, title, proba, color in zip(axes, titles, probas, colors):
        bins           = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers    = []
        bin_accuracies = []

        for i in range(n_bins):
            low, high = bins[i], bins[i + 1]
            mask = (proba >= low) & (proba < high)
            if mask.sum() == 0:
                continue
            bin_centers.append(proba[mask].mean())
            bin_accuracies.append(y_true[mask].mean())

        ece = compute_ece(y_true, proba, n_bins)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5,
                label='Perfect calibration')
        ax.plot(bin_centers, bin_accuracies,
                marker='o', color=color, linewidth=2,
                markersize=7, label=f'Model (ECE={ece:.4f})')
        ax.fill_between(bin_centers, bin_centers, bin_accuracies,
                        alpha=0.15, color=color)

        ax.set_xlabel('Mean predicted confidence')
        ax.set_ylabel('Actual fraction of positives')
        ax.set_title(f'{model_name} — {title}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    plt.suptitle(f'Calibration comparison — {model_name}', fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")

    plt.show()