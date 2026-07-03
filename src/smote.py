"""
smote.py
--------
A lightweight, dependency-free implementation of SMOTE (Synthetic Minority
Over-sampling Technique, Chawla et al. 2002) used to correct class
imbalance in the training set. This avoids relying on the
`imbalanced-learn` package so the project runs with just numpy/pandas.

SMOTE creates new synthetic minority-class samples by interpolating
between a minority sample and one of its k nearest minority-class
neighbours, rather than simply duplicating existing rows (random
oversampling) which can lead to overfitting.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_oversample(X, y, minority_label=1, k_neighbors=5, random_state=42):
    """
    Oversample the minority class using SMOTE until classes are balanced.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    minority_label : label considered the minority class
    k_neighbors : number of nearest neighbours to interpolate between
    random_state : int

    Returns
    -------
    X_resampled, y_resampled : balanced dataset (original + synthetic rows)
    """
    rng = np.random.default_rng(random_state)

    X_min = X[y == minority_label]
    X_maj = X[y != minority_label]
    n_min, n_maj = len(X_min), len(X_maj)

    n_to_generate = n_maj - n_min
    if n_to_generate <= 0:
        return X, y  # already balanced

    k = min(k_neighbors, n_min - 1) if n_min > 1 else 1
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, neighbor_idx = nn.kneighbors(X_min)

    synthetic = np.zeros((n_to_generate, X.shape[1]))
    for i in range(n_to_generate):
        sample_idx = rng.integers(0, n_min)
        # pick one of its k nearest neighbours (skip itself at position 0)
        neighbor = neighbor_idx[sample_idx][rng.integers(1, k + 1)]
        gap = rng.random()
        synthetic[i] = X_min[sample_idx] + gap * (X_min[neighbor] - X_min[sample_idx])

    X_resampled = np.vstack([X, synthetic])
    y_resampled = np.concatenate([y, np.full(n_to_generate, minority_label)])

    # shuffle
    perm = rng.permutation(len(X_resampled))
    return X_resampled[perm], y_resampled[perm]
