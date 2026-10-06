"""
feature_extraction.py
=======================
From-scratch Common Spatial Pattern (CSP) implementation, matching the
generalized-eigenvalue derivation in Section 4.2-4.3 of the project
blueprint. Used in Notebook 03 to cross-verify against `mne.decoding.CSP`
before Notebook 04 commits to the scikit-learn version for LOSO-CV.

Math recap
----------
Trace-normalized class covariances:
    C_k = (1/N_k) * sum_i  X_i X_i^T / trace(X_i X_i^T)

Spatial filters solve the generalized eigenproblem:
    C1 w = lambda (C1 + C2) w

Eigenvectors at the top and bottom of the eigenvalue spectrum give
filters that maximize the variance ratio between the two classes.
Log-variance of the projected signal is the classification feature:
    f_j = log( mean_t( Z_j(t)^2 ) )

ANTI-LEAKAGE CONTRACT
----------------------
`CustomCSP.fit()` must only ever see a single fold's TRAINING trials.
Never call `.fit()` or `.fit_transform()` on the full dataset before
subject-wise splitting (Section 8, "Data Leakage in CSP").
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

# Guards against log(0) = -inf if a channel goes flat/decoupled
# (Section 8, "Zero-Variance Log Trap").
_VARIANCE_FLOOR = 1e-10


class CustomCSP:
    """
    Minimal, dependency-light CSP implementation for pedagogical
    cross-verification against `mne.decoding.CSP`.

    Parameters
    ----------
    n_components : int, default=4
        Total number of spatial filters to keep, split evenly between
        the top and bottom of the eigenvalue spectrum (m = n_components // 2
        filters from each end, per Section 4.2).
    """

    def __init__(self, n_components: int = 4):
        if n_components % 2 != 0:
            raise ValueError("n_components must be even (m filters per class end).")
        self.n_components = n_components
        self.filters_: np.ndarray | None = None   # (n_components, n_channels)
        self.patterns_: np.ndarray | None = None   # (n_channels, n_components)
        self.eigenvalues_: np.ndarray | None = None

    # -- fitting -----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomCSP":
        """
        Fit CSP spatial filters on TRAINING trials only.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)
        y : np.ndarray, shape (n_trials,) -- exactly 2 unique classes
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP is a binary decomposition; got classes {classes}.")

        X1 = X[y == classes[0]]
        X2 = X[y == classes[1]]

        C1 = self._mean_trace_normalized_cov(X1)
        C2 = self._mean_trace_normalized_cov(X2)

        # Generalized eigenvalue problem: C1 w = lambda (C1 + C2) w
        eigenvalues, eigenvectors = eigh(C1, C1 + C2)

        # Descending eigenvalue order so index 0 = most Class-1-like
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        m = self.n_components // 2
        # Top m columns maximize Class 1 variance / minimize Class 2;
        # bottom m columns do the reverse (Section 4.2).
        W = np.concatenate([eigenvectors[:, :m], eigenvectors[:, -m:]], axis=1)

        self.filters_ = W.T               # (n_components, n_channels)
        self.patterns_ = np.linalg.pinv(W)  # scalp-topography patterns
        self.eigenvalues_ = np.concatenate([eigenvalues[:m], eigenvalues[-m:]])
        return self

    # -- transform -----------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project trials through the fitted spatial filters and return
        log-variance features.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        features : np.ndarray, shape (n_trials, n_components)
        """
        if self.filters_ is None:
            raise RuntimeError("CustomCSP.transform() called before fit().")

        features = np.empty((X.shape[0], self.n_components))
        for i, trial in enumerate(X):
            Z = self.filters_ @ trial                 # (n_components, n_times)
            variance = np.var(Z, axis=1)
            variance = np.clip(variance, _VARIANCE_FLOOR, None)  # zero-var guard
            features[i] = np.log(variance)
        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _mean_trace_normalized_cov(X_class: np.ndarray) -> np.ndarray:
        """Section 4.2: mean of trace-normalized per-trial covariances."""
        covs = np.empty((X_class.shape[0], X_class.shape[1], X_class.shape[1]))
        for i, trial in enumerate(X_class):
            c = trial @ trial.T
            trace = np.trace(c)
            trace = trace if trace > _VARIANCE_FLOOR else _VARIANCE_FLOOR
            covs[i] = c / trace
        return covs.mean(axis=0)


def trace_normalized_covariance(trial: np.ndarray) -> np.ndarray:
    """
    Standalone helper (used directly in Notebook 03 before the class is
    introduced) computing a single trial's trace-normalized covariance.

    Parameters
    ----------
    trial : np.ndarray, shape (n_channels, n_times)
    """
    c = trial @ trial.T
    trace = np.trace(c)
    trace = trace if trace > _VARIANCE_FLOOR else _VARIANCE_FLOOR
    return c / trace
