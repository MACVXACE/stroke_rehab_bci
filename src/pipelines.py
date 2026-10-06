"""
pipelines.py
=============
Pipeline builders and the Leave-One-Subject-Out Cross-Validation
(LOSO-CV) driver for subject-independent decoding (Section 4.4-4.5,
Notebook 04, Section 7.1 of the project blueprint).

Two benchmarking pipelines are provided:

  Pipeline A (Euclidean):   CSP(n_components=4, log=True) -> LDA
  Pipeline B (Riemannian):  Covariances(OAS) -> MDM(metric='riemann')

Both are wrapped in `sklearn.pipeline.Pipeline` / `make_pipeline` so
that CSP's spatial filters and the Riemannian covariance estimate are
refit from scratch on every training fold -- this is what keeps the
evaluation leakage-free (Section 8, "Data Leakage in CSP").

Regularization: Riemannian covariance estimation uses Oracle
Approximating Shrinkage (OAS) rather than the empirical estimator,
because acute-stroke recordings can have extensive low-voltage /
near-singular channel regions (Section 8, "Covariance Ill-Conditioning").
"""

from __future__ import annotations

import logging
from typing import Dict, List

import joblib
import numpy as np
from mne.decoding import CSP
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)

DEFAULT_ACTIVATION_THRESHOLD = 0.65  # Section 6, Script 05 neurofeedback gate


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def build_csp_lda_pipeline(n_components: int = 4):
    """Pipeline A: CSP (log-variance) + LDA (Section 4.2-4.4)."""
    return make_pipeline(
        CSP(n_components=n_components, log=True, norm_trace=False),
        LinearDiscriminantAnalysis(),
    )


def build_riemann_pipeline(shrinkage_estimator: str = "oas"):
    """Pipeline B: shrinkage-regularized covariance + Riemannian MDM (Section 4.5)."""
    return make_pipeline(
        Covariances(estimator=shrinkage_estimator),
        MDM(metric="riemann"),
    )


# ---------------------------------------------------------------------------
# LOSO-CV driver
# ---------------------------------------------------------------------------
def evaluate_loso(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    save_best_to: str | None = "models/trained_loso_pipeline.joblib",
) -> Dict[str, List[float]]:
    """
    Execute a strict Leave-One-Subject-Out Cross-Validation loop across
    both benchmark pipelines.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
    y : np.ndarray, shape (n_trials,)
    subjects : np.ndarray, shape (n_trials,)
        Grouping key. Every held-out fold trains on the OTHER subjects'
        trials only and evaluates on this subject's 40 trials
        (Section 6, Notebook 04) -- trials are never pooled before
        the subject split happens.
    save_best_to : str or None
        If given, serialize whichever of the two pipelines has the
        higher mean LOSO accuracy, refit on ALL subjects, to this path
        (Section 6: `models/trained_loso_pipeline.joblib`). Pass None
        to skip saving.

    Returns
    -------
    results : dict
        Per-fold lists: csp_lda_acc, csp_lda_bal_acc, csp_lda_auc,
        riemann_acc, riemann_bal_acc, riemann_auc, plus the pooled
        confusion matrices under '*_confusion'.
    """
    unique_subjects = np.unique(subjects)
    n_subs = len(unique_subjects)
    logger.info("Running LOSO-CV across %d subjects.", n_subs)

    results: Dict[str, List] = {
        "csp_lda_acc": [], "csp_lda_bal_acc": [], "csp_lda_auc": [],
        "riemann_acc": [], "riemann_bal_acc": [], "riemann_auc": [],
        "fold_subject": [],
    }
    y_true_all, y_pred_csp_all, y_pred_riemann_all = [], [], []

    for sub in unique_subjects:
        train_idx = subjects != sub
        test_idx = subjects == sub

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # -- Pipeline A: CSP + LDA --------------------------------------
        csp_lda_pipe = build_csp_lda_pipeline()
        csp_lda_pipe.fit(X_train, y_train)
        y_pred = csp_lda_pipe.predict(X_test)
        y_prob = csp_lda_pipe.predict_proba(X_test)[:, 1]

        results["csp_lda_acc"].append(accuracy_score(y_test, y_pred))
        results["csp_lda_bal_acc"].append(balanced_accuracy_score(y_test, y_pred))
        results["csp_lda_auc"].append(_safe_auc(y_test, y_prob))

        # -- Pipeline B: Riemannian MDM ----------------------------------
        riemann_pipe = build_riemann_pipeline()
        riemann_pipe.fit(X_train, y_train)
        y_pred_rm = riemann_pipe.predict(X_test)
        y_prob_rm = riemann_pipe.predict_proba(X_test)[:, 1]

        results["riemann_acc"].append(accuracy_score(y_test, y_pred_rm))
        results["riemann_bal_acc"].append(balanced_accuracy_score(y_test, y_pred_rm))
        results["riemann_auc"].append(_safe_auc(y_test, y_prob_rm))

        results["fold_subject"].append(sub)
        y_true_all.append(y_test)
        y_pred_csp_all.append(y_pred)
        y_pred_riemann_all.append(y_pred_rm)

        logger.info(
            "Subject %s | CSP+LDA acc=%.3f | Riemann acc=%.3f",
            sub, results["csp_lda_acc"][-1], results["riemann_acc"][-1],
        )

    y_true_all = np.concatenate(y_true_all)
    results["csp_lda_confusion"] = confusion_matrix(
        y_true_all, np.concatenate(y_pred_csp_all)
    )
    results["riemann_confusion"] = confusion_matrix(
        y_true_all, np.concatenate(y_pred_riemann_all)
    )

    if save_best_to is not None:
        _save_best_pipeline(results, X, y, save_best_to)

    return results


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC is undefined for a single-class fold; guard rather than crash."""
    if len(np.unique(y_true)) < 2:
        logger.warning("Fold has a single class present; AUC set to NaN.")
        return float("nan")
    return roc_auc_score(y_true, y_prob)


def _save_best_pipeline(results: Dict, X: np.ndarray, y: np.ndarray, path: str) -> None:
    """Refit the higher-mean-accuracy pipeline on ALL subjects and serialize it.

    NOTE: this final artifact is for the Script 05 real-time simulation demo
    only. It is fit on every subject and therefore must NEVER be used to
    report a decoding accuracy number -- that number comes exclusively from
    the per-fold `results` dict above.
    """
    csp_mean = float(np.nanmean(results["csp_lda_acc"]))
    riemann_mean = float(np.nanmean(results["riemann_acc"]))

    if csp_mean >= riemann_mean:
        logger.info("Saving CSP+LDA pipeline (mean acc %.3f >= %.3f).", csp_mean, riemann_mean)
        best_pipe = build_csp_lda_pipeline()
    else:
        logger.info("Saving Riemannian pipeline (mean acc %.3f > %.3f).", riemann_mean, csp_mean)
        best_pipe = build_riemann_pipeline()

    best_pipe.fit(X, y)
    joblib.dump(best_pipe, path)
    logger.info("Saved deployment pipeline to %s", path)
