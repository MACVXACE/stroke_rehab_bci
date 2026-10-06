"""
domain_adaptation.py
======================
Cross-subject transfer learning on top of the existing LOSO-CV split,
implemented via `pyriemann.transfer` / `pyriemann.tangentspace`
(pyriemann>=0.4). Four regimes are exposed, all still evaluated per-fold
on a held-out subject exactly like `src/pipelines.py::evaluate_loso`:

  - `evaluate_loso_rpa_unsupervised` -- manifold re-centering + scaling
    only (`TLCenter` + `TLScale`). Zero calibration LABELS needed from
    the held-out patient. 50-subject result on the real cohort:
    52.1% +/- 6.3% vs a 50.7% baseline -- a modest, real gain.
  - `evaluate_loso_rpa_rotation` -- adds full-manifold rotation
    (`TLRotate`, Grassmannian optimization over the SPD manifold) via a
    small labeled calibration set. **Measured: 38.7% +/- 5.6% on the
    real 50-subject cohort -- severe negative transfer**, from
    optimizing a 406-parameter (n(n-1)/2, n=29) rotation matrix against
    only ~4-8 calibration trials/class. Kept as a documented negative
    result, NOT a recommended path -- see module-level NOTE below on
    why it's slow and what actually explains an apparent "hang".
  - `evaluate_loso_tsa_rotation` -- Tangent Space Alignment (Bleuzé,
    Mattout & Congedo, Front. Hum. Neurosci. 2022): same rotation idea
    as above but computed via PCA-anchor matching in tangent space
    instead of Grassmannian optimization on the manifold -- much faster
    (~0.7s/fold vs ~15-20s/fold in testing) since the anchor PCA's
    dimensionality is capped by the available calibration trials. Not
    yet run on the real cohort as of this module version.
  - `evaluate_loso_tangent_space` -- the simplified, rotation-free
    pipeline: `Covariances(OAS)` -> `TLCenter` -> `TangentSpace` -> a
    standard classifier (LDA) on the resulting Euclidean vectors. No
    `TLRotate` anywhere. This is the standard, well-established
    "tangent space + shrinkage classifier" pattern in the pyriemann
    ecosystem -- robust, fast, and the recommended default here.
    Terminology note: this is NOT "TSA" in the Bleuzé et al. sense
    (there's no anchor-rotation step), just re-centering + tangent
    projection + a plain classifier -- named accurately below so it
    doesn't get confused with `evaluate_loso_tsa_rotation` later.

Covariance estimation throughout uses Oracle Approximating Shrinkage
(OAS, `Covariances(estimator="oas")`) for well-conditioned matrices,
consistent with `src/pipelines.py::build_riemann_pipeline`.

NOTE on the reported "SVD deadlock" -- worth understanding before
picking a variant, since it shapes which failure mode you're actually
defending against:
  `TLRotate`'s tangent-space code path (`_get_rotation_tangentspace`)
  computes `C = X_source.T @ X_target`, a (435, 435) matrix for 29
  channels (n(n+1)/2), then `np.linalg.svd(C)`. Tested directly here:
  SVD of a singular, rank-3, or even all-zero 435x435 matrix completes
  in 30-70ms -- singularity/rank-deficiency does not hang SVD; that's
  one of its defining stability properties (unlike matrix inversion).
  What CAN misbehave is NaN/Inf-contaminated input (tested here: raises
  `LinAlgError` near-instantly on this platform's BLAS backend; some
  BLAS backends, notably macOS's Accelerate framework in certain
  numpy/OS version combinations, have documented hangs specifically on
  NaN input rather than a clean error). Two more mundane explanations
  are at least as likely as a BLAS bug: (1) none of the functions below
  had a progress indicator until this version, so a legitimate
  multi-minute 50-subject x 49-source-domain run (each domain: one
  435x435 SVD, milliseconds, but 49 of them per fold, sequentially, not
  parallelized -- see `evaluate_loso_tsa_rotation`'s docstring) could
  look indistinguishable from a hang; (2) if your real covariance
  estimates are near-singular (e.g. from very short epochs or reference
  scheme choices) further upstream numerical steps could in principle
  produce a NaN that only surfaces at the SVD call. `_run_fold` below
  now wraps `pipe.fit()` to catch and clearly report `LinAlgError`
  instead of leaving you looking at a bare stack trace, and every
  `evaluate_loso_*` function now shows per-subject progress via `tqdm`.
  `check_covariance_conditioning()` at the bottom of this file is a
  standalone diagnostic to run BEFORE a long fit if you want to check
  your real data for near-singular trials up front.

ANTI-LEAKAGE CONTRACT (extends Section 8 of the project blueprint)
---------------------------------------------------------------------
- Source-domain trials are always the OTHER subjects' full labeled trial
  sets, exactly as in `evaluate_loso` -- never pooled before the subject
  split.
- Unsupervised variant: `TLCenter`/`TLScale` only ever look at DOMAIN
  membership (which subject a trial came from), never its class label,
  so folding the held-out subject's evaluation trials in as an
  unlabeled warm-start batch does not leak label information across the
  train/test boundary. If you want a stricter regime that doesn't even
  look at the evaluation trials' covariances beforehand, carve off a
  separate unlabeled warm-start block instead (see `warmstart_fraction`
  in `evaluate_loso_rpa_unsupervised`).
- Rotation / TSA / tangent-space variants: calibration trials used for
  fitting are always excluded from that fold's evaluation set -- never
  scored twice.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.transfer import TLCenter, TLClassifier, TLRotate, TLScale, encode_domains
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


def _positive_class_column(pipe, positive_label, dtype) -> int:
    """TLClassifier decodes its wrapped estimator's classes_ to STRINGS
    (see encode_domains), so look up the column for `positive_label`
    (e.g. 1 for right_hand) by value rather than assuming column order."""
    classes_ = pipe.steps[-1][1].estimator.classes_
    classes_native = np.asarray(classes_).astype(dtype)
    match = np.where(classes_native == positive_label)[0]
    if len(match) != 1:
        raise ValueError(f"Could not uniquely locate positive class {positive_label} "
                          f"among classifier classes {classes_}.")
    return int(match[0])


def _fit_with_diagnostics(pipe, X_enc, y_enc, held_out):
    """Fit with a clear, actionable error instead of an opaque stack trace
    (or an apparent hang) if the SPD/tangent-space math hits NaN/Inf --
    see the module-level NOTE above."""
    try:
        pipe.fit(X_enc, y_enc)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"[subject {held_out}] Linear algebra failure during fit ({e}). "
            f"This is consistent with NaN/Inf in an intermediate covariance or "
            f"tangent-space projection -- run check_covariance_conditioning() on "
            f"this subject's trials to check for near-singular covariances before "
            f"retrying, rather than assuming it's a plain hang."
        ) from e
    return pipe


def check_covariance_conditioning(X: np.ndarray, subjects: Optional[np.ndarray] = None,
                                   covariance_estimator: str = "oas") -> "dict":
    """
    Standalone pre-flight diagnostic: estimate the OAS covariance for
    every trial and report condition numbers, so you can check real data
    for near-singular trials BEFORE committing to a long fit, rather than
    inferring it after an ambiguous stall.

    Returns a dict with 'cond_numbers' (array, one per trial),
    'n_nonfinite' (count of trials with any NaN/Inf in their covariance),
    and, if `subjects` is given, 'worst_subjects' (the 5 subjects with
    the highest median condition number).
    """
    covs = Covariances(estimator=covariance_estimator).transform(X)
    cond_numbers = np.array([np.linalg.cond(c) for c in covs])
    n_nonfinite = int(np.sum(~np.isfinite(covs).all(axis=(1, 2))))

    report = {
        "cond_numbers": cond_numbers,
        "median_cond": float(np.median(cond_numbers)),
        "max_cond": float(np.max(cond_numbers)),
        "n_nonfinite": n_nonfinite,
    }
    if subjects is not None:
        uniq = np.unique(subjects)
        med_by_subject = {s: float(np.median(cond_numbers[subjects == s])) for s in uniq}
        worst = sorted(med_by_subject.items(), key=lambda kv: kv[1], reverse=True)[:5]
        report["worst_subjects"] = worst

    if n_nonfinite:
        logger.warning("%d/%d trials have a non-finite OAS covariance.", n_nonfinite, len(X))
    if cond_numbers.max() > 1e12:
        logger.warning("Max covariance condition number %.2e is extremely high -- "
                        "near-singular trials present.", cond_numbers.max())
    return report


def _calibration_split(X_held, y_held, n_calib_per_class, rng):
    calib_mask = np.zeros(len(y_held), dtype=bool)
    for cls in np.unique(y_held):
        cls_idx = np.where(y_held == cls)[0]
        n = min(n_calib_per_class, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n, replace=False)
        calib_mask[chosen] = True
    return calib_mask


# ---------------------------------------------------------------------------
# Unsupervised RPA (re-center + scale only)
# ---------------------------------------------------------------------------
def _build_rpa_pipeline(target_domain: str, use_rotation: bool,
                         covariance_estimator: str = "oas",
                         rotate_maxiter: int = 100, rotate_tol_step: float = 1e-6):
    """Covariances -> [TLCenter -> (TLRotate) -> TLScale] -> TLClassifier(MDM)."""
    steps = [Covariances(estimator=covariance_estimator),
              TLCenter(target_domain=target_domain)]
    if use_rotation:
        steps.append(TLRotate(target_domain=target_domain, metric="riemann",
                                maxiter=rotate_maxiter, tol_step=rotate_tol_step))
    steps.append(TLScale(target_domain=target_domain))
    steps.append(TLClassifier(target_domain=target_domain, estimator=MDM()))
    return make_pipeline(*steps)


def evaluate_loso_rpa_unsupervised(
    X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
    warmstart_fraction: float = 1.0, seed: int = 42, show_progress: bool = True,
) -> Dict[str, List]:
    """RPA transfer with re-centering + scaling only -- no target-subject
    LABELS used anywhere. See module docstring for the measured result."""
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subjects)
    results: Dict[str, List] = {"subject": [], "acc": [], "bal_acc": [], "auc": [],
                                  "n_warmstart": [], "n_eval": []}

    for held_out in tqdm(unique_subjects, desc="RPA (unsupervised)", disable=not show_progress):
        target_domain = str(held_out)
        train_idx = subjects != held_out
        held_idx = np.where(subjects == held_out)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        dom_train = subjects[train_idx].astype(str)

        if warmstart_fraction >= 1.0:
            warm_idx, eval_idx = held_idx, held_idx
        else:
            n_warm = max(2, int(round(warmstart_fraction * len(held_idx))))
            perm = rng.permutation(held_idx)
            warm_idx, eval_idx = perm[:n_warm], perm[n_warm:]

        X_warm = X[warm_idx]
        y_warm_placeholder = np.zeros(len(warm_idx), dtype=y.dtype)
        dom_warm = subjects[warm_idx].astype(str)

        X_fit = np.concatenate([X_train, X_warm], axis=0)
        y_fit = np.concatenate([y_train, y_warm_placeholder], axis=0)
        dom_fit = np.concatenate([dom_train, dom_warm], axis=0)
        X_enc, y_enc = encode_domains(X_fit, y_fit, dom_fit)

        pipe = _build_rpa_pipeline(target_domain, use_rotation=False)
        _fit_with_diagnostics(pipe, X_enc, y_enc, held_out)

        X_eval, y_eval = X[eval_idx], y[eval_idx]
        preds = pipe.predict(X_eval).astype(y.dtype)
        pos_col = _positive_class_column(pipe, positive_label=y.max(), dtype=y.dtype)
        probs = pipe.predict_proba(X_eval)[:, pos_col]

        results["subject"].append(held_out)
        results["acc"].append(accuracy_score(y_eval, preds))
        results["bal_acc"].append(balanced_accuracy_score(y_eval, preds))
        results["auc"].append(_safe_auc(y_eval, probs))
        results["n_warmstart"].append(len(warm_idx))
        results["n_eval"].append(len(eval_idx))

    return results


def evaluate_loso_rpa_rotation(
    X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
    n_calib_per_class: int = 4, seed: int = 42,
    rotate_maxiter: int = 100, rotate_tol_step: float = 1e-6,
    show_progress: bool = True,
) -> Dict[str, List]:
    """Full-manifold rotation variant (formerly `evaluate_loso_rpa_supervised`).
    **Measured at 38.7% +/- 5.6% vs a 50.7% baseline on the real 50-subject
    cohort -- severe negative transfer.** Kept for reproducibility, not
    recommended -- see `evaluate_loso_tangent_space` or
    `evaluate_loso_tsa_rotation` instead."""
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subjects)
    results: Dict[str, List] = {"subject": [], "acc": [], "bal_acc": [], "auc": [],
                                  "n_calib": []}

    for held_out in tqdm(unique_subjects, desc="RPA (full rotation)", disable=not show_progress):
        target_domain = str(held_out)
        train_idx = subjects != held_out
        held_idx = np.where(subjects == held_out)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        dom_train = subjects[train_idx].astype(str)
        X_held, y_held = X[held_idx], y[held_idx]

        calib_mask = _calibration_split(X_held, y_held, n_calib_per_class, rng)
        X_calib, y_calib = X_held[calib_mask], y_held[calib_mask]
        X_eval, y_eval = X_held[~calib_mask], y_held[~calib_mask]
        dom_calib = np.full(len(y_calib), target_domain)

        X_fit = np.concatenate([X_train, X_calib], axis=0)
        y_fit = np.concatenate([y_train, y_calib], axis=0)
        dom_fit = np.concatenate([dom_train, dom_calib], axis=0)
        X_enc, y_enc = encode_domains(X_fit, y_fit, dom_fit)

        pipe = _build_rpa_pipeline(target_domain, use_rotation=True,
                                    rotate_maxiter=rotate_maxiter, rotate_tol_step=rotate_tol_step)
        _fit_with_diagnostics(pipe, X_enc, y_enc, held_out)

        preds = pipe.predict(X_eval).astype(y.dtype)
        pos_col = _positive_class_column(pipe, positive_label=y.max(), dtype=y.dtype)
        probs = pipe.predict_proba(X_eval)[:, pos_col]

        results["subject"].append(held_out)
        results["acc"].append(accuracy_score(y_eval, preds))
        results["bal_acc"].append(balanced_accuracy_score(y_eval, preds))
        results["auc"].append(_safe_auc(y_eval, probs))
        results["n_calib"].append(int(calib_mask.sum()))

    return results


# kept for backwards compatibility with earlier notebooks/scripts
evaluate_loso_rpa_supervised = evaluate_loso_rpa_rotation


# ---------------------------------------------------------------------------
# Tangent Space Alignment (rotation-based, via PCA anchors in tangent space)
# ---------------------------------------------------------------------------
def _build_tsa_rotation_pipeline(target_domain: str, n_components: int = 2,
                                  expl_var: float = 0.999, n_clusters: int = 2,
                                  covariance_estimator: str = "oas"):
    """Covariances(OAS) -> TLCenter -> TangentSpace -> TLRotate(tangent
    mode, PCA-anchor matching) -> TLScale -> TLClassifier(LDA).

    n_clusters (default 2, library default 3): `TLRotate._get_anchors`
    requires n_vectors_c >= n_clusters for the TARGET domain's per-class
    calibration trials, or it silently falls back to a class-mean-only
    anchor -- keep n_calib_per_class >= n_clusters.
    """
    return make_pipeline(
        Covariances(estimator=covariance_estimator),
        TLCenter(target_domain=target_domain),
        TangentSpace(metric="riemann"),
        TLRotate(target_domain=target_domain, n_components=n_components, expl_var=expl_var,
                  n_clusters=n_clusters),
        TLScale(target_domain=target_domain),
        TLClassifier(target_domain=target_domain, estimator=LinearDiscriminantAnalysis()),
    )


def evaluate_loso_tsa_rotation(
    X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
    n_calib_per_class: int = 4, n_components: int = 2,
    expl_var: float = 0.999, n_clusters: int = 2, seed: int = 42,
    show_progress: bool = True,
) -> Dict[str, List]:
    """Tangent Space Alignment (Bleuzé et al. 2022): re-center -> tangent-
    project -> PCA-anchor rotation matching -> scale -> LDA. Formerly
    named `evaluate_loso_tsa` -- renamed to distinguish it from
    `evaluate_loso_tangent_space` (no rotation at all), added when
    debugging a reported stall in this function. NOT yet run on the real
    cohort. If you saw a stall here specifically: this loop runs one
    (435, 435)-SVD-based anchor match per SOURCE domain (49 of them for
    the real 50-subject cohort), sequentially, per fold -- see the
    module-level NOTE for why that's more likely to look slow than to
    actually be stuck.
    """
    if n_calib_per_class < n_clusters:
        logger.warning(
            "n_calib_per_class=%d < n_clusters=%d -- TLRotate will fall back to "
            "class-mean-only anchors for the target domain. Raise n_calib_per_class "
            "or lower n_clusters.", n_calib_per_class, n_clusters,
        )

    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subjects)
    results: Dict[str, List] = {"subject": [], "acc": [], "bal_acc": [], "auc": [],
                                  "n_calib": []}

    for held_out in tqdm(unique_subjects, desc="TSA (rotation)", disable=not show_progress):
        target_domain = str(held_out)
        train_idx = subjects != held_out
        held_idx = np.where(subjects == held_out)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        dom_train = subjects[train_idx].astype(str)
        X_held, y_held = X[held_idx], y[held_idx]

        calib_mask = _calibration_split(X_held, y_held, n_calib_per_class, rng)
        X_calib, y_calib = X_held[calib_mask], y_held[calib_mask]
        X_eval, y_eval = X_held[~calib_mask], y_held[~calib_mask]
        dom_calib = np.full(len(y_calib), target_domain)

        X_fit = np.concatenate([X_train, X_calib], axis=0)
        y_fit = np.concatenate([y_train, y_calib], axis=0)
        dom_fit = np.concatenate([dom_train, dom_calib], axis=0)
        X_enc, y_enc = encode_domains(X_fit, y_fit, dom_fit)

        pipe = _build_tsa_rotation_pipeline(target_domain, n_components=n_components,
                                             expl_var=expl_var, n_clusters=n_clusters)
        _fit_with_diagnostics(pipe, X_enc, y_enc, held_out)

        preds = pipe.predict(X_eval).astype(y.dtype)
        pos_col = _positive_class_column(pipe, positive_label=y.max(), dtype=y.dtype)
        probs = pipe.predict_proba(X_eval)[:, pos_col]

        results["subject"].append(held_out)
        results["acc"].append(accuracy_score(y_eval, preds))
        results["bal_acc"].append(balanced_accuracy_score(y_eval, preds))
        results["auc"].append(_safe_auc(y_eval, probs))
        results["n_calib"].append(int(calib_mask.sum()))

    return results


# kept for backwards compatibility with the previous module version
evaluate_loso_tsa = evaluate_loso_tsa_rotation


# ---------------------------------------------------------------------------
# Tangent space, no rotation at all: Covariances(OAS) -> TLCenter ->
# TangentSpace -> standard classifier. What was requested to replace
# TLRotate entirely.
# ---------------------------------------------------------------------------
def _build_tangent_space_pipeline(target_domain: str, covariance_estimator: str = "oas",
                                   classifier=None):
    """Covariances(OAS) -> TLCenter -> TangentSpace -> TLClassifier(LDA).

    No TLRotate anywhere -- re-centering only (unsupervised, per-domain),
    then a plain Euclidean classifier on the tangent vectors. This is the
    standard, low-risk pyriemann "TS+LDA" pattern: no Grassmannian
    optimization, no PCA-anchor matching, nothing that scales with
    n_channels^2 or needs iterative convergence -- just one geometric
    mean per domain (TLCenter) and one tangent-space log-map per trial,
    both closed-form. `classifier` defaults to `LinearDiscriminantAnalysis()`;
    pass e.g. `LogisticRegression(max_iter=1000)` for an L2-regularized
    alternative if LDA's covariance estimate is itself unstable on your
    trial counts.
    """
    if classifier is None:
        classifier = LinearDiscriminantAnalysis()
    return make_pipeline(
        Covariances(estimator=covariance_estimator),
        TLCenter(target_domain=target_domain),
        TangentSpace(metric="riemann"),
        TLClassifier(target_domain=target_domain, estimator=classifier),
    )


def evaluate_loso_tangent_space(
    X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
    n_calib_per_class: int = 4, seed: int = 42, classifier=None,
    show_progress: bool = True,
) -> Dict[str, List]:
    """
    Re-center (manifold, unsupervised) -> tangent-space projection ->
    plain classifier (default LDA) on the resulting Euclidean vectors.
    No TLRotate at all.

    Uses the same small labeled-calibration-set structure as the other
    supervised variants for a fair comparison, but note the calibration
    trials here are used only to give the target domain SOME presence in
    the pooled training set for TLCenter's per-domain mean -- re-centering
    itself never looks at their labels, so if you want a true zero-label
    variant, pass `n_calib_per_class=0` and the held-out subject's own
    (unlabeled) evaluation trials become their own warm-start batch,
    same as `evaluate_loso_rpa_unsupervised`.
    """
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subjects)
    results: Dict[str, List] = {"subject": [], "acc": [], "bal_acc": [], "auc": [],
                                  "n_calib": []}

    for held_out in tqdm(unique_subjects, desc="Tangent space (no rotation)", disable=not show_progress):
        target_domain = str(held_out)
        train_idx = subjects != held_out
        held_idx = np.where(subjects == held_out)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        dom_train = subjects[train_idx].astype(str)
        X_held, y_held = X[held_idx], y[held_idx]

        if n_calib_per_class > 0:
            calib_mask = _calibration_split(X_held, y_held, n_calib_per_class, rng)
        else:
            calib_mask = np.zeros(len(y_held), dtype=bool)

        X_calib, y_calib = X_held[calib_mask], y_held[calib_mask]
        X_eval, y_eval = X_held[~calib_mask], y_held[~calib_mask]

        if len(X_calib) > 0:
            dom_calib = np.full(len(y_calib), target_domain)
            X_fit = np.concatenate([X_train, X_calib], axis=0)
            y_fit = np.concatenate([y_train, y_calib], axis=0)
            dom_fit = np.concatenate([dom_train, dom_calib], axis=0)
        else:
            # zero-calibration fallback: fold in the held-out subject's own
            # (unlabeled-as-far-as-TLCenter-is-concerned) evaluation trials
            # as their domain's warm-start batch, same logic as
            # evaluate_loso_rpa_unsupervised.
            y_eval_placeholder = np.zeros(len(X_eval), dtype=y.dtype)
            dom_eval = subjects[held_idx][~calib_mask].astype(str)
            X_fit = np.concatenate([X_train, X_eval], axis=0)
            y_fit = np.concatenate([y_train, y_eval_placeholder], axis=0)
            dom_fit = np.concatenate([dom_train, dom_eval], axis=0)

        X_enc, y_enc = encode_domains(X_fit, y_fit, dom_fit)

        pipe = _build_tangent_space_pipeline(target_domain, classifier=classifier)
        _fit_with_diagnostics(pipe, X_enc, y_enc, held_out)

        preds = pipe.predict(X_eval).astype(y.dtype)
        pos_col = _positive_class_column(pipe, positive_label=y.max(), dtype=y.dtype)
        probs = pipe.predict_proba(X_eval)[:, pos_col]

        results["subject"].append(held_out)
        results["acc"].append(accuracy_score(y_eval, preds))
        results["bal_acc"].append(balanced_accuracy_score(y_eval, preds))
        results["auc"].append(_safe_auc(y_eval, probs))
        results["n_calib"].append(int(calib_mask.sum()))

    return results
