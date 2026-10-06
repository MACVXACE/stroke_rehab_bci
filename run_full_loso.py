#!/usr/bin/env python3
"""
run_full_loso.py
==================
Standalone, resumable Leave-One-Subject-Out Cross-Validation (LOSO-CV)
runner for all subjects in the Liu2024 acute-stroke MI dataset.

Checkpointing model
--------------------
- Loaded (X, y, subjects) arrays are cached to `--data-cache` (a single
  .npz) after the first successful MOABB pull, so re-running this
  script never re-downloads/re-epochs from scratch.
- Each subject's fold metrics are appended, with an fsync, to
  `results/loso_metrics.csv` IMMEDIATELY after that subject finishes --
  not batched at the end. Per-fold predictions (needed for the pooled
  confusion matrix) are saved alongside as
  `results/predictions/subject_<id>.npz`.
- On (re)start, any subject already present in `loso_metrics.csv` is
  skipped. If the process is killed at subject 30, rerunning the exact
  same command resumes at subject 31 -- nothing is recomputed.
- A subject whose fold raises an exception is logged and skipped
  WITHOUT being checkpointed, so it's automatically retried on the
  next run rather than silently counted as done.

Usage
-----
    python run_full_loso.py                       # full 50-subject run
    python run_full_loso.py --subjects 1,2,3,4,5   # quick partial run
    python run_full_loso.py                        # rerun to resume after an interruption
    python run_full_loso.py --force-restart        # ignore checkpoints, start over
    python run_full_loso.py --mne-data-dir /data/mne_cache   # custom MOABB cache location

Ctrl+C is safe: the subject currently mid-fit is lost, everything
before it is already durably on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_all_subjects  # noqa: E402
from src.pipelines import build_csp_lda_pipeline, build_riemann_pipeline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_CACHE = REPO_ROOT / "data" / "processed" / "liu2024_epochs.npz"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_MODEL_OUT = REPO_ROOT / "models" / "trained_loso_pipeline.joblib"

METRIC_COLUMNS = [
    "subject",
    "csp_lda_acc", "csp_lda_bal_acc", "csp_lda_auc",
    "riemann_acc", "riemann_bal_acc", "riemann_auc",
    "n_test_trials", "fold_seconds", "timestamp",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Checkpointed 50-subject LOSO-CV for the stroke MI-BCI pipeline.")
    p.add_argument("--subjects", type=str, default=None,
                    help="Comma-separated subject IDs to run, e.g. '1,2,3,4,5'. "
                         "Default: every subject in the dataset.")
    p.add_argument("--data-cache", type=Path, default=DEFAULT_DATA_CACHE,
                    help=f"Cache path for loaded epochs (default: {DEFAULT_DATA_CACHE}).")
    p.add_argument("--force-redownload", action="store_true",
                    help="Ignore an existing --data-cache file and reload from MOABB.")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                    help=f"Where checkpoints/summary are written (default: {DEFAULT_RESULTS_DIR}).")
    p.add_argument("--force-restart", action="store_true",
                    help="Ignore existing checkpoints and rerun every subject from scratch.")
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT,
                    help=f"Where to save the final deployment pipeline (default: {DEFAULT_MODEL_OUT}). "
                         "Refit on ALL subjects for the Script 05 demo -- not an accuracy estimate.")
    p.add_argument("--skip-model-save", action="store_true",
                    help="Skip refitting/saving the final deployment pipeline after LOSO completes.")
    p.add_argument("--mne-data-dir", type=str, default=None,
                    help="Override MOABB's dataset cache directory (default ~/mne_data). "
                         "Equivalent to moabb.utils.set_download_dir(...).")
    return p.parse_args()


def setup_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(results_dir / "run.log"),
        ],
    )


# ---------------------------------------------------------------------------
# Data loading + caching
# ---------------------------------------------------------------------------
def load_dataset_arrays(cache_path: Path, force_redownload: bool, subject_ids: list[int] | None):
    if cache_path.exists() and not force_redownload:
        logging.info("Loading cached epochs from %s", cache_path)
        data = np.load(cache_path)
        X, y, subjects = data["X"], data["y"], data["subjects"]
        if subject_ids is not None:
            mask = np.isin(subjects, subject_ids)
            X, y, subjects = X[mask], y[mask], subjects[mask]
        return X, y, subjects

    logging.info("Loading data from MOABB (subjects=%s)...", subject_ids or "all")
    t0 = time.time()
    X, y, subjects = load_all_subjects(subject_ids=subject_ids)
    logging.info("Loaded %d trials in %.1fs.", len(y), time.time() - t0)

    # Only cache a FULL pull -- a partial --subjects run must never
    # silently become the cache for a later full run.
    if subject_ids is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, subjects=subjects)
        logging.info("Cached epochs to %s", cache_path)

    return X, y, subjects


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def load_completed_subjects(results_dir: Path) -> set:
    csv_path = results_dir / "loso_metrics.csv"
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return set()
    if "subject" not in df.columns:
        return set()
    return set(df["subject"].tolist())


def append_checkpoint_row(results_dir: Path, row: dict) -> None:
    """Append one subject's metrics row durably -- flush + fsync before returning."""
    csv_path = results_dir / "loso_metrics.csv"
    write_header = not csv_path.exists()
    import csv
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def save_predictions(results_dir: Path, subject, y_test, pred_csp, prob_csp,
                      pred_riemann, prob_riemann) -> None:
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"subject_{subject}.npz",
        y_test=y_test,
        pred_csp=pred_csp, prob_csp=prob_csp,
        pred_riemann=pred_riemann, prob_riemann=prob_riemann,
    )


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


# ---------------------------------------------------------------------------
# One LOSO fold
# ---------------------------------------------------------------------------
def run_one_fold(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, subject):
    train_idx = subjects != subject
    test_idx = subjects == subject
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    t0 = time.time()

    csp_lda = build_csp_lda_pipeline()
    csp_lda.fit(X_train, y_train)
    pred_csp = csp_lda.predict(X_test)
    prob_csp = csp_lda.predict_proba(X_test)[:, 1]

    riemann = build_riemann_pipeline()
    riemann.fit(X_train, y_train)
    pred_riemann = riemann.predict(X_test)
    prob_riemann = riemann.predict_proba(X_test)[:, 1]

    elapsed = time.time() - t0

    row = {
        "subject": subject,
        "csp_lda_acc": accuracy_score(y_test, pred_csp),
        "csp_lda_bal_acc": balanced_accuracy_score(y_test, pred_csp),
        "csp_lda_auc": _safe_auc(y_test, prob_csp),
        "riemann_acc": accuracy_score(y_test, pred_riemann),
        "riemann_bal_acc": balanced_accuracy_score(y_test, pred_riemann),
        "riemann_auc": _safe_auc(y_test, prob_riemann),
        "n_test_trials": len(y_test),
        "fold_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return row, y_test, pred_csp, prob_csp, pred_riemann, prob_riemann


# ---------------------------------------------------------------------------
# Main LOSO loop
# ---------------------------------------------------------------------------
def run_loso(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
             results_dir: Path, force_restart: bool) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    if force_restart:
        csv_path = results_dir / "loso_metrics.csv"
        if csv_path.exists():
            csv_path.unlink()
        pred_dir = results_dir / "predictions"
        if pred_dir.exists():
            shutil.rmtree(pred_dir)
        logging.info("--force-restart: cleared existing checkpoints in %s", results_dir)

    unique_subjects = sorted(np.unique(subjects).tolist())
    completed = load_completed_subjects(results_dir)
    remaining = [s for s in unique_subjects if s not in completed]

    logging.info("%d/%d subjects already completed; %d remaining.",
                  len(completed), len(unique_subjects), len(remaining))

    if not remaining:
        logging.info("Nothing to run -- use --force-restart to rerun from scratch.")
        return

    pbar = tqdm(remaining, desc="LOSO-CV", unit="subject")
    for subject in pbar:
        pbar.set_postfix_str(f"subject={subject}")
        try:
            row, y_test, pred_csp, prob_csp, pred_riemann, prob_riemann = run_one_fold(
                X, y, subjects, subject
            )
        except KeyboardInterrupt:
            logging.warning(
                "Interrupted at subject %s. Every earlier subject is safely "
                "checkpointed -- rerun this exact command to resume.", subject
            )
            raise
        except Exception as e:  # noqa: BLE001 -- keep the batch alive across bad folds
            logging.error("Subject %s failed and was NOT checkpointed (will retry next run): %s",
                           subject, e, exc_info=True)
            continue

        append_checkpoint_row(results_dir, row)
        save_predictions(results_dir, subject, y_test, pred_csp, prob_csp, pred_riemann, prob_riemann)
        pbar.set_postfix_str(
            f"subject={subject} csp={row['csp_lda_acc']:.2f} riemann={row['riemann_acc']:.2f}"
        )

    logging.info("LOSO-CV pass complete. Rerun this script to retry any subject that errored above.")


# ---------------------------------------------------------------------------
# Summary + confusion matrix + deployment model
# ---------------------------------------------------------------------------
def summarize(results_dir: Path, model_out: Path | None,
              X: np.ndarray | None, y: np.ndarray | None) -> None:
    csv_path = results_dir / "loso_metrics.csv"
    if not csv_path.exists():
        logging.warning("No results at %s -- nothing to summarize.", csv_path)
        return

    df = pd.read_csv(csv_path).sort_values("subject")
    n = len(df)
    logging.info("Summarizing %d completed subject folds.", n)

    def _stats(col: str):
        vals = df[col].dropna().to_numpy()
        mean = float(vals.mean()) if len(vals) else float("nan")
        std = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        sem = float(std / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        return mean, std, sem

    summary = {"n_subjects": n}
    print(f"\n=== LOSO-CV summary ({n} subjects) ===")
    for pipe_name, key, acc_col, bal_col, auc_col in [
        ("CSP + LDA", "csp_lda", "csp_lda_acc", "csp_lda_bal_acc", "csp_lda_auc"),
        ("Riemannian MDM", "riemann", "riemann_acc", "riemann_bal_acc", "riemann_auc"),
    ]:
        acc = _stats(acc_col)
        bal = _stats(bal_col)
        auc = _stats(auc_col)
        print(f"\n{pipe_name}")
        print(f"  Accuracy:          {acc[0]:.3f} +/- {acc[1]:.3f}  (SEM {acc[2]:.3f})")
        print(f"  Balanced accuracy: {bal[0]:.3f} +/- {bal[1]:.3f}  (SEM {bal[2]:.3f})")
        print(f"  ROC-AUC:           {auc[0]:.3f} +/- {auc[1]:.3f}  (SEM {auc[2]:.3f})")
        summary[key] = {
            "accuracy": {"mean": acc[0], "std": acc[1], "sem": acc[2]},
            "balanced_accuracy": {"mean": bal[0], "std": bal[1], "sem": bal[2]},
            "roc_auc": {"mean": auc[0], "std": auc[1], "sem": auc[2]},
        }

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info("Wrote %s", summary_path)

    _plot_confusion_matrices(results_dir, df)

    if model_out is not None and X is not None and y is not None:
        _save_deployment_model(df, X, y, model_out)


def _plot_confusion_matrices(results_dir: Path, df: pd.DataFrame) -> None:
    pred_dir = results_dir / "predictions"
    y_true_all, pred_csp_all, pred_riemann_all = [], [], []
    for subject in df["subject"]:
        npz_path = pred_dir / f"subject_{subject}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        y_true_all.append(data["y_test"])
        pred_csp_all.append(data["pred_csp"])
        pred_riemann_all.append(data["pred_riemann"])

    if not y_true_all:
        logging.warning("No saved predictions found -- skipping confusion matrix plot.")
        return

    y_true_all = np.concatenate(y_true_all)
    pred_csp_all = np.concatenate(pred_csp_all)
    pred_riemann_all = np.concatenate(pred_riemann_all)

    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        have_sns = True
    except ImportError:
        have_sns = False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, preds, title in [
        (axes[0], pred_csp_all, "CSP + LDA"),
        (axes[1], pred_riemann_all, "Riemannian MDM"),
    ]:
        cm = confusion_matrix(y_true_all, preds)
        if have_sns:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["left", "right"], yticklabels=["left", "right"], ax=ax)
        else:
            ax.imshow(cm, cmap="Blues")
            for (i, j), val in np.ndenumerate(cm):
                ax.text(j, i, str(val), ha="center", va="center")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["left", "right"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["left", "right"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)
    fig.suptitle(f"Pooled LOSO-CV confusion matrices — {len(df)}-subject acute-stroke MI decoding")
    plt.tight_layout()
    out_path = results_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logging.info("Saved confusion matrix plot to %s", out_path)


def _save_deployment_model(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, model_out: Path) -> None:
    """Refit the higher-mean-accuracy pipeline on ALL subjects for the Script 05 demo.

    NEVER cite this model's own training performance as a generalization
    estimate -- that number is exclusively the per-fold LOSO results above.
    """
    csp_mean = df["csp_lda_acc"].mean()
    riemann_mean = df["riemann_acc"].mean()

    if csp_mean >= riemann_mean:
        logging.info("Refitting CSP+LDA on all subjects for deployment (%.3f >= %.3f).", csp_mean, riemann_mean)
        pipe = build_csp_lda_pipeline()
    else:
        logging.info("Refitting Riemannian MDM on all subjects for deployment (%.3f > %.3f).", riemann_mean, csp_mean)
        pipe = build_riemann_pipeline()

    pipe.fit(X, y)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)
    logging.info("Saved deployment pipeline to %s (for scripts/05_Simulated_Rehab_Interface.py only).", model_out)


# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    setup_logging(args.results_dir)

    if args.mne_data_dir:
        from moabb.utils import set_download_dir
        set_download_dir(args.mne_data_dir)
        logging.info("MOABB download dir set to %s", args.mne_data_dir)

    subject_ids = None
    if args.subjects:
        subject_ids = [int(s.strip()) for s in args.subjects.split(",")]

    X, y, subjects = load_dataset_arrays(args.data_cache, args.force_redownload, subject_ids)
    logging.info("Working set: %d trials across %d subjects.", len(y), len(np.unique(subjects)))

    try:
        run_loso(X, y, subjects, args.results_dir, args.force_restart)
    except KeyboardInterrupt:
        print("\nStopped by user. Rerun the same command to resume from the last checkpoint.")
        return 130

    summarize(args.results_dir, None if args.skip_model_save else args.model_out, X, y)
    return 0


if __name__ == "__main__":
    sys.exit(main())
