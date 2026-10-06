#!/usr/bin/env python3
"""
07_evaluate_dcd_rls.py
========================
Evaluates the DCD-RLS adaptive temporal filtering pipeline
(src/adaptive_filtering.py) -- causal EOG artifact cancellation, then
EITHER plain per-channel log-bandpower features OR CSP -- as an
alternative to the spatial covariance/Riemannian domain-adaptation
approaches in src/domain_adaptation.py.

Runs BOTH classifier variants by default so they're directly comparable
in one pass:
  - dcd_rls: cleaning -> log-bandpower (no spatial filter) -> LDA
  - dcd_rls_csp: cleaning -> CSP (fit per-fold, train-only) -> LDA
The CSP variant was added after the log-bandpower-only version measured
48.8% on the real cohort (near chance) -- CSP is a SUPERVISED spatial
filter, so unlike log-bandpower it must be (and is) fit fresh inside
every LOSO fold, never on pooled data. Validated on synthetic data
before trusting it: with a genuinely spatially-distributed class signal
(trace-preserving power swap along two random orthonormal directions --
the classic case CSP is built for) log-bandpower reached 88.7% while
CSP+LDA reached 98.8% on the same data -- confirms the CSP integration
captures spatial structure per-channel features miss, rather than just
running without crashing.

Unlike scripts/06_evaluate_rpa.py, this needs the EOG channels (dropped
by run_full_loso.py's cache), so it loads/caches its OWN
(X_eeg, X_eog, y, subjects) via
src/data_loader.py::load_all_subjects_with_eog -- separate cache file,
does not touch or require data/processed/liu2024_epochs.npz.

Usage
-----
    python scripts/07_evaluate_dcd_rls.py                      # full cohort, both variants
    python scripts/07_evaluate_dcd_rls.py --subjects 1,2,3,4,5  # quick partial run
    python scripts/07_evaluate_dcd_rls.py --skip-logvar         # CSP variant only
    python scripts/07_evaluate_dcd_rls.py --skip-csp            # log-bandpower variant only
    python scripts/07_evaluate_dcd_rls.py --csp-n-components 6

Writes
------
    data/processed/liu2024_epochs_with_eog.npz   (cache, reused on rerun)
    results/dcd_rls_metrics.csv
    results/dcd_rls_csp_metrics.csv
    results/dcd_rls_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mne  # noqa: E402
mne.set_log_level("ERROR")  # CSP's per-fold covariance-rank logging is extremely verbose otherwise

from src.adaptive_filtering import _HAVE_NUMBA, evaluate_loso_dcd_rls, evaluate_loso_dcd_rls_csp  # noqa: E402
from src.data_loader import load_all_subjects_with_eog  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_CACHE = REPO_ROOT / "data" / "processed" / "liu2024_epochs_with_eog.npz"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DCD-RLS adaptive filtering (+/- CSP) via LOSO-CV.")
    p.add_argument("--data-cache", type=Path, default=DEFAULT_DATA_CACHE)
    p.add_argument("--force-redownload", action="store_true")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--subjects", type=str, default=None,
                    help="Comma-separated subject IDs (default: all).")
    p.add_argument("--taps-per-ref", type=int, default=3,
                    help="FIR taps per reference (EOG) channel. Filter length = n_ref * this.")
    p.add_argument("--forgetting-factor", type=float, default=0.999)
    p.add_argument("--Nu", type=int, default=8, help="Max DCD coordinate updates per sample.")
    p.add_argument("--Mb", type=int, default=16, help="Max power-of-two step halvings per sample.")
    p.add_argument("--csp-n-components", type=int, default=4,
                    help="Matches the baseline CSP+LDA pipeline's default for comparability.")
    p.add_argument("--skip-logvar", action="store_true", help="Skip the log-bandpower (no CSP) variant.")
    p.add_argument("--skip-csp", action="store_true", help="Skip the CSP variant.")
    p.add_argument("--no-numba", action="store_true",
                    help="Force the pure-Python cleaning path. WARNING: measured at ~60 min for the "
                         "full 50-subject cohort vs ~50s with numba (installed by default here). "
                         "Only useful for debugging the numba path itself.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not _HAVE_NUMBA and not args.no_numba:
        print("[i] numba not installed -- falling back to the pure-Python cleaning path, which will "
              "be roughly 75x slower (measured: ~2s/trial vs ~0.024s/trial). "
              "`pip install numba` first if this run feels impractically slow.")

    subject_ids = None
    if args.subjects:
        subject_ids = [int(s.strip()) for s in args.subjects.split(",")]

    if args.data_cache.exists() and not args.force_redownload:
        print(f"Loading cached epochs (with EOG) from {args.data_cache}")
        data = np.load(args.data_cache)
        X_eeg, X_eog, y, subjects = data["X_eeg"], data["X_eog"], data["y"], data["subjects"]
        if subject_ids is not None:
            mask = np.isin(subjects, subject_ids)
            X_eeg, X_eog, y, subjects = X_eeg[mask], X_eog[mask], y[mask], subjects[mask]
    else:
        print(f"Loading data from MOABB (subjects={subject_ids or 'all'}), including EOG channels...")
        t0 = time.time()
        X_eeg, X_eog, y, subjects = load_all_subjects_with_eog(subject_ids=subject_ids)
        print(f"Loaded {len(y)} trials in {time.time()-t0:.1f}s.")
        if subject_ids is None:
            args.data_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.data_cache, X_eeg=X_eeg, X_eog=X_eog, y=y, subjects=subjects)
            print(f"Cached to {args.data_cache}")

    n_subjects = len(np.unique(subjects))
    print(f"Working set: {len(y)} trials across {n_subjects} subjects. "
          f"EEG shape {X_eeg.shape}, EOG (reference) shape {X_eog.shape}.")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary = {"n_subjects": n_subjects, "taps_per_ref": args.taps_per_ref,
               "forgetting_factor": args.forgetting_factor, "Nu": args.Nu, "Mb": args.Mb}

    baseline_path = args.results_dir / "loso_metrics.csv"
    if baseline_path.exists():
        base_df = pd.read_csv(baseline_path)
        summary["baseline_riemann_acc_mean"] = float(base_df["riemann_acc"].mean())
        summary["baseline_csp_lda_acc_mean"] = float(base_df["csp_lda_acc"].mean())
        print(f"Baseline (results/loso_metrics.csv): Riemannian MDM mean acc="
              f"{summary['baseline_riemann_acc_mean']:.3f}, CSP+LDA mean acc="
              f"{summary['baseline_csp_lda_acc_mean']:.3f}")

    if not args.skip_logvar:
        print(f"\nRunning DCD-RLS cleaning + log-bandpower (no spatial filter) + LDA "
              f"(taps_per_ref={args.taps_per_ref}, Nu={args.Nu}, Mb={args.Mb})...")
        t0 = time.time()
        res = evaluate_loso_dcd_rls(
            X_eeg, X_eog, y, subjects,
            taps_per_ref=args.taps_per_ref, forgetting_factor=args.forgetting_factor,
            Nu=args.Nu, Mb=args.Mb,
        )
        elapsed = time.time() - t0
        pd.DataFrame(res).to_csv(args.results_dir / "dcd_rls_metrics.csv", index=False)
        summary["dcd_rls_acc_mean"] = float(np.mean(res["acc"]))
        summary["dcd_rls_acc_std"] = float(np.std(res["acc"]))
        summary["dcd_rls_auc_mean"] = float(np.nanmean(res["auc"]))
        summary["dcd_rls_elapsed_seconds"] = round(elapsed, 1)
        print(f"  mean acc = {summary['dcd_rls_acc_mean']:.3f} +/- {summary['dcd_rls_acc_std']:.3f}  "
              f"mean AUC = {summary['dcd_rls_auc_mean']:.3f}  [{elapsed:.1f}s]")

    if not args.skip_csp:
        print(f"\nRunning DCD-RLS cleaning + CSP (fit per-fold) + LDA "
              f"(csp_n_components={args.csp_n_components})...")
        t0 = time.time()
        res_csp = evaluate_loso_dcd_rls_csp(
            X_eeg, X_eog, y, subjects,
            taps_per_ref=args.taps_per_ref, forgetting_factor=args.forgetting_factor,
            Nu=args.Nu, Mb=args.Mb, csp_n_components=args.csp_n_components,
        )
        elapsed_csp = time.time() - t0
        pd.DataFrame(res_csp).to_csv(args.results_dir / "dcd_rls_csp_metrics.csv", index=False)
        summary["dcd_rls_csp_acc_mean"] = float(np.mean(res_csp["acc"]))
        summary["dcd_rls_csp_acc_std"] = float(np.std(res_csp["acc"]))
        summary["dcd_rls_csp_auc_mean"] = float(np.nanmean(res_csp["auc"]))
        summary["dcd_rls_csp_elapsed_seconds"] = round(elapsed_csp, 1)
        print(f"  mean acc = {summary['dcd_rls_csp_acc_mean']:.3f} +/- {summary['dcd_rls_csp_acc_std']:.3f}  "
              f"mean AUC = {summary['dcd_rls_csp_auc_mean']:.3f}  [{elapsed_csp:.1f}s]")

    summary_path = args.results_dir / "dcd_rls_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
