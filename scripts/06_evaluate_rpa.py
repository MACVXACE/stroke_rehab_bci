#!/usr/bin/env python3
"""
06_evaluate_rpa.py
====================
Compares the zero-calibration LOSO baseline (results/loso_metrics.csv,
from run_full_loso.py) against cross-subject transfer variants in
src/domain_adaptation.py, reusing the SAME cached epochs run_full_loso.py
already produced (data/processed/liu2024_epochs.npz) -- no re-download
needed.

Default behavior runs the two fast, low-risk variants:
  - unsupervised RPA (re-center + scale, zero calibration labels)
  - tangent-space, no rotation (re-center -> tangent-project -> LDA)
Both the full-manifold rotation and the tangent-space-rotation (TSA)
variants are OFF by default -- opt in explicitly (see flags below) since
one is a documented negative-transfer result and the other hasn't been
validated on the real cohort yet.

Usage
-----
    python scripts/06_evaluate_rpa.py                          # unsupervised RPA + tangent-space (LDA)
    python scripts/06_evaluate_rpa.py --subjects 1,2,3,4,5      # quick partial run
    python scripts/06_evaluate_rpa.py --check-conditioning      # pre-flight: scan for near-singular trials
    python scripts/06_evaluate_rpa.py --run-tsa-rotation        # also run Tangent Space Alignment (rotation)
    python scripts/06_evaluate_rpa.py --run-full-rotation       # reproduce the documented negative-transfer result

Writes
------
    results/rpa_unsupervised_metrics.csv
    results/tangent_space_metrics.csv
    results/rpa_rotation_metrics.csv     (only with --run-full-rotation)
    results/tsa_rotation_metrics.csv     (only with --run-tsa-rotation)
    results/rpa_comparison_summary.json
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

from src.domain_adaptation import (  # noqa: E402
    check_covariance_conditioning,
    evaluate_loso_rpa_rotation,
    evaluate_loso_rpa_unsupervised,
    evaluate_loso_tangent_space,
    evaluate_loso_tsa_rotation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_CACHE = REPO_ROOT / "data" / "processed" / "liu2024_epochs.npz"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def parse_args():
    p = argparse.ArgumentParser(description="Compare cross-subject transfer variants against the LOSO baseline.")
    p.add_argument("--data-cache", type=Path, default=DEFAULT_DATA_CACHE,
                    help=f"Cached (X, y, subjects) .npz from run_full_loso.py (default: {DEFAULT_DATA_CACHE}).")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--subjects", type=str, default=None,
                    help="Comma-separated subject IDs to restrict to (default: all cached subjects).")
    p.add_argument("--n-calib-per-class", type=int, default=4,
                    help="Labeled calibration trials per class for every calibrated variant.")
    p.add_argument("--check-conditioning", action="store_true",
                    help="Run check_covariance_conditioning() on the loaded data first and print a "
                         "report (median/max condition number, any non-finite covariances) before "
                         "fitting anything -- use this if a run seems to stall, to rule out "
                         "near-singular trials as the cause before assuming a library bug.")

    p.add_argument("--skip-unsupervised", action="store_true")

    p.add_argument("--skip-tangent-space", action="store_true",
                    help="Skip the rotation-free tangent-space + LDA variant (on by default).")

    p.add_argument("--run-tsa-rotation", action="store_true",
                    help="Run Tangent Space Alignment (TLRotate in tangent-space/PCA-anchor mode). "
                         "OFF by default -- not yet validated on the real cohort. Fast in testing "
                         "(~0.7s/fold) but not risk-free; use --check-conditioning first.")
    p.add_argument("--tsa-n-components", type=int, default=2)
    p.add_argument("--tsa-expl-var", type=float, default=0.999)
    p.add_argument("--tsa-n-clusters", type=int, default=2,
                    help="Must be <= --n-calib-per-class or TSA silently falls back to "
                         "class-mean-only anchors for the target domain.")

    p.add_argument("--run-full-rotation", action="store_true",
                    help="Run the full-manifold TLRotate variant (Grassmannian optimization over "
                         "all n_channels(n_channels-1)/2 params). OFF by default: measured at "
                         "38.7%% vs 50.7%% baseline (severe negative transfer) and ~15-20s/fold "
                         "on the real cohort. Kept only for reproducing that documented result.")
    p.add_argument("--rotate-maxiter", type=int, default=100)
    p.add_argument("--rotate-tol-step", type=float, default=1e-6)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data_cache.exists():
        print(f"[!] No cached epochs at {args.data_cache}.")
        print("    Run run_full_loso.py first -- it caches (X, y, subjects) there "
              "after the first successful MOABB pull, and this script reuses that cache.")
        return 1

    data = np.load(args.data_cache)
    X, y, subjects = data["X"], data["y"], data["subjects"]
    if args.subjects:
        wanted = [int(s.strip()) for s in args.subjects.split(",")]
        mask = np.isin(subjects, wanted)
        X, y, subjects = X[mask], y[mask], subjects[mask]

    n_subjects = len(np.unique(subjects))
    print(f"Loaded {len(y)} trials across {n_subjects} subjects.")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.check_conditioning:
        print("\nChecking OAS covariance conditioning across all loaded trials...")
        report = check_covariance_conditioning(X, subjects)
        print(f"  median condition number: {report['median_cond']:.3g}")
        print(f"  max condition number:    {report['max_cond']:.3g}")
        print(f"  non-finite covariances:  {report['n_nonfinite']}/{len(y)}")
        if "worst_subjects" in report:
            print(f"  worst 5 subjects (by median cond): {report['worst_subjects']}")
        if report["n_nonfinite"] > 0 or report["max_cond"] > 1e12:
            print("  [!] Near-singular or non-finite trials found -- this is a much more likely "
                  "explanation for any prior stall than a plain SVD/singularity issue.")
        else:
            print("  Looks well-conditioned -- a prior stall was more likely a lack of progress "
                  "feedback than a genuine numerical problem.")

    summary = {"n_subjects": n_subjects}

    baseline_path = args.results_dir / "loso_metrics.csv"
    if baseline_path.exists():
        base_df = pd.read_csv(baseline_path)
        summary["baseline_riemann_acc_mean"] = float(base_df["riemann_acc"].mean())
        summary["baseline_csp_lda_acc_mean"] = float(base_df["csp_lda_acc"].mean())
        print(f"\nBaseline (results/loso_metrics.csv): "
              f"Riemannian MDM mean acc={summary['baseline_riemann_acc_mean']:.3f}, "
              f"CSP+LDA mean acc={summary['baseline_csp_lda_acc_mean']:.3f}")
    else:
        print("\n[i] No results/loso_metrics.csv found -- run run_full_loso.py first for a baseline "
              "to compare against. Continuing with transfer-only numbers.")

    if not args.skip_unsupervised:
        print("\nRunning RPA (unsupervised: re-center + scale)...")
        t0 = time.time()
        res = evaluate_loso_rpa_unsupervised(X, y, subjects)
        pd.DataFrame(res).to_csv(args.results_dir / "rpa_unsupervised_metrics.csv", index=False)
        summary["rpa_unsupervised_acc_mean"] = float(np.mean(res["acc"]))
        summary["rpa_unsupervised_acc_std"] = float(np.std(res["acc"]))
        print(f"  mean acc = {summary['rpa_unsupervised_acc_mean']:.3f} "
              f"+/- {summary['rpa_unsupervised_acc_std']:.3f}  [{time.time() - t0:.1f}s]")

    if not args.skip_tangent_space:
        print(f"\nRunning tangent-space transfer (re-center + tangent-project + LDA, no rotation, "
              f"{args.n_calib_per_class} calib trials/class)...")
        t0 = time.time()
        res = evaluate_loso_tangent_space(X, y, subjects, n_calib_per_class=args.n_calib_per_class)
        pd.DataFrame(res).to_csv(args.results_dir / "tangent_space_metrics.csv", index=False)
        summary["tangent_space_acc_mean"] = float(np.mean(res["acc"]))
        summary["tangent_space_acc_std"] = float(np.std(res["acc"]))
        print(f"  mean acc = {summary['tangent_space_acc_mean']:.3f} "
              f"+/- {summary['tangent_space_acc_std']:.3f}  [{time.time() - t0:.1f}s]")

    if args.run_tsa_rotation:
        print(f"\nRunning Tangent Space Alignment (rotation via PCA anchors, "
              f"{args.n_calib_per_class} calib trials/class)...")
        t0 = time.time()
        res = evaluate_loso_tsa_rotation(
            X, y, subjects,
            n_calib_per_class=args.n_calib_per_class,
            n_components=args.tsa_n_components,
            expl_var=args.tsa_expl_var,
            n_clusters=args.tsa_n_clusters,
        )
        pd.DataFrame(res).to_csv(args.results_dir / "tsa_rotation_metrics.csv", index=False)
        summary["tsa_rotation_acc_mean"] = float(np.mean(res["acc"]))
        summary["tsa_rotation_acc_std"] = float(np.std(res["acc"]))
        print(f"  mean acc = {summary['tsa_rotation_acc_mean']:.3f} "
              f"+/- {summary['tsa_rotation_acc_std']:.3f}  [{time.time() - t0:.1f}s]")

    if args.run_full_rotation:
        print(f"\nRunning RPA (full-manifold rotation, "
              f"{args.n_calib_per_class} calib trials/class, rotate_maxiter={args.rotate_maxiter})...")
        t0 = time.time()
        res = evaluate_loso_rpa_rotation(
            X, y, subjects,
            n_calib_per_class=args.n_calib_per_class,
            rotate_maxiter=args.rotate_maxiter,
            rotate_tol_step=args.rotate_tol_step,
        )
        pd.DataFrame(res).to_csv(args.results_dir / "rpa_rotation_metrics.csv", index=False)
        summary["rpa_rotation_acc_mean"] = float(np.mean(res["acc"]))
        summary["rpa_rotation_acc_std"] = float(np.std(res["acc"]))
        print(f"  mean acc = {summary['rpa_rotation_acc_mean']:.3f} "
              f"+/- {summary['rpa_rotation_acc_std']:.3f}  [{time.time() - t0:.1f}s]")

    summary_path = args.results_dir / "rpa_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
