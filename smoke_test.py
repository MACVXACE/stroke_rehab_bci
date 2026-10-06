#!/usr/bin/env python3
"""
smoke_test.py
==============
Fast sanity check before committing to the full 50-subject LOSO-CV run.

Downloads/loads ONLY subjects 1 and 2, then verifies:
  1. 29 EEG channels are present (EOG dropped).
  2. Sampling rate is 500 Hz.
  3. The 8-30 Hz band-pass was actually applied (checked via epochs.info).
  4. CustomCSP.fit/.transform run without shape mismatches.
  5. The real pipelines (mne CSP+LDA, pyriemann Covariances+MDM) fit/predict
     without error on this tiny 2-subject slice.

This is a PLUMBING check only — fitting/predicting on just 2 subjects
pooled together is NOT a valid subject-independent estimate. Nothing
here should be reported as an accuracy number; that's what
`run_full_loso.py`'s real LOSO-CV is for.

Usage:
    python smoke_test.py
Exit code 0 = all checks passed, non-zero = something needs attention
before you kick off the long run.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

SMOKE_SUBJECTS = [1, 2]
TOL_HZ = 0.5  # tolerance when comparing reported filter edges to 8/30 Hz

PASS, FAIL = "\u2705", "\u274c"


def check(label: str, fn):
    """Run one check, print PASS/FAIL, return True/False, never raises."""
    try:
        fn()
        print(f"{PASS} {label}")
        return True
    except AssertionError as e:
        print(f"{FAIL} {label} — {e}")
        return False
    except Exception as e:  # noqa: BLE001 — smoke test wants to keep going
        print(f"{FAIL} {label} — unexpected error: {e}")
        traceback.print_exc(limit=2)
        return False


def main() -> int:
    print(f"=== Smoke test: subjects {SMOKE_SUBJECTS} ===\n")
    results = []

    # -- Step 1: imports -----------------------------------------------------
    def _imports():
        global get_dataset, get_paradigm, N_EEG_CHANNELS, SFREQ_HZ, FMIN, FMAX
        global CustomCSP, build_csp_lda_pipeline, build_riemann_pipeline
        from src.data_loader import get_dataset, get_paradigm, N_EEG_CHANNELS, SFREQ_HZ, FMIN, FMAX
        from src.feature_extraction import CustomCSP
        from src.pipelines import build_csp_lda_pipeline, build_riemann_pipeline
    results.append(check("Import project modules (src.data_loader, src.feature_extraction, src.pipelines)", _imports))
    if not results[-1]:
        print("\nCan't continue without these imports — check your environment (see README setup).")
        return 1

    # -- Step 2: download/load 2 subjects (return_epochs for rich metadata) --
    print(f"\nDownloading/loading subjects {SMOKE_SUBJECTS} via MOABB "
          f"(first run may take a few minutes)...")
    t0 = time.time()
    dataset = get_dataset()
    paradigm = get_paradigm()
    try:
        epochs_X, labels, meta = paradigm.get_data(
            dataset=dataset, subjects=SMOKE_SUBJECTS, return_epochs=True
        )
    except Exception as e:
        print(f"{FAIL} Data download/epoching failed: {e}")
        print("  Check internet access and that moabb>=1.0 is installed (pip install -U moabb).")
        return 1
    print(f"  Loaded in {time.time() - t0:.1f}s. n_trials={len(epochs_X)}")

    # -- Step 3: channel count ------------------------------------------------
    def _channels():
        import mne
        eeg_picks = mne.pick_types(epochs_X.info, eeg=True, eog=False)
        assert len(eeg_picks) == N_EEG_CHANNELS, \
            f"expected {N_EEG_CHANNELS} EEG channels, got {len(eeg_picks)}"
    results.append(check(f"29 EEG channels present", _channels))

    # -- Step 4: sampling rate -------------------------------------------------
    def _sfreq():
        sfreq = epochs_X.info["sfreq"]
        assert abs(sfreq - SFREQ_HZ) < 1e-6, f"expected {SFREQ_HZ} Hz, got {sfreq} Hz"
    results.append(check(f"Sampling rate == {SFREQ_HZ} Hz", _sfreq))

    # -- Step 5: band-pass edges ------------------------------------------------
    def _bandpass():
        hp, lp = epochs_X.info["highpass"], epochs_X.info["lowpass"]
        assert abs(hp - FMIN) < TOL_HZ, f"highpass={hp}, expected ~{FMIN}"
        assert abs(lp - FMAX) < TOL_HZ, f"lowpass={lp}, expected ~{FMAX}"
    results.append(check(f"Band-pass ~{FMIN}-{FMAX} Hz applied", _bandpass))

    # -- Step 6: CustomCSP fit/transform shape check ----------------------------
    X = epochs_X.get_data(copy=True)  # (n_trials, n_channels, n_times)
    unique_labels = sorted(set(labels))
    y = np.array([0 if l == unique_labels[0] else 1 for l in labels])

    def _custom_csp():
        csp = CustomCSP(n_components=4)
        feats = csp.fit_transform(X, y)
        assert feats.shape == (X.shape[0], 4), f"unexpected feature shape {feats.shape}"
        assert np.isfinite(feats).all(), "CustomCSP produced non-finite features"
    results.append(check("CustomCSP.fit_transform — shapes and finiteness OK", _custom_csp))

    # -- Step 7: real pipelines fit/predict --------------------------------------
    def _csp_lda_pipeline():
        pipe = build_csp_lda_pipeline()
        pipe.fit(X, y)
        preds = pipe.predict(X)
        probs = pipe.predict_proba(X)
        assert preds.shape == (X.shape[0],)
        assert probs.shape == (X.shape[0], 2)
    results.append(check("mne CSP + LDA pipeline fit/predict — no shape mismatch", _csp_lda_pipeline))

    def _riemann_pipeline():
        pipe = build_riemann_pipeline()
        pipe.fit(X, y)
        preds = pipe.predict(X)
        probs = pipe.predict_proba(X)
        assert preds.shape == (X.shape[0],)
        assert probs.shape == (X.shape[0], 2)
    results.append(check("pyriemann Covariances + MDM pipeline fit/predict — no shape mismatch", _riemann_pipeline))

    # -- summary -------------------------------------------------------------
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} checks passed.")
    if n_pass == n_total:
        print(f"{PASS} Environment looks good — safe to launch run_full_loso.py.")
        return 0
    else:
        print(f"{FAIL} Fix the failing checks above before starting the full LOSO-CV run.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
