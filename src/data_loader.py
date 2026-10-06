"""
data_loader.py
================
MOABB dataset ingestion helpers for the Liu et al. (2024) acute-stroke
motor imagery (MI) EEG dataset (Xuanwu Hospital, Capital Medical
University; 50 patients; ZhenTec NT1 wireless system; 29 EEG + 2 EOG
channels @ 500 Hz; reference CPz, ground FPz).

Reference: Liu H. et al. "An EEG motor imagery dataset for brain
computer interface in acute stroke patients." Sci Data 11, 131 (2024).
https://doi.org/10.1038/s41597-023-02787-8

IMPORTANT -- ANTI-LEAKAGE CONTRACT
-----------------------------------
This module ONLY loads and lightly reshapes data. It never fits,
transforms, normalizes, or otherwise "learns" anything from the
signal. Every supervised transform (CSP, covariance estimation, LDA,
MDM) must be fit exclusively inside a training fold -- see
`src/pipelines.py::evaluate_loso`, which is the only place `.fit()`
is allowed to touch these arrays.

Requires: moabb>=1.0 (Liu2024 was added in a relatively recent
release -- if the import below fails, run `pip install -U moabb` and
confirm with `moabb.datasets.utils.dataset_search("Liu2024")`).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

try:
    from moabb.datasets import Liu2024
    from moabb.paradigms import MotorImagery
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "moabb (with the Liu2024 dataset) is required for data_loader.py.\n"
        "  pip install -U moabb\n"
        "If Liu2024 still isn't found under moabb.datasets, check "
        "moabb.datasets.utils.dataset_search('Liu2024') for the current "
        "import path in your installed version."
    ) from exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants (Section 3 of the project blueprint)
# ---------------------------------------------------------------------------
N_EEG_CHANNELS = 29          # 10-10 system, active electrodes
N_EOG_CHANNELS = 2
SFREQ_HZ = 500.0
REFERENCE_CHANNEL = "CPz"
GROUND_CHANNEL = "FPz"

FMIN, FMAX = 8.0, 30.0        # mu + beta band, zero-phase FIR (Section 4.1)
TMIN, TMAX = 0.0, 4.0         # post-cue motor-imagery window (Section 3)
N_TIMES = int((TMAX - TMIN) * SFREQ_HZ)   # 2000 samples

TRIALS_PER_SUBJECT = 40       # 20 left, 20 right (Section 3)

LABEL_NAMES = {0: "left_hand", 1: "right_hand"}


# ---------------------------------------------------------------------------
# Dataset / paradigm construction
# ---------------------------------------------------------------------------
def get_dataset() -> "Liu2024":
    """Instantiate the Liu et al. (2024) 50-patient acute-stroke MI dataset."""
    return Liu2024()


def get_paradigm(fmin: float = FMIN, fmax: float = FMAX,
                  tmin: float = TMIN, tmax: float = TMAX) -> "MotorImagery":
    """
    Build the MOABB MotorImagery paradigm with the protocol's canonical
    zero-phase FIR band (8-30 Hz) and the 4 s post-cue MI window.

    This paradigm object is responsible for the ONLY filtering/epoching
    step that is allowed to touch the full dataset before subject
    splitting -- it is an unsupervised, fixed-parameter operation
    (a band-pass filter + a fixed time crop), not a fitted statistical
    transform, so it does not leak label information across subjects.
    """
    return MotorImagery(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)


def load_subject_raw(subject_id: int):
    """
    Return the raw (un-epoched) MNE data for a single subject.

    Intended for Notebook 01's sensor-layout / PSD auditing, where we
    want to inspect channel counts, sampling rate, and reference before
    any epoching or filtering happens.
    """
    dataset = get_dataset()
    data = dataset.get_data(subjects=[subject_id])
    return data[subject_id]


def load_all_subjects(
    subject_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load epoched MI trials for all (or a subset of) subjects.

    Returns
    -------
    X : np.ndarray, shape (n_trials_total, n_eeg_channels, n_times)
        EEG-only epochs (EOG channels dropped). n_times == N_TIMES (2000).
    y : np.ndarray, shape (n_trials_total,)
        Binary labels: 0 = left_hand, 1 = right_hand.
    subjects : np.ndarray, shape (n_trials_total,)
        Subject identifier per trial. REQUIRED for LOSO-CV grouping --
        never discard this array. `evaluate_loso` (src/pipelines.py)
        groups exclusively on it and will silently do the wrong thing
        (or crash) without it.
    """
    dataset = get_dataset()
    paradigm = get_paradigm()

    subject_ids = subject_ids or dataset.subject_list

    X, labels, meta = paradigm.get_data(
        dataset=dataset, subjects=subject_ids, return_epochs=False
    )

    # Section 8, "Montage Mismatch": assert EOG channels were dropped
    # rather than silently trusting the paradigm defaults.
    n_channels = X.shape[1]
    if n_channels != N_EEG_CHANNELS:
        logger.warning(
            "Expected %d EEG channels, got %d. Verify the 2 EOG channels "
            "were excluded upstream (see Notebook 01's channel-type audit "
            "and drop them explicitly with raw.drop_channels(...) if not).",
            N_EEG_CHANNELS, n_channels,
        )

    y = _encode_labels(labels)
    subjects = meta["subject"].to_numpy()

    return X, y, subjects


def load_all_subjects_with_eog(
    subject_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Like `load_all_subjects`, but additionally returns the 2 EOG channels
    as a separate, time-aligned array -- needed as the reference input
    for src/adaptive_filtering.py's DCD-RLS artifact cancellation, which
    the default `load_all_subjects` doesn't provide (it deliberately
    drops EOG per the anti-leakage/montage-mismatch contract above).

    VERSION-INDEPENDENT BY DESIGN: does not use the dataset-level
    `return_all_modalities` kwarg at all (dropped after it hit a
    `TypeError` in a moabb install without it). Instead it discovers the
    real channel names directly from the dataset's own raw data (one
    lightweight probe fetch via the low-level `dataset.get_data()`, not
    the paradigm), then passes that explicit name list to
    `MotorImagery(..., channels=...)`.

    This matters for a second reason, not just version robustness:
    `RawToEpochs.transform` (moabb/datasets/preprocessing.py) resolves
    channels two different ways --
      - `channels=None` (the default used elsewhere in this module) ->
        `pick_channels_for_modalities(raw.info, return_all_modalities)`,
        which is type-based (`mne.pick_types(eeg=True, ...)` when False).
      - `channels=[...]` (explicit names, used here) ->
        `mne.pick_channels(available_channels, include=self.channels)`,
        purely name-based, with NO dependency on return_all_modalities.
    Passing explicit names also means the EOG channels stay correctly
    TYPED as 'eog' the whole time -- confirmed directly (see dev notes
    below) that the paradigm's band-pass filter step uses
    `raw.filter(..., picks="data")`, and `picks="data"` does NOT include
    EOG-typed channels. So this approach never touches the EOG channels
    with the 8-30 Hz EEG band-pass, unlike an earlier considered
    alternative (relabeling EOG channels to type 'eeg' to smuggle them
    through type-based picking) which WOULD have band-pass-filtered them
    and destroyed the low-frequency blink/saccade content that makes
    them useful as a reference signal in the first place. Don't
    reintroduce that relabeling trick.

    Returns
    -------
    X_eeg : np.ndarray, shape (n_trials_total, n_eeg_channels, n_times)
    X_eog : np.ndarray, shape (n_trials_total, n_eog_channels, n_times)
        Time-aligned with X_eeg -- X_eog[i] is trial i's EOG channels,
        same trial, same time window as X_eeg[i].
    y : np.ndarray, shape (n_trials_total,)
    subjects : np.ndarray, shape (n_trials_total,)
    """
    import mne  # local import: only needed for this EOG-splitting path

    dataset = get_dataset()  # plain construction, no version-sensitive kwargs
    subject_ids = subject_ids or dataset.subject_list

    # Probe one subject's raw data directly (bypassing the paradigm) purely
    # to discover real channel names -- {subject: {session: {run: raw}}}.
    probe = dataset.get_data(subjects=[subject_ids[0]])
    probe_session = next(iter(probe[subject_ids[0]].values()))
    probe_raw = next(iter(probe_session.values()))

    ch_names = probe_raw.info["ch_names"]
    ch_kinds = probe_raw.get_channel_types()
    keep_channels = [ch for ch, kind in zip(ch_names, ch_kinds) if kind in ("eeg", "eog")]
    n_probe_eog = sum(1 for _, kind in zip(ch_names, ch_kinds) if kind == "eog")

    if n_probe_eog == 0:
        raise RuntimeError(
            f"0 EOG-typed channels found in the raw data itself (channel types seen: "
            f"{sorted(set(ch_kinds))}). The Liu2024 dataset is documented (moabb's own "
            f"metadata and channel-typing code) to have 2 EOG channels named VEOR/HEOL -- "
            f"if this fires, the installed moabb version's Liu2024._get_single_subject_data "
            f"may differ from what this function was written against. Inspect "
            f"probe_raw.info['ch_names'] / probe_raw.get_channel_types() directly before "
            f"proceeding; do not silently continue with zero reference channels."
        )

    paradigm = MotorImagery(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX, channels=keep_channels)

    epochs, labels, meta = paradigm.get_data(
        dataset=dataset, subjects=subject_ids, return_epochs=True
    )

    eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False)
    eog_picks = mne.pick_types(epochs.info, eeg=False, eog=True)
    if len(eeg_picks) != N_EEG_CHANNELS:
        logger.warning("Expected %d EEG channels, got %d.", N_EEG_CHANNELS, len(eeg_picks))

    if len(eog_picks) == 0:
        raise RuntimeError(
            f"0 EOG channels survived epoching even though the probe found {n_probe_eog} "
            f"and they were explicitly requested via channels=. This means something in "
            f"the epochs pipeline dropped them after the paradigm call -- inspect "
            f"epochs.info['ch_names'] directly rather than assuming this function's "
            f"channel-name plumbing is still correct. Raising here instead of returning "
            f"an empty reference array: src/adaptive_filtering.py's DCD-RLS would otherwise "
            f"silently 'clean' every channel against zero reference signal and produce a "
            f"misleadingly-normal-looking but meaningless result -- this is exactly what "
            f"happened before this check existed."
        )
    if len(eog_picks) != N_EOG_CHANNELS:
        logger.warning(
            "Expected %d EOG channels, got %d (nonzero, but not the expected count -- "
            "proceeding, but verify these are really the intended reference channels).",
            N_EOG_CHANNELS, len(eog_picks),
        )

    data = epochs.get_data(copy=True)  # (n_trials, n_all_channels, n_times)
    X_eeg = data[:, eeg_picks, :]
    X_eog = data[:, eog_picks, :]

    y = _encode_labels(labels)
    subjects = meta["subject"].to_numpy()

    return X_eeg, X_eog, y, subjects


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    """Map MOABB's string event labels to {0: left_hand, 1: right_hand}."""
    labels = np.asarray(labels)
    unique = sorted(set(labels))
    if len(unique) != 2:
        raise ValueError(f"Expected binary MI labels, found: {unique}")
    # 'left_hand' sorts before 'right_hand' alphabetically, which
    # conveniently matches the 0/1 convention used throughout this repo --
    # but we map explicitly rather than relying on that being stable.
    preferred_order = ["left_hand", "right_hand"]
    if set(unique) == set(preferred_order):
        label_map = {name: i for i, name in enumerate(preferred_order)}
    else:
        label_map = {unique[0]: 0, unique[1]: 1}
        logger.warning(
            "Unexpected label set %s; falling back to positional mapping %s. "
            "Verify this matches left_hand=0 / right_hand=1.",
            unique, label_map,
        )
    return np.array([label_map[label] for label in labels])
