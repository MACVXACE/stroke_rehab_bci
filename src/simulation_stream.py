"""
simulation_stream.py
======================
Circular-buffer streaming simulator that turns a held-out patient's
continuous epoched recording into a mock real-time data feed, exactly
as Script 05 needs it (Section 6):

    * 2000-sample (4.0 s @ 500 Hz) sliding window
    * advances 125 samples (250 ms) per step
    * yields (window, probability) pairs by pushing each window through
      a pre-trained pipeline's `predict_proba`

This module does NOT fit anything -- `pipeline` is passed in already
trained (loaded from models/trained_loso_pipeline.joblib), so there is
no leakage concern here; it is pure inference-time plumbing.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator, NamedTuple, Optional

import numpy as np

WINDOW_SAMPLES = 2000   # 4.0 s @ 500 Hz (Section 6, Script 05)
STEP_SAMPLES = 125      # 250 ms step size (Section 6, Script 05)


class StreamPrediction(NamedTuple):
    window_start_sample: int
    window: np.ndarray          # (n_channels, WINDOW_SAMPLES)
    proba_left: float
    proba_right: float
    predicted_label: int        # 0 = left, 1 = right


class CircularEEGBuffer:
    """
    Fixed-size sliding window over a continuous (n_channels, n_times)
    recording, advancing in fixed-size steps to emulate an online
    acquisition + inference loop.
    """

    def __init__(
        self,
        continuous_data: np.ndarray,
        window_samples: int = WINDOW_SAMPLES,
        step_samples: int = STEP_SAMPLES,
    ):
        """
        Parameters
        ----------
        continuous_data : np.ndarray, shape (n_channels, n_times_total)
            A held-out subject's concatenated trial data (or a single
            long trial), acting as the mock real-time stream.
        """
        if continuous_data.ndim != 2:
            raise ValueError(
                f"Expected (n_channels, n_times), got shape {continuous_data.shape}"
            )
        self.data = continuous_data
        self.n_channels, self.n_times = continuous_data.shape
        self.window_samples = window_samples
        self.step_samples = step_samples

        if self.n_times < window_samples:
            raise ValueError(
                f"Recording has only {self.n_times} samples; need at least "
                f"{window_samples} to fill one window."
            )

    def windows(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (start_sample, window) pairs advancing by `step_samples`."""
        start = 0
        while start + self.window_samples <= self.n_times:
            yield start, self.data[:, start:start + self.window_samples]
            start += self.step_samples


class StreamingPredictor:
    """
    Wraps a pre-trained pipeline (CSP+LDA or Riemannian MDM, both expose
    `predict_proba`) and a `CircularEEGBuffer` to produce a sequence of
    `StreamPrediction`s, matching Script 05's inference loop.
    """

    def __init__(self, pipeline, continuous_data: np.ndarray,
                 window_samples: int = WINDOW_SAMPLES,
                 step_samples: int = STEP_SAMPLES):
        self.pipeline = pipeline
        self.buffer = CircularEEGBuffer(continuous_data, window_samples, step_samples)

    def stream(self) -> Iterator[StreamPrediction]:
        for start, window in self.buffer.windows():
            # pipeline expects a (n_trials, n_channels, n_times) batch
            proba = self.pipeline.predict_proba(window[np.newaxis, ...])[0]
            predicted_label = int(np.argmax(proba))
            yield StreamPrediction(
                window_start_sample=start,
                window=window,
                proba_left=float(proba[0]),
                proba_right=float(proba[1]),
                predicted_label=predicted_label,
            )


class RollingConfidence:
    """
    Small helper the Pygame GUI (Script 05) uses to smooth the raw
    per-window probability stream before comparing it against the
    activation threshold -- reduces flicker from single noisy windows
    triggering/un-triggering the neurofeedback animation.
    """

    def __init__(self, maxlen: int = 4):
        self._buf: deque[float] = deque(maxlen=maxlen)

    def push(self, proba_target: float) -> float:
        self._buf.append(proba_target)
        return sum(self._buf) / len(self._buf)

    def reset(self) -> None:
        self._buf.clear()
