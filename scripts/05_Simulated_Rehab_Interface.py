"""
05_Simulated_Rehab_Interface.py
=================================
Standalone script modeling real-time clinical deployment of the
trained LOSO pipeline (Section 6 of the project blueprint).

This is a SIMULATION: it replays a held-out patient's already-recorded
epochs through a 4.0 s / 250 ms sliding window (src/simulation_stream.py)
and drives a Pygame closed-loop neurofeedback GUI as if the data were
arriving live. No hardware acquisition happens here.

Usage
-----
    python scripts/05_Simulated_Rehab_Interface.py --subject 7

Controls
--------
    ESC or window close  -> quit
    SPACE                -> advance to the next trial cue manually
                             (auto-advances every N steps otherwise)

Neurofeedback rule (Section 6): if the *smoothed* predicted probability
for the cued class exceeds ACTIVATION_THRESHOLD, the hand-grasp
animation plays (reinforcing correct cortical firing). Otherwise the
hand freezes -- incorrect/uncertain decoding gets no visual reward.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_all_subjects  # noqa: E402
from src.pipelines import DEFAULT_ACTIVATION_THRESHOLD  # noqa: E402
from src.simulation_stream import RollingConfidence, StreamingPredictor  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "trained_loso_pipeline.joblib"

# -- display -----------------------------------------------------------------
WIDTH, HEIGHT = 900, 560
FPS = 30
BG_COLOR = (18, 20, 28)
TEXT_COLOR = (235, 235, 240)
LEFT_COLOR = (90, 170, 250)
RIGHT_COLOR = (250, 140, 90)
BAR_BG = (45, 48, 60)
FREEZE_COLOR = (90, 92, 100)
ACTIVATE_COLOR = (110, 220, 150)

ACTIVATION_THRESHOLD = DEFAULT_ACTIVATION_THRESHOLD  # P > 0.65, Section 6


def parse_args():
    parser = argparse.ArgumentParser(description="Simulated closed-loop MI-BCI rehab interface.")
    parser.add_argument("--subject", type=int, default=None,
                         help="Held-out subject ID to replay. Defaults to the "
                              "first subject with data available.")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH),
                         help="Path to a trained_loso_pipeline.joblib.")
    return parser.parse_args()


def load_holdout_stream(subject_id: int | None):
    """
    Load one subject's epochs and concatenate them along time to form a
    single continuous mock stream, plus the per-trial cue labels so the
    GUI can show what SHOULD have been decoded (for visual comparison).
    """
    X, y, subjects = load_all_subjects(subject_ids=[subject_id] if subject_id else None)
    if subject_id is None:
        subject_id = int(subjects[0])
        mask = subjects == subject_id
        X, y = X[mask], y[mask]

    # Concatenate trials along the time axis -> (n_channels, n_trials * n_times)
    continuous = np.concatenate([trial for trial in X], axis=1)
    trial_boundaries = [i * X.shape[2] for i in range(X.shape[0] + 1)]
    return continuous, y, trial_boundaries, subject_id


def cue_label_for_sample(sample_idx: int, trial_boundaries: list[int], y: np.ndarray) -> int:
    """Which trial (and therefore which cued class) a given sample index falls in."""
    for i in range(len(trial_boundaries) - 1):
        if trial_boundaries[i] <= sample_idx < trial_boundaries[i + 1]:
            return int(y[i])
    return int(y[-1])


def draw_arrow(surface, center, pointing_right: bool, color):
    x, y = center
    if pointing_right:
        points = [(x - 40, y - 30), (x + 10, y), (x - 40, y + 30)]
    else:
        points = [(x + 40, y - 30), (x - 10, y), (x + 40, y + 30)]
    pygame.draw.polygon(surface, color, points)


def draw_confidence_bar(surface, font, x, y, width, height, value, color, label):
    pygame.draw.rect(surface, BAR_BG, (x, y, width, height), border_radius=6)
    fill_w = int(width * max(0.0, min(1.0, value)))
    pygame.draw.rect(surface, color, (x, y, fill_w, height), border_radius=6)
    pygame.draw.rect(surface, TEXT_COLOR, (x, y, width, height), width=2, border_radius=6)
    txt = font.render(f"{label}: {value:.2f}", True, TEXT_COLOR)
    surface.blit(txt, (x, y - 26))


def draw_hand(surface, center, activated: bool, pulse: float):
    color = ACTIVATE_COLOR if activated else FREEZE_COLOR
    radius = 34 + (6 * pulse if activated else 0)
    pygame.draw.circle(surface, color, center, int(radius), width=0 if activated else 3)
    # simple "fingers" to read as a hand/grasp glyph
    for angle_deg in (-40, -13, 13, 40):
        import math
        rad = math.radians(angle_deg)
        fx = center[0] + math.sin(rad) * (radius + (14 if activated else 8))
        fy = center[1] - math.cos(rad) * (radius + (14 if activated else 8))
        pygame.draw.line(surface, color, center, (fx, fy), 6)


def main():
    args = parse_args()

    if not Path(args.model).exists():
        print(f"[!] No trained pipeline found at {args.model}.")
        print("    Run Notebook 04 first to produce models/trained_loso_pipeline.joblib.")
        sys.exit(1)

    pipeline = joblib.load(args.model)
    continuous, y, trial_boundaries, subject_id = load_holdout_stream(args.subject)
    predictor = StreamingPredictor(pipeline, continuous)
    smoother_left = RollingConfidence(maxlen=4)
    smoother_right = RollingConfidence(maxlen=4)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Simulated MI-BCI Rehab Interface — Subject {subject_id}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 22)
    small_font = pygame.font.SysFont("arial", 16)

    stream_iter = predictor.stream()
    pulse_t = 0.0
    running = True
    current = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        try:
            current = next(stream_iter)
        except StopIteration:
            running = False
            continue

        cue = cue_label_for_sample(current.window_start_sample, trial_boundaries, y)
        smoothed_left = smoother_left.push(current.proba_left)
        smoothed_right = smoother_right.push(current.proba_right)
        smoothed = [smoothed_left, smoothed_right]

        activated = smoothed[cue] > ACTIVATION_THRESHOLD
        pulse_t = (pulse_t + 0.15) % (2 * np.pi)
        pulse = (np.sin(pulse_t) + 1) / 2

        # -- draw --------------------------------------------------------
        screen.fill(BG_COLOR)

        cue_text = font.render(
            f"Cue: {'LEFT HAND' if cue == 0 else 'RIGHT HAND'}", True, TEXT_COLOR
        )
        screen.blit(cue_text, (WIDTH // 2 - cue_text.get_width() // 2, 30))
        draw_arrow(screen, (WIDTH // 2, 110), pointing_right=(cue == 1),
                   color=LEFT_COLOR if cue == 0 else RIGHT_COLOR)

        draw_confidence_bar(screen, small_font, 80, 230, 320, 28,
                             smoothed_left, LEFT_COLOR, "P(left)")
        draw_confidence_bar(screen, small_font, 500, 230, 320, 28,
                             smoothed_right, RIGHT_COLOR, "P(right)")

        draw_hand(screen, (WIDTH // 2, 400), activated, pulse)
        status = "NEUROFEEDBACK ACTIVE" if activated else "hold imagery..."
        status_color = ACTIVATE_COLOR if activated else FREEZE_COLOR
        status_txt = font.render(status, True, status_color)
        screen.blit(status_txt, (WIDTH // 2 - status_txt.get_width() // 2, 470))

        footer = small_font.render(
            f"window start sample {current.window_start_sample} | "
            f"threshold={ACTIVATION_THRESHOLD:.2f} | ESC to quit",
            True, (140, 142, 150),
        )
        screen.blit(footer, (20, HEIGHT - 30))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
