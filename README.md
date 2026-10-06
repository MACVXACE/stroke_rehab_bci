# Subject-Independent BCI for Acute Stroke Rehabilitation

A motor-imagery (MI) brain-computer interface pipeline that decodes
imagined left- vs. right-hand grasp from EEG in acute stroke patients,
evaluated for **subject-independent (zero-calibration) generalization**
via Leave-One-Subject-Out Cross-Validation (LOSO-CV), and demonstrated
through a simulated real-time closed-loop neurofeedback interface.

Built on the [Liu et al. (2024)](https://doi.org/10.1038/s41597-023-02787-8)
dataset — 50 acute stroke patients recorded at Xuanwu Hospital, Capital
Medical University — accessed via [MOABB](https://moabb.neurotechx.com/).

## Why subject-independent matters clinically

Acute stroke patients cannot tolerate the long per-patient calibration
sessions that most MI-BCI systems require — they're often recruited
within days of a hemorrhage or infarct, fatigue quickly, and can't
reliably attend multiple calibration blocks. A model trained once on a
population and deployed with **zero new calibration data** is what
makes closed-loop MI neurofeedback feasible in an acute ward. LOSO-CV
is the honest way to measure whether a pipeline actually achieves
that, since every fold's test subject is completely unseen during
training.

## Repository structure

```
stroke_rehab_bci/
├── data/raw/                          # MOABB's cached downloads land here (mne_data)
├── notebooks/
│   ├── 01_Data_Ingestion_and_EDA.ipynb
│   ├── 02_Signal_Processing_and_ERD.ipynb
│   ├── 03_Spatial_Filtering_CSP.ipynb
│   └── 04_Cross_Subject_Classification.ipynb
├── src/
│   ├── data_loader.py                 # MOABB ingestion (no fitting happens here)
│   ├── feature_extraction.py          # From-scratch CSP (NumPy/SciPy)
│   ├── pipelines.py                   # sklearn + pyriemann pipelines, LOSO-CV driver
│   └── simulation_stream.py           # Circular-buffer streaming simulator
├── models/
│   └── trained_loso_pipeline.joblib   # produced by Notebook 04
├── scripts/
│   └── 05_Simulated_Rehab_Interface.py  # Pygame closed-loop demo
├── environment.yml
├── requirements.txt
└── README.md
```

## Setup

```bash
conda env create -f environment.yml
conda activate stroke_rehab_bci
# or: pip install -r requirements.txt
```

The first run of Notebook 01 will trigger MOABB to download the
Liu2024 dataset into `~/mne_data` (cached thereafter). **This requires
outbound internet access to MOABB's data host** — if you're running
these notebooks in a network-restricted sandbox, the download step
will fail there and needs to be run somewhere with open internet
access first, after which the cached files can be copied into
`data/raw/`.

`Liu2024` was added to MOABB in a relatively recent release. If
`from moabb.datasets import Liu2024` fails, run `pip install -U moabb`
and confirm it's available with:
```python
from moabb.datasets.utils import dataset_search
dataset_search("Liu2024")
```

## Local execution runbook

### 1. Environment setup

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate stroke_rehab_bci
```

This pipeline is **classical ML only** (CSP, LDA, Riemannian geometry) —
there is no GPU dependency to verify. It's CPU-bound and single-threaded
per fold; if you want to sanity-check your BLAS backend for numpy/scipy:

```bash
python -c "import os; print('CPU cores:', os.cpu_count())"
python -c "import numpy; numpy.show_config()"
```

**Network requirement:** downloading Liu2024 pulls from
`ndownloader.figshare.com` (the dataset's primary host), with an optional
faster path through `data.nemar.org`. If you're behind a firewall or
running in a restricted sandbox, both hosts need outbound access — a
`403`/`Host not in allowlist` error on either is a network egress
problem, not a bug in this code.

**MOABB cache path:** downloaded/cached files default to `~/mne_data`.
To use a different location (e.g. limited disk on your home partition):
```python
from moabb.utils import set_download_dir
set_download_dir("/path/to/bigger/disk/mne_data")
```
or pass `--mne-data-dir /path/to/bigger/disk/mne_data` to `run_full_loso.py`.

### 2. Smoke test (run this first)

```bash
python smoke_test.py
```

Downloads only subjects 1 and 2 and checks: 29 EEG channels present,
500 Hz sampling rate, the 8-30 Hz band-pass was actually applied, and
that `CustomCSP` plus both real pipelines (`mne` CSP+LDA,
`pyriemann` Covariances+MDM) fit/transform/predict without shape
errors. Exits non-zero if anything fails — fix that before committing
to the full run. (These 2-subject numbers are a plumbing check only,
never a valid accuracy estimate.)

### 3. Full 50-subject LOSO-CV, checkpointed and resumable

```bash
python run_full_loso.py
```

- First call downloads all 50 subjects, caches the epoched arrays to
  `data/processed/liu2024_epochs.npz`, then runs LOSO-CV with a
  `tqdm` progress bar, fsync-ing each subject's metrics to
  `results/loso_metrics.csv` and predictions to
  `results/predictions/subject_<id>.npz` immediately after that fold
  finishes.
- **If it dies at subject 30** (network blip, Ctrl+C, OOM, whatever):
  just rerun the exact same command. Subjects already in
  `loso_metrics.csv` are skipped; only the remaining ones run.
- A subject whose fold raises an exception is logged to
  `results/run.log` and **not** checkpointed, so it's automatically
  retried on your next run rather than silently marked done.
- Useful flags: `--subjects 1,2,3,4,5` (quick partial run),
  `--force-restart` (wipe checkpoints and start over),
  `--skip-model-save` (skip the final deployment-model refit).
- When all subjects are done, it prints mean/std/SEM for accuracy,
  balanced accuracy, and ROC-AUC for both pipelines, writes
  `results/summary.json`, saves
  `results/confusion_matrix.png`, and refits the better-performing
  pipeline on **all** subjects to `models/trained_loso_pipeline.joblib`
  for the Script 05 demo (never cite that model's own fit as an
  accuracy number — that's what `loso_metrics.csv` is for).

### 4. Simulated closed-loop Pygame demo

Requires `models/trained_loso_pipeline.joblib` to exist (produced by
step 3, or by running Notebook 04 manually).

```bash
python scripts/05_Simulated_Rehab_Interface.py --subject 7
```

- `--subject <id>` picks which held-out patient's recording to replay
  as the mock real-time stream; omit it to use the first available
  subject.
- `--model path/to/other_pipeline.joblib` points at a different
  serialized pipeline if you want to compare CSP+LDA vs. Riemannian
  MDM interactively (train/save each separately, e.g. by editing
  `_save_deployment_model` or saving both from Notebook 04).
- The GUI shows the trial cue (left/right arrow), a smoothed
  confidence bar per class, and triggers the hand-grasp neurofeedback
  animation whenever the cued class's smoothed probability exceeds
  the `DEFAULT_ACTIVATION_THRESHOLD = 0.65` gate (`src/pipelines.py`).
  Press ESC or close the window to quit.

## Domain adaptation (Riemannian transfer learning)

`src/domain_adaptation.py` adds cross-subject transfer on top of the
LOSO-CV split, via `pyriemann.transfer` / `pyriemann.tangentspace`. Four
regimes, all still one-subject-held-out at a time — with real
50-subject results where available:

| Function | Mechanism | Calibration | Measured acc | Time/fold |
|---|---|---|---|---|
| `evaluate_loso_rpa_unsupervised` | `TLCenter`+`TLScale` (manifold) | none | **52.1% ± 6.3%** (baseline 50.7%) | ~0.05s |
| `evaluate_loso_tangent_space` | `TLCenter`→`TangentSpace`→LDA, no rotation | small (or 0) | not yet run on real data | ~0.05s |
| `evaluate_loso_tsa_rotation` | `TangentSpace`→`TLRotate` (PCA-anchor matching) | small | not yet run on real data | ~0.05-0.1s |
| `evaluate_loso_rpa_rotation` (alias: `..._supervised`) | full-manifold `TLRotate` (Grassmannian) | small | **38.7% ± 5.6% — severe negative transfer** | ~15-20s |

Script defaults now match this: `scripts/06_evaluate_rpa.py` runs the
two fast, low-risk variants (`rpa_unsupervised`, `tangent_space`) by
default; `--run-tsa-rotation` and `--run-full-rotation` are explicit
opt-ins.

```bash
python scripts/06_evaluate_rpa.py                        # unsupervised RPA + tangent-space (LDA)
python scripts/06_evaluate_rpa.py --check-conditioning    # pre-flight: scan real trials for near-singular covariances
python scripts/06_evaluate_rpa.py --run-tsa-rotation      # also try TSA (rotation via PCA anchors)
python scripts/06_evaluate_rpa.py --run-full-rotation     # reproduce the negative-transfer result
```

**Why `evaluate_loso_rpa_rotation` failed:** optimizing a full n×n
rotation matrix costs n(n-1)/2 free parameters — 406 for 29 channels —
against only ~8 calibration trials total. Confirmed catastrophic
overfitting on the real cohort (below-chance accuracy).

**On a reported "SVD deadlock":** `TLRotate`'s tangent-space code path
computes a (435, 435) cross-covariance matrix (`n(n+1)/2` for 29
channels) and SVDs it. Tested directly: SVD of a singular, rank-3, or
even all-zero 435×435 matrix completes in 30-70ms — singularity does
not hang SVD, that's one of its defining stability properties (unlike
matrix inversion). NaN/Inf-contaminated input can misbehave (raises
`LinAlgError` near-instantly on this codebase's platform; some BLAS
backends — notably macOS's Accelerate in certain OS/numpy combinations
— have documented hangs specifically on NaN input). Two more mundane
explanations are at least as likely: no progress indicator existed
before this version (fixed — every `evaluate_loso_*` function now shows
a `tqdm` bar), and 49 sequential source-domain fits per fold can
legitimately take a while even though each individual step is fast.
`_fit_with_diagnostics` now wraps every fit to turn a `LinAlgError` into
an actionable message instead of a bare trace or an ambiguous stall,
and `check_covariance_conditioning(X, subjects)` is a standalone
pre-flight check (also exposed as `--check-conditioning`) — run it on
your real data before a long fit if you want to rule out near-singular
trials up front rather than inferring it after the fact.

**Why `evaluate_loso_tangent_space` is the recommended default:** no
`TLRotate` anywhere — just `TLCenter` (closed-form per-domain geometric
mean) → `TangentSpace` (closed-form log-map) → LDA. Nothing iterative,
nothing that scales with n_channels², nothing that needs convergence.
Pass `n_calib_per_class=0` for a fully zero-calibration variant (the
held-out subject's own evaluation trials become their unlabeled
warm-start batch for `TLCenter`, same mechanism as
`evaluate_loso_rpa_unsupervised`).

**A caution on validating any of this with synthetic data:** the
affine-invariant Riemannian metric MDM uses is exactly invariant to any
invertible linear distortion applied uniformly to a subject's
covariances (`δ(AP₁Aᵀ, AP₂Aᵀ) = δ(P₁, P₂)`) — confirmed numerically
while building this. A synthetic domain shift built that way (one fixed
transform per subject) can *never* show a pooled baseline struggling,
so it's not a meaningful way to sanity-check whether alignment will
help. Trust the real `results/*_metrics.csv` numbers over any toy demo.

## Adaptive temporal filtering (DCD-RLS)

`src/adaptive_filtering.py` is a different approach entirely from
`src/domain_adaptation.py` above: instead of aligning spatial covariance
structure across subjects, it causally cancels EOG-correlated artifact
from each EEG channel per-trial via **Dichotomous Coordinate Descent
RLS** (Zakharov, White & Liu, IEEE Trans. Signal Process. 2008) — an
adaptive filter with no explicit matrix inversion anywhere, well suited
to constrained hardware. Two classifier front-ends on top of the cleaned
trials:

| Function | Front-end | Measured (real cohort) |
|---|---|---|
| `evaluate_loso_dcd_rls` | per-channel log-bandpower (no spatial filter) | **48.8%** — near chance |
| `evaluate_loso_dcd_rls_csp` | CSP (fit per-fold, train-only) → LDA | not yet run on real data |

```bash
python scripts/07_evaluate_dcd_rls.py                       # full cohort, both variants
python scripts/07_evaluate_dcd_rls.py --subjects 1,2,3,4,5   # quick partial run
python scripts/07_evaluate_dcd_rls.py --skip-logvar          # CSP variant only
```

**Why CSP was added:** per-channel log-bandpower alone hit only 48.8% on
the real cohort — near chance, close to every other variant tried so
far. CSP is specifically built to find the linear channel-combination
that maximizes between-class variance, which plain per-channel power
can miss entirely if the discriminative signal isn't aligned with any
single electrode. CSP is a *supervised* spatial filter, so — unlike
cleaning and log-bandpower, which are unsupervised and safe to compute
once for all trials — it's fit fresh inside every LOSO fold, on that
fold's training subjects' cleaned trials only, exactly like
`src/pipelines.py::build_csp_lda_pipeline` (never on pooled data,
Section 8 of the project blueprint).

**Verified, not just implemented (both the filter and the CSP
addition):** a known lagged, scaled artifact injected into a synthetic
signal was recovered to within a few percent of its true lag and gain,
with >99% MSE reduction vs. the true clean signal. The batched
multi-channel filter was checked bit-for-bit against 29 independent
single-channel filters. For CSP specifically: on synthetic data with a
genuinely spatially-distributed class signal (a trace-preserving power
swap along two random orthonormal directions — the textbook case CSP is
built for, chosen so per-channel variance carries only weak class
information), log-bandpower reached 88.7% while CSP+LDA reached 98.8%
on the *same* data — confirming the CSP integration captures spatial
structure that per-channel features miss, not just that it runs without
crashing. (Two earlier, easier synthetic attempts hit 100% ceiling for
*both* variants and had to be discarded as uninformative before finding
a version that actually discriminated between them.)

**Performance:** a naive pure-Python per-sample loop measured at ~2s per
real-size trial (29 channels, 2000 samples) — ~60 minutes for the full
2000-trial cohort. A numba `@njit` version of the same loop, verified to
produce numerically identical output, cut that to ~0.024s/trial (~50s
total, ~75x faster) — `numba` is now a dependency for exactly this
reason; `--no-numba` exists for debugging only.

**Scope note on the algorithm name:** this implements DCD-RLS's core
structure faithfully (auxiliary normal equations on the weight
increment, solved via bounded power-of-two coordinate descent, rank-1
recursive covariance update, no explicit inversion) but not the original
papers' exact bit-serial coordinate-cycling order or fixed-point
bit-width bookkeeping. Validate against the original paper or a
reference implementation before an actual embedded port — see
`src/adaptive_filtering.py`'s module docstring for the precise scope.

## Run order

1. **`01_Data_Ingestion_and_EDA.ipynb`** — pull the dataset, verify
   channel counts/reference/sampling rate, plot sensor layout and PSD.
2. **`02_Signal_Processing_and_ERD.ipynb`** — epoch the MI window,
   compute Morlet time-frequency maps, confirm contralateral
   mu/beta ERD over C3/C4.
3. **`03_Spatial_Filtering_CSP.ipynb`** — derive CSP from scratch
   (`src/feature_extraction.py`), cross-check against `mne.decoding.CSP`,
   inspect spatial patterns and log-variance feature separation.
4. **`04_Cross_Subject_Classification.ipynb`** — run `evaluate_loso`
   (`src/pipelines.py`) across all 50 subjects, compare CSP+LDA vs.
   Riemannian MDM, save the better pipeline to
   `models/trained_loso_pipeline.joblib`.
5. **`scripts/05_Simulated_Rehab_Interface.py`** — replay a held-out
   subject's recording through a 4.0 s / 250 ms sliding window and
   drive the Pygame neurofeedback GUI:
   ```bash
   python scripts/05_Simulated_Rehab_Interface.py --subject 7
   ```

## Anti-leakage guarantees

- CSP and covariance estimation are only ever `.fit()` inside a LOSO
  training fold (`src/pipelines.py::evaluate_loso`), never on the
  pooled dataset.
- `src/data_loader.py` performs no supervised fitting — it only loads,
  filters (fixed 8–30 Hz FIR, not a learned transform), and reshapes.
- `evaluate_loso`'s saved deployment model (fit on *all* subjects) is
  explicitly documented as unsuitable for reporting an accuracy number
  — every accuracy/AUC figure comes from the per-fold held-out results.
- Variance/log computations are clipped (`1e-10` floor) to avoid
  `-inf` from decoupled or flat channels.
- Riemannian covariance estimation uses OAS shrinkage rather than the
  empirical estimator to stay well-conditioned on low-voltage stroke
  recordings.

## Clinical / dataset realism notes

- 40 trials per patient (20 left, 20 right) is small by
  healthy-subject BCI standards — this pipeline deliberately avoids
  assumptions (e.g. abundant clean epochs, 64+ channel research rigs)
  that don't hold in an acute stroke ward.
- Acute-phase EEG carries lesion-dependent asymmetries, edema-driven
  conductivity changes, and blunted/delayed ERD — expect more
  between-subject variance than in healthy-cohort MI datasets, which
  is exactly what LOSO-CV is designed to expose rather than hide.

## Key references

- Liu, H. et al. "An EEG motor imagery dataset for brain computer
  interface in acute stroke patients." *Scientific Data* 11, 131 (2024).
- Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. "Optimal spatial
  filtering of single trial EEG during imagined hand movement." *IEEE
  Trans. Rehabil. Eng.* (2000). — canonical CSP derivation.
- Barachant, A. et al. "Multiclass brain-computer interface
  classification by Riemannian geometry." *IEEE Trans. Biomed. Eng.* (2012).
- Pfurtscheller, G. & Lopes da Silva, F.H. "Event-related EEG/MEG
  synchronization and desynchronization: basic principles." *Clin.
  Neurophysiol.* (1999).
