# Modeling Human Activity States Using Hidden Markov Models

Formative 2 — Hidden Markov Models | Chol Monykuch

This is a summary of the project. The full formatted report is `Formative2_HMM_Report.pdf` in this
folder; the full implementation and analysis is in `notebooks/HMM_Activity_Recognition.ipynb`.

## Background

Wearable and smartphone sensors stream noisy accelerometer/gyroscope signals, but the activity
that produced them is hidden and must be inferred. This project uses that idea for single-device
activity recognition (still, standing, walking, jumping) — the same problem behind fall-detection
alerts, fitness-tracker workout segmentation, and smart-home occupancy sensing. An HMM fits well
because activity is sequential and persistent: a person walking now is far more likely to still be
walking a second later than to have jumped state, which is exactly what a transition matrix
captures.

## Data Collection

Recorded with Sensor Logger (iPhone 13 Pro Max) at a fixed 100 Hz, confirmed identical across all
52 sessions. 13 sessions per activity, each 5-10s, totaling 111-113s per activity — clears the
assignment's 50-file and 90-second minimums.

## Feature Extraction

Each session is windowed (1s, 100 samples, 50% overlap) and 32 features are extracted per window:
time-domain (mean, std, SMA, inter-axis correlation, RMS) and frequency-domain (dominant FFT
frequency, spectral energy, spectral entropy), computed over the accelerometer, gyroscope, and
their 3D magnitudes. Features are Z-score standardized (fit on training data only), then reduced
via PCA to the components retaining 95% of training variance, for numerical stability given the
limited training set.

## HMM Setup and Implementation

Hidden states are the four activities; observations are the PCA-reduced feature vectors; the
transition matrix, emission distributions, and initial-state distribution are all learned via
Baum-Welch. The model is implemented from scratch in NumPy/SciPy (`src/hmm.py`) — log-space
forward-backward, Baum-Welch with K-means++ initialization and a genuine log-likelihood
convergence check, and Viterbi decoding — with no external HMM library. It was validated on
synthetic data with known ground truth before being applied to real data (100% recovery accuracy).

Each recording is an isolated single-activity clip, so training/test sessions are split at the
session level (8 sessions held out, 44 used for training), then chained round-robin into composite
sequences so Baum-Welch has real activity transitions to learn from.

## Results

Training converges in ~5 iterations under the log-likelihood convergence check. Evaluated on 8
held-out sessions never seen during training (125 windows):

| Activity | Samples | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| Still | 32 | 1.000 | 1.000 | 1.000 |
| Standing | 31 | 1.000 | 1.000 | 1.000 |
| Walking | 32 | 1.000 | 1.000 | 1.000 |
| Jumping | 30 | 1.000 | 1.000 | 1.000 |

This is a genuinely held-out result (session-level split, transformers fit on training data only),
verified further in the notebook via a leakage check, 5-split cross-validation, and a label-shuffle
sanity test. It reflects a comparatively easy separation task — the four activities were recorded
as deliberately distinct, controlled motions with large gaps in movement intensity (jumping's
signal variance is ~80x still's) — not evidence the model would generalize equally well to
ambiguous, continuously-monitored real-world movement.

## Discussion

Jumping was easiest to classify (its motion intensity is an order of magnitude larger than any
other activity); still vs. standing was hardest, both near-zero-motion states differing only in
subtle micro-movement. Because recordings are isolated single-activity clips, the learned
transition matrix mostly reflects the constructed training-sequence ordering rather than naturally
observed behavior — a dataset limitation, not a model flaw. At 100 Hz, frequency-domain features
help distinguish walking/jumping but are mostly noise for still/standing, where the model instead
relies on time-domain intensity features. Future improvements: genuine continuous multi-activity
recordings, a barometer channel to help separate vertical from horizontal motion, and adaptive
window sizes to sharpen the still/standing boundary.

## Conclusion

A from-scratch Gaussian HMM, trained with Baum-Welch and decoded with Viterbi, recovers all four
activities with high fidelity on entirely unseen recording sessions, converges via a genuine
convergence check, and learns a physically sensible structure — matching real
human-activity-recognition intuition rather than acting as an opaque black box.

This was completed individually — data collection, implementation, evaluation, and this report are
all my own work, tracked through incremental commits.
