# Human Activity Recognition with a Hidden Markov Model

Formative 2 (Hidden Markov Models) — infers four activity states (still, standing, walking,
jumping) from smartphone accelerometer/gyroscope data, using a Gaussian-emission HMM trained
with Baum-Welch and decoded with Viterbi, both implemented from scratch in NumPy.

## Results at a glance

- 52 recordings collected with Sensor Logger (iPhone 13 Pro Max, 100 Hz), all within the required 5-10s window
- 822 windows extracted (1s, 50% overlap), 32 time- and frequency-domain features per window
- Baum-Welch trained on 6 composite sequences built from 44 training sessions, converges in a handful of iterations under a real log-likelihood convergence check
- Evaluated on 8 held-out sessions never seen during training: 100% accuracy, sensitivity and specificity of 1.0 on every activity
- Result verified with a session-level leakage check, cross-validation across 5 independent train/test splits, and a label-shuffle sanity check (all in the notebook)

## Repository layout

```
data/                 52 labelled recording sessions, one folder per activity
src/                  the pipeline, as importable modules
  data_utils.py          load sessions, window them, extract features
  hmm.py                  GaussianHMM: Baum-Welch training, Viterbi decoding
  sequences.py            train/test session split, composite sequence construction
  evaluate.py             sensitivity/specificity/accuracy/confusion matrix
  visualize.py             all figure-generating functions
  run_pipeline.py         end-to-end script: data -> features -> model -> metrics
  make_figures.py         generates every figure into outputs/figures/
tests/
  test_hmm_synthetic.py  validates the HMM against synthetic data with known ground truth
notebooks/
  HMM_Activity_Recognition.ipynb   the full, executed walkthrough
outputs/
  models/                trained model + scaler/PCA, saved
  results/               metrics and tables as CSV
  figures/               every plot, as PNG
reports/
  report.md / .pdf       the short report
```

## Running it

```
pip install -r requirements.txt
python3 tests/test_hmm_synthetic.py     # sanity-checks the HMM math on synthetic data
python3 src/run_pipeline.py             # runs the full pipeline, prints metrics
python3 src/make_figures.py             # regenerates every figure in outputs/figures/
jupyter nbconvert --to notebook --execute --inplace notebooks/HMM_Activity_Recognition.ipynb
```

## Notes on the approach

Raw recordings are isolated single-activity clips, so composite sequences are built by chaining
sessions from different activities in rotation — this gives Baum-Welch real transitions to learn
from while keeping every window's ground-truth label for evaluation. This construction, and its
implications for how literally to read the learned transition matrix, is discussed in the report
and in the notebook.
