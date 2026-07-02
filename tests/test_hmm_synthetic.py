"""Sanity-check the from-scratch GaussianHMM on synthetic data with known ground truth."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hmm import GaussianHMM

rng = np.random.default_rng(42)
K, D = 3, 2
true_pi = np.array([0.5, 0.3, 0.2])
true_A = np.array([[0.85, 0.10, 0.05],
                    [0.05, 0.85, 0.10],
                    [0.10, 0.05, 0.85]])
true_means = np.array([[0, 0], [6, 6], [-6, 6]])
true_vars = np.array([[1, 1], [1, 1], [1, 1]])


def sample_sequence(T):
    states = np.zeros(T, dtype=int)
    states[0] = rng.choice(K, p=true_pi)
    for t in range(1, T):
        states[t] = rng.choice(K, p=true_A[states[t - 1]])
    X = np.array([rng.normal(true_means[s], np.sqrt(true_vars[s])) for s in states])
    return X, states


X_list, states_list = [], []
for _ in range(20):
    X, s = sample_sequence(150)
    X_list.append(X)
    states_list.append(s)

model = GaussianHMM(n_states=K, n_features=D, random_state=1)
model.fit(X_list, max_iter=100, tol=1e-4, verbose=False)

print("log-likelihood history (should be non-decreasing):")
print([round(h, 1) for h in model.history_])
diffs = np.diff(model.history_)
print("monotonic non-decreasing:", np.all(diffs >= -1e-6))
print("converged in", model.n_iter_, "iterations")

# decode accuracy (allowing for label permutation since EM is unsupervised)
from itertools import permutations
total_correct, total_n = 0, 0
for X, true_s in zip(X_list, states_list):
    pred, _ = model.decode(X)
    best_acc = 0
    for perm in permutations(range(K)):
        mapped = np.array(perm)[pred]
        acc = (mapped == true_s).mean()
        best_acc = max(best_acc, acc)
    total_correct += best_acc * len(true_s)
    total_n += len(true_s)

print("best-permutation Viterbi decode accuracy:", total_correct / total_n)
