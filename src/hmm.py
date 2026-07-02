"""Gaussian-emission Hidden Markov Model implemented from scratch with numpy.

Supports multiple independent observation sequences for Baum-Welch (EM)
training, log-space forward-backward for numerical stability, a
log-likelihood convergence check, and Viterbi decoding.
"""
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.special import logsumexp

MIN_VAR = 1e-3


class GaussianHMM:
    """HMM with a diagonal-covariance Gaussian per hidden state. `fit` runs Baum-Welch
    and populates log_pi, log_A, means, vars (plus history_/n_iter_); `decode` runs
    Viterbi on a trained model."""

    def __init__(self, n_states, n_features, random_state=0):
        self.n_states = n_states
        self.n_features = n_features
        self.rng = np.random.default_rng(random_state)
        self.log_pi = None   # (K,)
        self.log_A = None    # (K,K)  log_A[i,j] = log P(state_t=j | state_{t-1}=i)
        self.means = None    # (K,D)
        self.vars = None     # (K,D) diagonal covariance

    def _init_params(self, X_list):
        """Seed pi, A, and the emission params from a K-means clustering of every
        observation pooled across X_list."""
        K, D = self.n_states, self.n_features
        X_all = np.vstack(X_list)
        # k-means init (not random point picks): gives well-separated, representative
        # starting centroids so Baum-Welch is far less likely to settle in a bad
        # local optimum (e.g. merging two activities into one state).
        seed = int(self.rng.integers(0, 2**31 - 1))
        centroids, labels = kmeans2(X_all, K, minit="++", seed=seed)
        self.means = centroids.copy()
        global_var = X_all.var(axis=0) + MIN_VAR
        self.vars = np.array([
            X_all[labels == k].var(axis=0) + MIN_VAR if np.sum(labels == k) > 1 else global_var
            for k in range(K)
        ])
        self.log_pi = np.log(np.full(K, 1.0 / K))
        A = np.full((K, K), 0.1 / (K - 1)) if K > 1 else np.ones((1, 1))
        np.fill_diagonal(A, 0.9)
        self.log_A = np.log(A)

    def _log_gauss_b(self, X):
        """log N(x_t | mu_k, var_k) for every timestep and state -> shape (T, n_states)."""
        T = len(X)
        K, D = self.n_states, self.n_features
        logB = np.zeros((T, K))
        for k in range(K):
            var = self.vars[k]
            diff = X - self.means[k]
            logB[:, k] = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum((diff ** 2) / var, axis=1)
        return logB

    def _forward_backward(self, logB):
        """Log-space forward-backward for one sequence's emission likelihoods logB.
        Returns log_alpha, log_beta, and the sequence log-likelihood."""
        T, K = logB.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = self.log_pi + logB[0]
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + self.log_A, axis=0) + logB[t]
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(self.log_A + logB[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
        loglik = logsumexp(log_alpha[-1])
        return log_alpha, log_beta, loglik

    def score(self, X):
        """Log-likelihood of one observation sequence X under the current model."""
        logB = self._log_gauss_b(X)
        _, _, loglik = self._forward_backward(logB)
        return loglik

    def fit(self, X_list, max_iter=100, tol=1e-4, verbose=False):
        """Baum-Welch, jointly over multiple independent sequences in X_list (E-step
        stats get summed across sequences before each M-step update). Stops once the
        total log-likelihood improves by less than `tol` between iterations, rather
        than always running to `max_iter`."""
        self._init_params(X_list)
        K, D = self.n_states, self.n_features
        history = []
        prev_loglik = -np.inf
        for it in range(max_iter):
            total_loglik = 0.0
            pi_acc = np.zeros(K)
            A_num = np.zeros((K, K))
            A_den = np.zeros(K)
            mean_num = np.zeros((K, D))
            var_num = np.zeros((K, D))
            gamma_den = np.zeros(K)

            for X in X_list:
                logB = self._log_gauss_b(X)
                log_alpha, log_beta, loglik = self._forward_backward(logB)
                total_loglik += loglik
                log_gamma = log_alpha + log_beta - loglik
                gamma = np.exp(log_gamma)

                pi_acc += gamma[0]

                T = len(X)
                if T > 1:
                    log_xi = (log_alpha[:-1][:, :, None] + self.log_A[None, :, :]
                              + logB[1:][:, None, :] + log_beta[1:][:, None, :] - loglik)
                    xi = np.exp(log_xi)
                    A_num += xi.sum(axis=0)
                    A_den += gamma[:-1].sum(axis=0)

                gamma_den += gamma.sum(axis=0)
                mean_num += gamma.T @ X
                for k in range(K):
                    diff = X - self.means[k]
                    var_num[k] += (gamma[:, k][:, None] * diff ** 2).sum(axis=0)

            # M-step
            self.log_pi = np.log(pi_acc / len(X_list) + 1e-12)
            with np.errstate(divide="ignore"):
                self.log_A = np.log(A_num / A_den[:, None] + 1e-12)
            self.means = mean_num / gamma_den[:, None]
            self.vars = np.maximum(var_num / gamma_den[:, None], MIN_VAR)

            history.append(total_loglik)
            if verbose:
                print(f"iter {it:3d}  loglik={total_loglik:.3f}")
            if abs(total_loglik - prev_loglik) < tol:
                break
            prev_loglik = total_loglik

        self.history_ = history
        self.n_iter_ = len(history)
        return self

    def decode(self, X):
        """Viterbi in log-space. Returns (best_state_path, log_prob_of_that_path)."""
        logB = self._log_gauss_b(X)
        T, K = logB.shape
        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = self.log_pi + logB[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + self.log_A
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + logB[t]
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, np.max(delta[-1])


def fit_with_restarts(n_states, n_features, X_list, n_restarts=8, max_iter=200, tol=1e-3,
                       base_seed=0, verbose=False):
    """EM is only guaranteed to find a local optimum, and a bad k-means seed can merge
    two activities into one state. Refitting from several seeds and keeping the run
    with the highest converged log-likelihood makes this far more robust."""
    best_model, best_ll = None, -np.inf
    for r in range(n_restarts):
        model = GaussianHMM(n_states, n_features, random_state=base_seed + r)
        model.fit(X_list, max_iter=max_iter, tol=tol, verbose=False)
        ll = model.history_[-1]
        if verbose:
            print(f"  restart {r}: final loglik={ll:.2f}  iters={model.n_iter_}")
        if ll > best_ll:
            best_ll, best_model = ll, model
    return best_model
