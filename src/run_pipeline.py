"""End-to-end pipeline: features -> PCA -> Baum-Welch training -> Viterbi
evaluation on held-out sessions. Saves trained model + scaler/PCA + metrics
to outputs/{models,results}. Run from the project root: python3 src/run_pipeline.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import build_feature_table, ACTIVITIES
from hmm import GaussianHMM, fit_with_restarts
from sequences import split_sessions, build_composite_sequences, map_states_to_activities
from evaluate import per_class_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "outputs", "models")
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOW_SEC = 1.0
OVERLAP = 0.5
PCA_VARIANCE_TARGET = 0.95
N_TRAIN_SEQUENCES = 6
N_TEST_SEQUENCES = 2
N_TEST_SESSIONS_PER_ACTIVITY = 2
RANDOM_SEED = 42


def main():
    print("=== 1. Feature extraction ===")
    table, feature_cols, window_size, step = build_feature_table(DATA_DIR, WINDOW_SEC, OVERLAP)
    print(f"windows={len(table)}  window_size={window_size} samples  step={step} samples  n_features={len(feature_cols)}")
    print(table.groupby("activity").size().to_string())

    print("\n=== 2. Session-level train/test split ===")
    train_sessions, test_sessions = split_sessions(table, N_TEST_SESSIONS_PER_ACTIVITY, seed=RANDOM_SEED)
    for act in ACTIVITIES:
        print(f"{act}: train={len(train_sessions[act])} sessions, test={test_sessions[act]}")

    train_mask = table.apply(lambda r: r.session_id in train_sessions[r.activity], axis=1)
    train_table = table[train_mask].reset_index(drop=True)

    print("\n=== 3. Standardize + PCA (fit on TRAIN windows only) ===")
    scaler = StandardScaler().fit(train_table[feature_cols].values)
    X_train_scaled = scaler.transform(train_table[feature_cols].values)
    pca_full = PCA(random_state=RANDOM_SEED).fit(X_train_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum_var, PCA_VARIANCE_TARGET) + 1)
    print(f"components needed for >={PCA_VARIANCE_TARGET:.0%} variance: {n_components} / {len(feature_cols)}")
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED).fit(X_train_scaled)

    pc_cols = [f"pc_{i}" for i in range(n_components)]
    scaled_all = scaler.transform(table[feature_cols].values)
    pcs_all = pca.transform(scaled_all)
    table_pca = table[["activity", "session_id", "window_idx"]].copy()
    for i, c in enumerate(pc_cols):
        table_pca[c] = pcs_all[:, i]

    print("\n=== 4. Build composite training/testing sequences ===")
    train_sequences = build_composite_sequences(table_pca, pc_cols, train_sessions, N_TRAIN_SEQUENCES, seed=1)
    test_sequences = build_composite_sequences(table_pca, pc_cols, test_sessions, N_TEST_SEQUENCES, seed=2)
    print(f"train composite sequences: {len(train_sequences)}, lengths={[len(s['X']) for s in train_sequences]}")
    print(f"test composite sequences: {len(test_sequences)}, lengths={[len(s['X']) for s in test_sequences]}")

    print("\n=== 5. Train Gaussian HMM with Baum-Welch (log-likelihood convergence, multi-restart) ===")
    model = fit_with_restarts(
        n_states=len(ACTIVITIES), n_features=n_components,
        X_list=[s["X"] for s in train_sequences],
        n_restarts=10, max_iter=200, tol=1e-3, base_seed=7, verbose=True)
    print(f"best run: converged after {model.n_iter_} iterations, final loglik={model.history_[-1]:.2f}")

    print("\n=== 6. Map HMM states -> activities (majority vote on training decode) ===")
    mapping, votes = map_states_to_activities(model, train_sequences)
    for k, act in mapping.items():
        print(f"state {k} -> {act}   votes={votes[k]}")
    is_bijection = len(set(mapping.values())) == len(ACTIVITIES)
    print("bijective state<->activity mapping:", is_bijection)

    def decode_sequences(seqs):
        y_true, y_pred = [], []
        for s in seqs:
            path, _ = model.decode(s["X"])
            y_true.extend(s["labels"])
            y_pred.extend([mapping[p] for p in path])
        return np.array(y_true), np.array(y_pred)

    print("\n=== 7. Evaluate on TRAIN (sanity check) ===")
    yt_train, yp_train = decode_sequences(train_sequences)
    cm_train, train_df, train_acc = per_class_metrics(yt_train, yp_train, ACTIVITIES)
    print(train_df.to_string(index=False))
    print("train overall accuracy:", round(train_acc, 4))

    print("\n=== 8. Evaluate on TEST (held-out, unseen sessions) ===")
    yt_test, yp_test = decode_sequences(test_sequences)
    cm_test, test_df, test_acc = per_class_metrics(yt_test, yp_test, ACTIVITIES)
    print(test_df.to_string(index=False))
    print("test overall accuracy:", round(test_acc, 4))
    print("confusion matrix (rows=true, cols=pred), order=", ACTIVITIES)
    print(cm_test)

    print("\n=== 9. Save model, transformers, and metrics ===")
    np.savez(os.path.join(MODELS_DIR, "hmm_model.npz"),
             log_pi=model.log_pi, log_A=model.log_A, means=model.means, vars=model.vars,
             n_states=model.n_states, n_features=model.n_features)
    with open(os.path.join(MODELS_DIR, "state_mapping.pkl"), "wb") as f:
        pickle.dump(mapping, f)
    with open(os.path.join(MODELS_DIR, "scaler_pca.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca, "feature_cols": feature_cols, "pc_cols": pc_cols}, f)

    test_df.to_csv(os.path.join(RESULTS_DIR, "evaluation_metrics_test.csv"), index=False)
    train_df.to_csv(os.path.join(RESULTS_DIR, "evaluation_metrics_train.csv"), index=False)
    pd.DataFrame(cm_test, index=[f"true_{a}" for a in ACTIVITIES], columns=[f"pred_{a}" for a in ACTIVITIES]).to_csv(
        os.path.join(RESULTS_DIR, "confusion_matrix_test.csv"))
    A = np.exp(model.log_A)
    ordered = [k for k in range(len(ACTIVITIES))]  # state index order as-is; mapping kept separately
    pd.DataFrame(A, index=[f"state_{k}({mapping[k]})" for k in range(model.n_states)],
                 columns=[f"state_{k}({mapping[k]})" for k in range(model.n_states)]).to_csv(
        os.path.join(RESULTS_DIR, "transition_matrix.csv"))
    pd.Series(model.history_, name="loglik").to_csv(os.path.join(RESULTS_DIR, "loglik_history.csv"), index_label="iter")
    table.to_csv(os.path.join(RESULTS_DIR, "feature_table_raw.csv"), index=False)

    pd.DataFrame({"state": list(mapping.keys()), "activity": list(mapping.values())}).to_csv(
        os.path.join(RESULTS_DIR, "state_mapping.csv"), index=False)

    print("Saved model -> outputs/models/, metrics -> outputs/results/")
    return {
        "table": table, "feature_cols": feature_cols, "pc_cols": pc_cols,
        "scaler": scaler, "pca": pca, "model": model, "mapping": mapping,
        "train_sessions": train_sessions, "test_sessions": test_sessions,
        "train_sequences": train_sequences, "test_sequences": test_sequences,
        "cm_test": cm_test, "cm_train": cm_train, "test_df": test_df, "train_df": train_df,
        "test_acc": test_acc, "train_acc": train_acc,
        "yt_test": yt_test, "yp_test": yp_test,
        "yt_train": yt_train, "yp_train": yp_train,
    }


if __name__ == "__main__":
    main()
