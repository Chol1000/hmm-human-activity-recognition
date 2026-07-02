"""Plotting helpers for the HMM activity-recognition pipeline. Every function
saves a PNG to outputs/figures/ and returns the path."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import ACTIVITIES, load_session

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 150, "axes.titleweight": "bold"})

ACT_COLORS = {"still": "#4C72B0", "standing": "#55A868", "walking": "#C44E52", "jumping": "#8172B2"}


def _savefig(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_raw_samples(data_dir, sessions_by_activity, out_dir):
    """Grid of raw accelerometer/gyroscope magnitude vs. time, one representative
    session per activity, to visually confirm the four classes are distinguishable
    before any feature engineering."""
    fig, axes = plt.subplots(len(ACTIVITIES), 2, figsize=(11, 9), sharex=False)
    for row, act in enumerate(ACTIVITIES):
        session_dir = os.path.join(data_dir, act, sessions_by_activity[act][0])
        df = load_session(session_dir)
        axes[row, 0].plot(df.t, df.a_mag, color=ACT_COLORS[act], lw=1)
        axes[row, 0].set_ylabel(f"{act}\n|accel| (m/s²)")
        axes[row, 1].plot(df.t, df.g_mag, color=ACT_COLORS[act], lw=1)
        axes[row, 1].set_ylabel("|gyro| (rad/s)")
        if row == 0:
            axes[row, 0].set_title("Accelerometer magnitude")
            axes[row, 1].set_title("Gyroscope magnitude")
        if row == len(ACTIVITIES) - 1:
            axes[row, 0].set_xlabel("time (s)")
            axes[row, 1].set_xlabel("time (s)")
    fig.suptitle("Sample raw sensor signals per activity (one session each)")
    return _savefig(fig, out_dir, "01_raw_signal_samples.png")


def plot_transition_heatmap(A, state_labels, out_dir):
    """Heatmap of a transition matrix A (rows=from-state, cols=to-state), annotated
    with each transition probability."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(A, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                xticklabels=state_labels, yticklabels=state_labels,
                cbar_kws={"label": "P(transition)"}, ax=ax, square=True)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title("Learned transition matrix A (Baum-Welch)")
    return _savefig(fig, out_dir, "02_transition_matrix_heatmap.png")


def plot_confusion_matrix(cm, labels, out_dir, name="03_confusion_matrix_test.png", title="Confusion matrix (test, unseen sessions)"):
    """Annotated confusion-matrix heatmap, rows=true activity, cols=predicted activity."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "# windows"}, ax=ax, square=True)
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    ax.set_title(title)
    return _savefig(fig, out_dir, name)


def plot_decoded_sequence(seq, mapping, out_dir, name="04_decoded_sequence_test.png", title="Decoded vs true activity sequence (test)"):
    """Step plot of the true activity label vs. the Viterbi-decoded activity over a
    composite sequence's window index, so predicted activity transitions are visible
    directly against ground truth. `seq` must include a '_model' key (the fitted
    GaussianHMM) alongside the usual 'X'/'labels' from `build_composite_sequences`."""
    path, _ = seq["_model"].decode(seq["X"])
    pred_labels = [mapping[p] for p in path]
    true_labels = list(seq["labels"])
    idx = np.arange(len(true_labels))
    act_to_y = {a: i for i, a in enumerate(ACTIVITIES)}

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.step(idx, [act_to_y[a] for a in true_labels], where="post", label="True", color="black", lw=2, alpha=0.7)
    ax.step(idx, [act_to_y[a] for a in pred_labels], where="post", label="Viterbi decoded", color="crimson", lw=1.5, linestyle="--")
    ax.set_yticks(range(len(ACTIVITIES)))
    ax.set_yticklabels(ACTIVITIES)
    ax.set_xlabel("window index (composite test sequence, 1s windows, 50% overlap)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return _savefig(fig, out_dir, name)


def plot_emission_summary(model, mapping, feature_table_train, feature_cols_raw, out_dir,
                            raw_features=("acc_mag_std", "gyro_mag_std", "acc_dom_freq")):
    """Bar chart of a few interpretable RAW features' mean value per learned state,
    computed from the windows Viterbi-assigned to that state (not the PCA space,
    which isn't directly interpretable)."""
    n_states = model.n_states
    fig, axes = plt.subplots(1, len(raw_features), figsize=(4 * len(raw_features), 4))
    state_order = sorted(range(n_states), key=lambda k: ACTIVITIES.index(mapping[k]))
    labels = [mapping[k] for k in state_order]
    for ax, feat in zip(axes, raw_features):
        means = [feature_table_train.loc[feature_table_train["_state"] == k, feat].mean() for k in state_order]
        stds = [feature_table_train.loc[feature_table_train["_state"] == k, feat].std() for k in state_order]
        ax.bar(labels, means, yerr=stds, color=[ACT_COLORS[l] for l in labels], capsize=4)
        ax.set_title(feat)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Emission characteristics per learned HMM state (mean ± std of raw features)")
    return _savefig(fig, out_dir, "05_emission_summary.png")


def plot_emission_means_heatmap(model, mapping, out_dir, name="08_emission_means_heatmap.png"):
    """Heatmap of the fitted Gaussian emission means B (states x PCA feature dims) --
    the direct, literal visualization of the emission parameters, complementing the
    more human-interpretable raw-feature summary in `plot_emission_summary`."""
    state_order = sorted(range(model.n_states), key=lambda k: ACTIVITIES.index(mapping[k]))
    state_labels = [mapping[k] for k in state_order]
    means_ordered = model.means[state_order]

    fig, ax = plt.subplots(figsize=(max(6, means_ordered.shape[1] * 0.55), 4))
    sns.heatmap(means_ordered, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                xticklabels=[f"pc_{i}" for i in range(means_ordered.shape[1])],
                yticklabels=state_labels, cbar_kws={"label": "standardized mean value"}, ax=ax)
    ax.set_xlabel("PCA feature dimension")
    ax.set_ylabel("HMM state (activity)")
    ax.set_title("Emission probabilities: Gaussian mean per state, per feature dimension")
    return _savefig(fig, out_dir, name)


def plot_loglik_curve(history, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history, marker="o", ms=3)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("log-likelihood")
    ax.set_title("Baum-Welch convergence (log-likelihood per iteration)")
    return _savefig(fig, out_dir, "06_baum_welch_convergence.png")
