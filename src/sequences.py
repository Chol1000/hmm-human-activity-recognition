"""Build train/test session splits and synthetic continuous-activity composite
sequences (round-robin across activities) for HMM training and evaluation.

Raw recordings are isolated single-activity clips, so a composite sequence is
built by chaining sessions from different activities in rotation. This gives
the Baum-Welch step real activity-to-activity transitions to learn from,
while every window keeps its ground-truth activity label for evaluation.
"""
import numpy as np

ACTIVITIES = ["still", "standing", "walking", "jumping"]


def split_sessions(table, n_test_per_activity=2, seed=42):
    """Per-activity train/test split at the session level (not per-window), so
    overlapping windows from the same session never end up on both sides. Returns
    two {activity: [session_id, ...]} dicts."""
    rng = np.random.default_rng(seed)
    train, test = {}, {}
    for act in ACTIVITIES:
        sids = sorted(table.loc[table.activity == act, "session_id"].unique())
        sids = list(sids)
        rng.shuffle(sids)
        test[act] = sorted(sids[:n_test_per_activity])
        train[act] = sorted(sids[n_test_per_activity:])
    return train, test


def _session_windows(table, feature_cols, activity, session_id):
    """Feature matrix, true-label array, and (activity, session_id, window_idx) key
    list for one session's windows, in window_idx order."""
    sub = table[(table.session_id == session_id) & (table.activity == activity)].sort_values("window_idx")
    keys = list(zip(sub["activity"], sub["session_id"], sub["window_idx"]))
    return sub[feature_cols].values, sub["activity"].values, keys


def build_composite_sequences(table, feature_cols, sessions_by_activity, n_sequences, seed=0):
    """Chain sessions into n_sequences composite timelines, round-robin across
    activities (still, standing, walking, jumping, still, ...). Each returned dict
    has X (the concatenated features), labels (true activity per row), keys
    (activity/session_id/window_idx per row, for tracing back to raw windows), and
    session_order (which sessions got chained, in order)."""
    rng = np.random.default_rng(seed)
    activity_chunks = {}
    for act in ACTIVITIES:
        sids = list(sessions_by_activity[act])
        rng.shuffle(sids)
        activity_chunks[act] = [list(c) for c in np.array_split(sids, n_sequences)]

    sequences = []
    for seq_i in range(n_sequences):
        pool = {act: list(activity_chunks[act][seq_i]) for act in ACTIVITIES}
        order = []
        while any(pool[a] for a in ACTIVITIES):
            for a in ACTIVITIES:
                if pool[a]:
                    order.append((a, pool[a].pop(0)))

        X_parts, label_parts, key_parts, session_order = [], [], [], []
        for act, sid in order:
            X, labels, keys = _session_windows(table, feature_cols, act, sid)
            if len(X) == 0:
                continue
            X_parts.append(X)
            label_parts.append(labels)
            key_parts.extend(keys)
            session_order.append((act, sid, len(X)))
        sequences.append({
            "X": np.vstack(X_parts),
            "labels": np.concatenate(label_parts),
            "keys": key_parts,  # (activity, session_id, window_idx) per row of X, same order
            "session_order": session_order,
        })
    return sequences


def map_states_to_activities(model, train_sequences):
    """Majority-vote label each HMM state index using Viterbi decodes on training data."""
    K = model.n_states
    votes = {k: {a: 0 for a in ACTIVITIES} for k in range(K)}
    for seq in train_sequences:
        path, _ = model.decode(seq["X"])
        for state, true_label in zip(path, seq["labels"]):
            votes[state][true_label] += 1
    mapping = {k: max(votes[k], key=votes[k].get) for k in range(K)}
    return mapping, votes
