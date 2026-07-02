"""Generate every figure required by the assignment into outputs/figures/.
Run from the project root: python3 src/make_figures.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_pipeline import main as run_pipeline, DATA_DIR
from data_utils import ACTIVITIES
import visualize as viz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    r = run_pipeline()
    model, mapping = r["model"], r["mapping"]
    table = r["table"]

    print("\n=== Generating figures ===")

    p1 = viz.plot_raw_samples(DATA_DIR, {**r["train_sessions"]}, FIG_DIR)
    print("saved", p1)

    state_order = sorted(range(model.n_states), key=lambda k: ACTIVITIES.index(mapping[k]))
    state_labels = [mapping[k] for k in state_order]
    A_full = np.exp(model.log_A)
    A_ordered = A_full[np.ix_(state_order, state_order)]
    p2 = viz.plot_transition_heatmap(A_ordered, state_labels, FIG_DIR)
    print("saved", p2)

    p3 = viz.plot_confusion_matrix(r["cm_test"], ACTIVITIES, FIG_DIR)
    print("saved", p3)

    test_seq = dict(r["test_sequences"][0])
    test_seq["_model"] = model
    p4 = viz.plot_decoded_sequence(test_seq, mapping, FIG_DIR)
    print("saved", p4)

    # tag raw training windows with their Viterbi-decoded state for the emission summary
    table = table.copy()
    table["_state"] = -1
    for seq in r["train_sequences"]:
        path, _ = model.decode(seq["X"])
        for (act, sid, w_idx), state in zip(seq["keys"], path):
            mask = (table.activity == act) & (table.session_id == sid) & (table.window_idx == w_idx)
            table.loc[mask, "_state"] = state
    train_tagged = table[table["_state"] >= 0]
    p5 = viz.plot_emission_summary(model, mapping, train_tagged, r["feature_cols"], FIG_DIR)
    print("saved", p5)

    p6 = viz.plot_loglik_curve(model.history_, FIG_DIR)
    print("saved", p6)

    p7 = viz.plot_confusion_matrix(
        r["cm_train"], ACTIVITIES, FIG_DIR, name="07_confusion_matrix_train.png",
        title="Confusion matrix (train, sanity check)"
    )
    print("saved", p7)

    p8 = viz.plot_emission_means_heatmap(model, mapping, FIG_DIR)
    print("saved", p8)

    print("\nAll figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
