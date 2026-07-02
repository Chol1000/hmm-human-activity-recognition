"""Per-class confusion matrix, sensitivity, specificity and overall accuracy."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def per_class_metrics(y_true, y_pred, labels):
    """One-vs-rest sensitivity, specificity and sample counts per activity, plus
    overall accuracy. Returns the confusion matrix, a results DataFrame formatted
    to match the table the assignment asks for, and the overall accuracy float."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    rows = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        n_samples = int(cm[i, :].sum())
        rows.append({
            "Activity": label,
            "Number of Samples": n_samples,
            "Sensitivity": round(sensitivity, 3),
            "Specificity": round(specificity, 3),
        })
    overall_acc = float(np.trace(cm) / total)
    df = pd.DataFrame(rows)
    df["Overall Accuracy"] = round(overall_acc, 3)
    return cm, df, overall_acc
