import os
import glob

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix


def get_save_dir(model_name):
    results_dir = "results"
    save_dir = os.path.join(results_dir, model_name)
    num_exp_dir = len(glob.glob(os.path.join(save_dir, 'exp_*')))
    save_dir = os.path.join(save_dir, "exp_" + str(num_exp_dir + 1))
    return save_dir


def calculate_metrics(pred, label):
    pred, label = pred.flatten(), label.flatten()
    ap = average_precision_score(y_true=label, y_score=pred)

    thresholds = np.linspace(pred.min(), pred.max(), 100)
    f1_list = []
    for thr in thresholds:
        bin_pred = (pred >= thr).astype(int)
        f1_list.append(f1_score(label, bin_pred))
    f1_arr = np.array(f1_list)
    best_threshold = thresholds[np.argmax(f1_arr)]
    best_f1 = np.max(f1_arr)

    bin_pred = (pred >= best_threshold).astype(int)
    tn, fn, fp, tp = confusion_matrix(y_true=label, y_pred=bin_pred).flatten()

    metrics = {
        "AP": ap,
        "f1": best_f1,
        "tn": tn,
        "fn": fn,
        "fp": fp,
        "tp": tp
    }

    return metrics
