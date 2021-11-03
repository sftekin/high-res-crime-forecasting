import os
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix, roc_curve


def get_save_dir(model_name):
    results_dir = "results"
    save_dir = os.path.join(results_dir, model_name)
    num_exp_dir = len(glob.glob(os.path.join(save_dir, 'exp_*')))
    save_dir = os.path.join(save_dir, "exp_" + str(num_exp_dir + 1))
    return save_dir


def calculate_metrics(pred, label):
    target_count = pred.shape[-1]
    # multi label
    pred, label = pred.reshape(-1, target_count), label.reshape(-1, target_count)
    # find best threshold
    for i in range(target_count):
        pred[:, i] = bin_pred(pred[:, i], label[:, i])
    if target_count == 1:
        f1 = f1_score(label, pred)
        metrics = {"f1": f1}
    else:
        micro_f1 = f1_score(label, pred, average="micro")
        macro_f1 = f1_score(label, pred, average="macro")

        metrics = {
            "f1_micro": micro_f1,
            "f1_macro": macro_f1
        }

    return metrics


def bin_pred(pred, label):
    fpr, tpr, thresholds = roc_curve(y_true=label, y_score=pred)
    opt_thr = thresholds[np.argmax(tpr - fpr)]
    pred = (pred >= opt_thr).astype(int)
    return pred


def get_set_end_date(set_size, start_date):
    months = int(set_size)
    days = 30 * (set_size - int(set_size))
    date_offset = pd.DateOffset(months=months, days=int(days))
    end_date = start_date + date_offset
    return end_date


def get_set_ids(date_r, start_date, end_date):
    ids = np.argwhere((date_r > start_date) & (date_r < end_date)).squeeze()
    return ids

