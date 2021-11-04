import os
import warnings
import multiprocessing
import pickle as pkl

import numpy as np
import pandas as pd
from tqdm import tqdm
from models.arima import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from configs.stats_config import StatsConfig
from data_generators.grid_creator import GridCreator
from helpers.static_helper import calculate_metrics, f1_score, get_save_dir, get_set_ids, get_set_end_date, bin_pred


model_dispatcher = {
    "arima": ARIMA
}


def run():
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', UserWarning)

    config = StatsConfig()
    grid_creator = GridCreator(data_params=config.data_params,
                               grid_params=config.grid_params)
    num_process = grid_creator.num_process

    if not grid_creator.check_is_created():
        print(f"Data is not found in {grid_creator.grid_save_dir}. Starting data creation...")
        grid_creator.create_grid()
    else:
        print(f"Data is found.")

    model_name = "arima"
    model_params = config.model_params[model_name]

    # create save path
    save_dir = get_save_dir(model_name=model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_len = config.experiment_params["train_size"] + \
               config.experiment_params["val_size"] + config.experiment_params["test_size"]
    for i in range(0, int(12 - data_len) + 1):
        stride_offset = pd.DateOffset(months=i)
        start_date = grid_creator.date_r[0] + stride_offset
        start_date_str = start_date.strftime("%Y-%m-%d")

        train_end_date = get_set_end_date(set_size=config.experiment_params["train_size"], start_date=start_date)
        val_end_date = get_set_end_date(set_size=config.experiment_params["val_size"], start_date=train_end_date)
        test_end_date = get_set_end_date(set_size=config.experiment_params["test_size"], start_date=val_end_date)

        train_ids = get_set_ids(grid_creator.date_r, start_date, train_end_date)
        val_ids = get_set_ids(grid_creator.date_r, train_end_date, val_end_date)
        test_ids = get_set_ids(grid_creator.date_r, val_end_date, test_end_date)

        results_list = []
        scores_list = []
        for crime in grid_creator.crime_types:
            grid = grid_creator.load_grid(dataset_name=crime)

            time_len, height, width, feat_count = grid.shape
            grid_flatten = grid.reshape((-1, height * width, feat_count))
            train_ts = grid_flatten[train_ids]
            val_ts = grid_flatten[val_ids]
            test_ts = grid_flatten[test_ids]

            target_idx = 2
            labels = np.concatenate([train_ts, val_ts, test_ts, grid_flatten[[test_ids[-1] + 1]]], axis=0)
            labels = (labels[1:, :, target_idx] > 0).astype(int)  # shift labels 1 step to the right

            arg_list = []
            for i in range(height * width):
                sets = train_ts[:, i, target_idx], val_ts[:, i, target_idx], test_ts[:, i, target_idx]
                model = model_dispatcher[model_name](**model_params)
                arg_list.append((model, sets))

            with multiprocessing.Pool(processes=num_process) as pool:
                preds = list(tqdm(pool.imap(fit_transform, arg_list), total=len(arg_list)))
                preds = np.stack(preds, axis=1)

            set_sizes = [len(s) for s in [train_ts, val_ts, test_ts]]
            result = get_result_dict(preds, labels, set_sizes)
            pred_cp_path = os.path.join(save_dir, f"{start_date_str}_{crime}_results.pkl")
            with open(pred_cp_path, "wb") as f:
                pkl.dump(result, f)

            scores_cp_path = os.path.join(save_dir, f"{start_date_str}_{crime}_scores.pkl")
            scores = get_scores(result)
            with open(scores_cp_path, "wb") as f:
                pkl.dump(scores, f)

            results_list.append(result)
            scores_list.append(scores)

            print(f"Train Scores for the {crime}: F1: {scores['train']:.5f}")
            print(f"Val Scores for the {crime}: F1: {scores['val']:.5f}")
            print(f"Test Scores for the {crime}: F1: {scores['test']:.5f}")

        results_save_path = os.path.join(save_dir, f"{start_date_str}_results.pkl")
        with open(results_save_path, "wb") as f:
            pkl.dump(results_list, f)

        scores_save_path = os.path.join(save_dir, f"{start_date_str}_scores.pkl")
        with open(scores_save_path, "wb") as f:
            pkl.dump(scores_list, f)


def fit_transform(arg_list):
    model, sets = arg_list
    train_ts, val_ts, test_ts = sets
    train_val_ts = np.concatenate([train_ts, val_ts])

    model.fit(endog=train_ts)
    train_pred = model.fitted_values()
    val_pred = model.predict(endog=val_ts)

    model.fit(endog=train_val_ts)
    test_pred = model.predict(endog=test_ts)

    predictions = np.concatenate([train_pred, val_pred, test_pred])
    return predictions


def get_result_dict(preds, labels, set_sizes):
    train_size, val_size, test_size = set_sizes
    result_dict = {
        "train": [preds[:train_size], labels[:train_size]],
        "val": [preds[train_size:train_size+val_size], labels[train_size:train_size+val_size]],
        "test": [preds[train_size+val_size:], labels[train_size+val_size:]]
    }
    return result_dict


def get_scores(result_dict):
    score_dict = {}
    for key, val in result_dict.items():
        pred, label = val
        pred = bin_pred(pred.flatten(), label.flatten())
        f1 = f1_score(y_true=label.flatten(), y_pred=pred)
        score_dict[key] = f1

    return score_dict


if __name__ == '__main__':
    run()
