import os
import pickle as pkl
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from stats_config import StatsConfig
from data_generators.grid_creator import GridCreator
from models.arima import ARIMA
from sklearn.metrics import f1_score, average_precision_score
from concurrent.futures import ThreadPoolExecutor

from helpers.static_helper import calculate_metrics, get_save_dir

model_dispatcher = {
    "arima": ARIMA
}


def run():
    warnings.simplefilter('ignore', ConvergenceWarning)

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

    data_len = config.batch_gen_params["train_size"] + \
               config.batch_gen_params["val_size"] + config.batch_gen_params["test_size"]
    for i in range(0, int(12 - data_len) + 1):
        stride_offset = pd.DateOffset(months=i)
        start_date = grid_creator.date_r[0] + stride_offset
        train_end_date = get_set_end_date(set_size=config.batch_gen_params["train_size"], start_date=start_date)
        val_end_date = get_set_end_date(set_size=config.batch_gen_params["val_size"], start_date=train_end_date)
        test_end_date = get_set_end_date(set_size=config.batch_gen_params["test_size"], start_date=val_end_date)

        train_ids = get_set_ids(grid_creator.date_r, start_date, train_end_date)
        val_ids = get_set_ids(grid_creator.date_r, train_end_date, val_end_date)
        test_ids = get_set_ids(grid_creator.date_r, val_end_date, test_end_date)

        y_true = []
        y_pred = []
        for crime in grid_creator.crime_types:
            grid_paths = grid_creator.get_paths(dataset_name=crime)
            grid = []
            for path in grid_paths:
                with open(path, "rb") as f:
                    grid.append(np.load(f))
            grid = np.stack(grid)

            time_len, height, width, feat_count = grid.shape
            grid_flatten = grid.reshape((-1, height * width, feat_count))
            train_ts = grid_flatten[train_ids]
            val_ts = grid_flatten[val_ids]
            test_ts = grid_flatten[test_ids]

            arg_list = []
            target_idx = 2
            for i in range(height * width):
                sets = train_ts[:-1, i, target_idx], val_ts[:-1, i, target_idx], test_ts[:-1, i, target_idx]
                model = model_dispatcher[model_name](**model_params)
                arg_list.append((model, sets))
            with ThreadPoolExecutor(max_workers=num_process) as executor:
                preds = list(tqdm(executor.map(lambda p: fit_transform(*p), arg_list), total=len(arg_list)))
                preds = np.stack(preds, axis=1)

            labels = (grid_flatten[1:, :, target_idx] > 0).astype(int)
            y_true.append(labels)
            y_pred.append(preds)

        y_true = np.stack(y_true, axis=-1)
        y_pred = np.stack(y_true, axis=-1)

        # create save path
        save_dir = get_save_dir(model_name=model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        start_date_str = start_date.strftime("%Y-%m-%d")
        true_save_path = os.path.join(save_dir, f"{start_date_str}_y_true.npy")
        pred_save_path = os.path.join(save_dir, f"{start_date_str}_y_pred.npy")
        np.save(true_save_path, y_true)
        np.save(pred_save_path, y_pred)


def fit_transform(model, sets):
    train_ts, val_ts, test_ts = sets
    train_val_ts = np.concatenate([train_ts, val_ts])

    model.fit(endog=train_ts)
    train_pred = model.fitted_values()
    val_pred = model.predict(endog=val_ts)

    model.fit(endog=train_val_ts)
    train_val_pred = model.fitted_values()
    test_pred = model.predict(endog=test_ts)

    predictions = np.concatenate([train_pred, val_pred, train_val_pred, test_pred])
    return predictions


def get_set_end_date(set_size, start_date):
    months = int(set_size)
    days = 30 * (set_size - int(set_size))
    date_offset = pd.DateOffset(months=months, days=int(days))
    end_date = start_date + date_offset
    return end_date


def get_set_ids(date_r, start_date, end_date):
    ids = np.argwhere((date_r > start_date) & (date_r < end_date)).squeeze()
    return ids


if __name__ == '__main__':
    run()
