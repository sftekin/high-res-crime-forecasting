import os
import pickle as pkl
import itertools

import numpy as np
import pandas as pd

from configs.grid_config import GridConfig
from data_generators.grid_creator import GridCreator
from data_generators.data_creator import DataCreator
from batch_generators.batch_generator import BatchGenerator

from helpers.static_helper import get_save_dir, get_set_ids, get_set_end_date, calculate_metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from helpers.graph_helper import bin_pred, f1_score, accuracy_score, confusion_matrix


def run():
    config = GridConfig()
    grid_creator = GridCreator(data_params=config.data_params,
                               grid_params=config.grid_params)

    if not grid_creator.check_is_created():
        print(f"Data is not found in {grid_creator.grid_save_dir}. Starting data creation...")
        grid_creator.create_grid()
    else:
        print(f"Data is found.")

    spatial_res = config.grid_params["spatial_res"]
    coords = create_coord_grid(grid_creator.coord_range, config.grid_params["spatial_res"])
    coords = coords.mean(axis=2).reshape(-1, 2)

    # create save path
    model_name = "gmm"
    save_dir = get_save_dir(model_name=model_name)

    data_len = config.experiment_params["train_size"] + \
               config.experiment_params["val_size"] + config.experiment_params["test_size"]
    start_date = pd.to_datetime(grid_creator.start_date)
    end_date = pd.to_datetime(grid_creator.end_date)
    num_months = int((end_date - start_date) / np.timedelta64(1, 'M'))
    for i in range(0, int(num_months - data_len) + 1):
        stride_offset = pd.DateOffset(years=i)
        start_date = grid_creator.date_r[0] + stride_offset
        start_date_str = start_date.strftime("%Y-%m-%d")

        train_end_date = get_set_end_date(set_size=config.experiment_params["train_size"], start_date=start_date)
        val_end_date = get_set_end_date(set_size=config.experiment_params["val_size"], start_date=train_end_date)
        test_end_date = get_set_end_date(set_size=config.experiment_params["test_size"], start_date=val_end_date)

        train_ids = get_set_ids(grid_creator.date_r, start_date, train_end_date)
        val_ids = get_set_ids(grid_creator.date_r, train_end_date, val_end_date)
        test_ids = get_set_ids(grid_creator.date_r, val_end_date, test_end_date)

        win_in_len = 3
        config.batch_gen_params["window_in_len"] = win_in_len
        val_ids = np.concatenate([train_ids[-win_in_len:], val_ids])
        test_ids = np.concatenate([val_ids[-win_in_len:], test_ids])
        set_ids = [train_ids, val_ids, test_ids]

        # grid_crimes = [grid_creator.load_grid(c)[..., [2]] for c in grid_creator.crime_types]
        grid = grid_creator.load_grid(dataset_name="all")

        config.batch_gen_params["batch_size"] = 1
        generator = BatchGenerator(in_data=grid,
                                   labels=grid[..., [2]],
                                   set_ids=set_ids,
                                   batch_gen_params=config.batch_gen_params)

        for dataset_name in ["val", "test"]:
            labels = []
            predictons = []
            for i, (x, y) in enumerate(generator.generate(dataset_name=dataset_name)):
                print(i)
                in_data = x.squeeze()[..., :2].numpy().reshape(-1, 2)
                in_label = x.squeeze()[..., 2:].numpy().flatten()

                kernel = RBF()
                gp = GaussianProcessRegressor(kernel=kernel,
                                              n_restarts_optimizer=3)

                gp.fit(in_data, in_label)
                pred = gp.predict(coords).reshape(spatial_res)
                y = (y.squeeze().numpy() > 0).astype(int)

                labels.append(y)
                predictons.append(pred)

                print()

            labels = np.stack(labels)
            predictons = np.stack(predictons)

            pred = bin_pred(predictons.flatten(), labels.flatten())
            f1 = f1_score(labels.flatten(), pred)
            print(f"{dataset_name} Scores")
            print(f"F1: {f1}")

            tn, fn, fp, tp = confusion_matrix(y_true=labels.flatten(), y_pred=pred).flatten()
            acc = accuracy_score(y_true=labels.flatten(), y_pred=pred)

            print(f"Confusion Matrix = TN:{tn}, FN:{fn}, FP:{fp}, TP:{tp}")
            print(f"Accuracy = {acc:.4f}")


def create_coord_grid(coord_range, spatial_size):
    m, n = spatial_size
    x = np.linspace(coord_range[1][0], coord_range[1][1], n + 1)
    y = np.linspace(coord_range[0][0], coord_range[0][1], m + 1)

    coord_grid = np.zeros((m, n, 4, 2))
    for j in range(m):
        for i in range(n):
            coords = np.array(list(itertools.product(x[i:i + 2], y[j:j + 2])))
            coords_ordered = coords[[0, 1, 3, 2], :]
            coord_grid[m - j - 1, i, :] = coords_ordered

    return coord_grid


if __name__ == '__main__':
    run()
