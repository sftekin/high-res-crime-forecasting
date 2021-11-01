import os
import pickle as pkl
import numpy as np

from stats_config import StatsConfig
from data_generators.grid_creator import GridCreator
from models.arima import ARIMA
from sklearn.metrics import f1_score, average_precision_score

from helpers.static_helper import calculate_metrics, get_save_dir

model_dispatcher = {
    "arima": ARIMA
}


def run():
    config = StatsConfig()
    grid_creator = GridCreator(data_params=config.data_params,
                               grid_params=config.grid_params)

    if not grid_creator.check_is_created():
        print(f"Data is not found in {grid_creator.grid_save_dir}. Starting data creation...")
        grid_creator.create_grid()

    print(f"Data is found.")
    for crime in grid_creator.crime_types:
        grid_paths = grid_creator.get_paths(dataset_name=crime)
        grid = []
        for path in grid_paths:
            with open(path, "rb") as f:
                grid.append(np.load(f))
        grid = np.stack(grid)

    # todo: train val test size

    test_size = config.batch_gen_params["test_size"]
    val_ratio = config.batch_gen_params["val_ratio"]
    train_val_size = len(grid) - test_size
    val_size = int(train_val_size * val_ratio)
    train_size = train_val_size - val_size

    model_name = "arima"
    model_params = config.model_params[model_name]
    model = model_dispatcher[model_name](**model_params)

    # create save path
    save_dir = get_save_dir(model_name=model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    time_len, height, width, feat_count = grid.shape
    grid_flatten = grid.reshape(-1, height * width, feat_count)
    train_ts = grid_flatten[:train_size]
    val_ts = grid_flatten[train_size:train_val_size]
    train_val_ts = grid_flatten[:train_val_size]
    test_ts = grid_flatten[train_val_size:]

    target_idx = 2
    exog_idx = list([i for i in range(feat_count) if i != target_idx])

    train_pred, val_pred, train_val_pred, test_pred = [], [], [], []
    for i in range(height * width):
        print(f"{i/(height*width) * 100:.2f}%")
        train_endog = train_ts[:, i, target_idx]
        train_exog = train_ts[:, i, exog_idx]
        train_pred.append(fit(model, train_endog, train_exog, time_len))

        val_endog = val_ts[:, i, target_idx]
        val_exog = val_ts[:, i, exog_idx]
        val_pred.append(predict(model, val_endog, val_exog, time_len))

        train_val_endog = train_val_ts[:, i, target_idx]
        train_val_exog = train_val_ts[:, i, exog_idx]
        train_val_pred.append(fit(model, train_val_endog, train_val_exog, time_len))

        test_endog = test_ts[:, i, target_idx]
        test_exog = test_ts[:, i, exog_idx]
        test_pred.append(predict(model, test_endog, test_exog, time_len))

    predictions = [train_pred, val_pred, train_val_pred, test_pred]
    labels = [ts[..., target_idx] for ts in [train_ts, val_ts, train_val_ts, test_ts]]
    with open("preds.pkl", "wb") as f:
        pkl.dump(predictions, f)

    with open("labels.pkl", "wb") as f:
        pkl.dump(labels, f)

    results = {
        "train": None,
        "val": None,
        "train_val": None,
        "test": None
    }
    for i, key in enumerate(results.keys()):
        pred_grid = np.stack(predictions[i], axis=1)
        label = (labels[i] > 0).astype(int)
        results[key] = calculate_metrics(pred=pred_grid[:-1], label=label[1:])

    print(results)

    save_path = os.path.join(save_dir, "results.pkl")
    with open(save_path, "wb") as f:
        pkl.dump(results, f)


def fit(model, endog, exog, time_len):
    model.fit(endog=endog, exog=exog)
    return model.fitted_values()


def predict(model, endog, exog, time_len):
    return model.predict(endog=endog, exog=exog)


if __name__ == '__main__':
    run()
