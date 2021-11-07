import os
import pickle as pkl

import numpy as np
import pandas as pd

from configs.grid_config import GridConfig
from data_generators.grid_creator import GridCreator
from batch_generators.batch_generator import BatchGenerator
from models.convlstm import ConvLSTM
from models.convlstm_one_block import ConvLSTMOneBlock
from trainer import Trainer
from helpers.static_helper import get_save_dir, get_set_ids, get_set_end_date, calculate_metrics

model_dispatcher = {
    "convlstm": ConvLSTM,
    "convlstm_one_block": ConvLSTMOneBlock
}


def run():
    config = GridConfig()
    grid_creator = GridCreator(data_params=config.data_params,
                               grid_params=config.grid_params)

    if not grid_creator.check_is_created():
        print(f"Data is not found in {grid_creator.grid_save_dir}. Starting data creation...")
        grid_creator.create_grid()
    else:
        print(f"Data is found.")

    # create save path
    model_name = "convlstm"
    save_dir = get_save_dir(model_name=model_name)

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

        win_in_len = config.batch_gen_params["window_in_len"]
        val_ids = np.concatenate([train_ids[-win_in_len:], val_ids])
        test_ids = np.concatenate([val_ids[-win_in_len:], test_ids])
        set_ids = [train_ids, val_ids, test_ids]

        # grid_crimes = [grid_creator.load_grid(c)[..., [2]] for c in grid_creator.crime_types]
        grid = grid_creator.load_grid(dataset_name="THEFT")

        if config.trainer_params["loss_function"] == "BCE":
            labels = (grid[..., [2]] > 0).astype(int)
        else:
            labels = grid[..., [2]]

        generator = BatchGenerator(in_data=grid[..., [2]],
                                   labels=labels,
                                   set_ids=set_ids,
                                   batch_gen_params=config.batch_gen_params)

        model = model_dispatcher[model_name](device=config.trainer_params["device"],
                                             **config.model_params[model_name])

        date_dir = os.path.join(save_dir, start_date_str)
        if not os.path.exists(date_dir):
            os.makedirs(date_dir)
        trainer = Trainer(**config.trainer_params, save_dir=date_dir)

        # train model
        trainer.fit(model=model, batch_generator=generator)

        # perform prediction
        trainer.transform(model=model, batch_generator=generator)

        pred_dict = trainer.model_step_preds
        label_dict = trainer.model_step_labels
        scores = get_scores(pred_dict, label_dict)
        scores_path = os.path.join(date_dir, "scores.pkl")
        with open(scores_path, "wb") as f:
            pkl.dump(scores, f)
        print(scores)

        break


def get_scores(pred_dict, label_dict):
    pred_dict = {key: np.concatenate(val) for key, val in pred_dict.items()}
    label_dict = {key: np.concatenate(val) for key, val in label_dict.items()}

    scores = {}
    for key in pred_dict.keys():
        scores[key] = calculate_metrics(pred=pred_dict[key], label=label_dict[key])

    return scores


if __name__ == '__main__':
    run()
