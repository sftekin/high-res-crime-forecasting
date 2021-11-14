import os
import pickle as pkl
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from configs.graph_config import GraphConfig
from data_generators.graph_creator import GraphCreator
from data_generators.grid_creator import GridCreator
from batch_generators.batch_generator import BatchGenerator
from trainer import Trainer
from models.graph_conv_gru import GraphConvGRU
from models.lstm import LSTM
from helpers.static_helper import get_save_dir, get_set_ids, get_set_end_date

model_dispatcher = {
    "graph_conv_gru": GraphConvGRU,
    "lstm": LSTM
}


def run():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    config = GraphConfig()
    grid_creator = GridCreator(data_params=config.data_params, grid_params=config.grid_params)

    if not grid_creator.check_is_created():
        print(f"Data is not found in {grid_creator.grid_save_dir}. Starting data creation...")
        grid_creator.create_grid()

    # create save path
    model_name = "graph_conv_gru"
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

        win_in_len = config.batch_gen_params["window_in_len"]
        val_ids = np.concatenate([train_ids[-win_in_len:], val_ids])
        test_ids = np.concatenate([val_ids[-win_in_len:], test_ids])
        set_ids = [train_ids, val_ids, test_ids]

        for c in ["all"]:
            print(c)
            grid = grid_creator.load_grid(dataset_name=c)[..., [2]]
            graph_creator = GraphCreator(data_params=config.data_params,
                                         graph_params=config.graph_params,
                                         save_dir=f"{start_date_str}_{c}")
            loaded = graph_creator.load()
            if not loaded:
                graph_creator.create_graph(grid=grid)

            loss_type = config.trainer_params["loss_function"]
            if loss_type == "prob_loss":
                labels = (grid > 0).astype(int)
            else:
                labels = graph_creator.create_labels(crime_type=c)
                labels = np.array(labels)
            generator = BatchGenerator(in_data=graph_creator.node_features,
                                       labels=labels,
                                       set_ids=set_ids,
                                       edge_index=graph_creator.edge_index,
                                       regions=graph_creator.regions,
                                       batch_gen_params=config.batch_gen_params,
                                       loss_type=loss_type)

            model = model_dispatcher[model_name](device=config.trainer_params["device"],
                                                 node_count=graph_creator.node_features.shape[1],
                                                 **config.model_params[model_name])

            date_dir = os.path.join(save_dir, start_date_str, c)
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)
            trainer = Trainer(**config.trainer_params,
                              save_dir=date_dir,
                              node2cell=graph_creator.node2cells,
                              regions=graph_creator.regions,
                              nodes=graph_creator.node_features[0, :, :2],
                              coord_range=[[0, 1], [0, 1]],
                              spatial_res=config.grid_params["spatial_res"],
                              k_nearest=5,
                              edge_weights=graph_creator.edge_weights)

            # train model
            trainer.fit(model=model, batch_generator=generator)

            # perform prediction
            trainer.transform(model=model, batch_generator=generator)

            print(f"Experiment finished for {c}")
            stats = trainer.stats
            for key, val in stats.items():
                print(f"{key} F1 Score: {val[0]:.5f}")
            stats_path = os.path.join(date_dir, "stats.pkl")
            with open(stats_path, "wb") as f:
                pkl.dump(stats, f)


if __name__ == '__main__':
    run()
