import os
import pickle as pkl

import pandas as pd
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from configs.graph_config import GraphConfig
from data_generators.graph_creator import GraphCreator
from data_generators.grid_creator import GridCreator
from batch_generators.batch_generator import BatchGenerator
from trainer import Trainer
from models.graph_model import GraphModel
from helpers.static_helper import get_save_dir, get_set_ids, get_set_end_date, bin_pred, f1_score
from helpers.graph_helper import sample_dist, get_grid_label


def run():
    config = GraphConfig()
    grid_creator = GridCreator(data_params=config.data_params, grid_params=config.grid_params)

    if not grid_creator.check_is_created():
        print(f"Data is not found in {grid_creator.grid_save_dir}. Starting data creation...")
        grid_creator.create_grid()

    # create save path
    model_name = "graph_model"
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
        set_ids = [train_ids, val_ids, test_ids]

        for c in grid_creator.crime_types:
            print(c)
            grid = grid_creator.load_grid(dataset_name=c)[..., [2]]
            graph_creator = GraphCreator(data_params=config.data_params,
                                         graph_params=config.graph_params,
                                         save_dir=f"{start_date_str}_{c}")
            loaded = graph_creator.load()
            if not loaded:
                graph_creator.create_graph(grid=grid, crime_type=c)

            labels = np.array(graph_creator.labels)
            generator = BatchGenerator(in_data=graph_creator.node_features,
                                       labels=labels,
                                       set_ids=set_ids,
                                       edge_index=graph_creator.edge_index,
                                       regions=graph_creator.regions,
                                       batch_gen_params=config.batch_gen_params)

            model = GraphModel(device=config.trainer_params["device"],
                               node_count=graph_creator.node_features.shape[1],
                               **config.model_params["graph_model"])

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
                              k_nearest=2)

            # train model
            trainer.fit(model=model, batch_generator=generator)

            # perform prediction
            trainer.transform(model=model, batch_generator=generator)

            print(f"Experiment finished for {c}, calculating scores and obtaining samples")
            pred_dict = trainer.model_step_preds
            label_dict = trainer.model_step_labels
            stats = get_stats(pred_dict, label_dict,
                              coord_range=[[0, 1], [0, 1]],
                              grid_shape=config.grid_params["spatial_res"])
            stats_path = os.path.join(date_dir, "stats.pkl")
            with open(stats_path, "wb") as f:
                pkl.dump(stats, f)
            print(stats[0])


def get_stats(pred_dict, label_dict, coord_range, grid_shape):
    scores_dict = {}
    preds_dict = {}
    labels_dict = {}

    for key in pred_dict.keys():
        print(key)
        pred_list = pred_dict[key]
        label_list = label_dict[key]

        pred_arr = []
        label_arr = []
        score = 0
        for t in range(len(pred_list)):
            print(t)
            pred_mu, pred_sigma = pred_list[t]
            label = label_list[t]

            batch_dists = []
            for i in range(pred_mu.shape[0]):
                dists = []
                for j in range(pred_mu.shape[1]):
                    mu = torch.from_numpy(pred_mu[i, j])
                    sigma = torch.eye(2) * torch.from_numpy(pred_sigma[i, j])
                    m = MultivariateNormal(mu.T, sigma)
                    dists.append(m)
                batch_dists.append(dists)

            grid_pred = sample_dist(batch_dists, coord_range=coord_range, grid_shape=grid_shape)
            grid_label = get_grid_label(label, coord_range=coord_range, grid_shape=grid_shape)
            pred = bin_pred(grid_pred.flatten(), grid_label.flatten())
            score += f1_score(grid_label.flatten(), pred)

            pred_arr.append(grid_pred)
            label_arr.append(grid_label)

        mean_score = score / len(pred_list)
        print(f"{key}, F1 Score: {mean_score}")
        scores_dict[key] = mean_score
        preds_dict[key] = np.concatenate(pred_arr)
        labels_dict[key] = np.concatenate(label_arr)
        scores_dict[key] = mean_score

    return scores_dict, preds_dict, labels_dict


if __name__ == '__main__':
    run()
