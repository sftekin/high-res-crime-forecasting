import os
import pickle as pkl

import torch
import numpy as np
from data_generators.graph_creator import GraphCreator
from configs.graph_config import GraphConfig
from helpers.graph_helper import get_probs, inverse_label, flatten_labels
from helpers.plot_helper import _plot_2d


def run():
    model_name = "graph_model"
    exp_name = "exp_3"
    model_dir = os.path.join("results", model_name, exp_name)

    config = GraphConfig()
    graph_creator = GraphCreator(data_params=config.data_params,
                                 graph_params=config.graph_params)
    loaded = graph_creator.load()
    if not loaded:
        raise RuntimeError("Data is not created")

    # load model
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model = pkl.load(f)

    # prepare in data
    win_in_len = 10
    time_step = np.random.randint(1200, 1300)
    events = graph_creator.labels > 0
    events = events.astype(int)
    f_labels = flatten_labels(labels=events, regions=graph_creator.regions)
    x = torch.from_numpy(graph_creator.node_features[time_step:time_step+win_in_len]).unsqueeze(0).float()
    y = torch.from_numpy(f_labels[time_step+win_in_len]).unsqueeze(0).float()
    edge_index = torch.from_numpy(graph_creator.edge_index)

    node2cell = graph_creator.node2cells
    for i, arr in node2cell.items():
        node2cell[i] = torch.from_numpy(arr).float()

    model.to("cpu")
    pred = model(x, edge_index)
    batch_prob = get_probs(pred, node2cell)
    grid_pred = inverse_label(pred=batch_prob,
                              label_shape=config.grid_params["spatial_res"],
                              regions=graph_creator.regions)
    grid_pred = grid_pred.squeeze().detach().numpy()
    _plot_2d(in_grid=grid_pred,
             coord_range=graph_creator.coord_range,
             grid_size=config.grid_params["spatial_res"],
             title="Predictions")

    # _plot_3d(grid_pred,
    #          coord_range=graph_creator.coord_range,
    #          grid_size=config.grid_params["spatial_res"],
    #          title="Predictions")

    # load stats
    stats_path = os.path.join(model_dir, "statistics.pkl")
    with open(stats_path, "rb") as f:
        stats = pkl.load(f)


if __name__ == '__main__':
    run()
