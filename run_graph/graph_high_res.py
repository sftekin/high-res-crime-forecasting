import os
import pickle as pkl
import itertools

import torch
import numpy as np
from data_generators.graph_creator import GraphCreator
from data_generators.grid_creator import GridCreator
from configs.graph_config import GraphConfig
from helpers.graph_helper import get_log_like, inverse_label, flatten_labels
from helpers.plot_helper import _plot_2d
from shapely.geometry import Point


def create_grid(spatial_res):
    config = GraphConfig()
    config.grid_params["spatial_res"] = spatial_res
    grid_creator = GridCreator(data_params=config.data_params, grid_params=config.grid_params)
    if not grid_creator.check_is_created():
        grid_creator.create_grid()
    grid = grid_creator.load_grid(dataset_name="all")[..., [2]]
    return grid


def create_graph(spatial_res, grid):
    config = GraphConfig()
    config.grid_params["spatial_res"] = spatial_res
    graph_creator = GraphCreator(data_params=config.data_params,
                                 graph_params=config.graph_params,
                                 save_dir="2015-01-01_all")
    graph_creator.create_graph(grid=grid)
    return graph_creator


def run():
    model_name = "graph_conv_gru"
    exp_name = "exp_9/2015-01-01/all"
    model_dir = os.path.join("results", model_name, exp_name)
    # load low res model
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model = pkl.load(f)

    # create low and high res data
    grid_high = create_grid(spatial_res=(100, 66))
    grid_low = create_grid(spatial_res=(50, 33))
    graph_creator = create_graph(spatial_res=(50, 33), grid=grid_low)

    # prepare in data
    win_in_len = 10
    time_step = np.random.randint(1200, 1300)
    events = (grid_high > 0).astype(int)
    f_labels = flatten_labels(labels=events, regions=graph_creator.regions)
    x = torch.from_numpy(graph_creator.node_features[time_step:time_step+win_in_len]).unsqueeze(0).float()
    y = torch.from_numpy(f_labels[time_step+win_in_len]).unsqueeze(0).float()
    edge_index = torch.from_numpy(graph_creator.edge_index)

    # match cell coords and nodes
    coord_grid = graph_creator.create_coord_grid(m=100, n=66)
    coord_grid = coord_grid.mean(axis=2).reshape(-1, 2)
    polygons = graph_creator.polygons
    node2cell = {}
    node2cell_id = {}
    for i, poly in enumerate(polygons):
        node2cell[i] = []
        node2cell_id[i] = []
        for j in range(len(coord_grid)):
            if poly.contains(Point(coord_grid[j])):
                node2cell[i].append(coord_grid[j])
                node2cell_id[i].append(j)

    for i, arr in node2cell.items():
        node2cell[i] = torch.from_numpy(np.stack(arr)).float()

    for i, arr in node2cell_id.items():
        node2cell_id[i] = np.stack(arr)

    model.to("cpu")
    pred = model(x, edge_index)
    batch_prob = get_log_like(pred, node2cell).exp().detach().cpu()

    node2prob = {}
    c = 0
    for i, arr in node2cell.items():
        node2prob[i] = batch_prob[:, c:c + len(arr)]
        c += len(arr)

    grid_pred = inverse_label(pred=batch_prob.exp().detach().cpu(),
                              label_shape=(50, 33),
                              regions=graph_creator.regions)
    grid_pred = grid_pred.squeeze()
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
