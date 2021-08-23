import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import collections as mc
from shapely.geometry import Polygon, Point, LineString
import torch
import torch.nn as nn
import torch.optim as optim


from models.mlp import MLP


def adj_to_grid():
    pass


def run():
    # read data
    data = pd.read_csv("simulation_data.csv")

    # get the corner points of the rectangels
    polygon_points = []
    for i in range(1, 5):
        corners = data[[f"x{i}", f"y{i}"]].values.T
        polygon_points.append(corners)
    polygon_points = np.stack(polygon_points, axis=0)

    # convert to polygons
    polygons = []
    for i in range(len(data)):
        ply = Polygon(polygon_points[:, :, i])
        polygons.append(ply)

    # find neighbours of each ply
    intersections = {}
    for i in range(len(data)):
        intersections[i] = []
        for j in range(len(data)):
            if i == j:
                continue
            if polygons[i].intersects(polygons[j]) and \
                    isinstance(polygons[i].intersection(polygons[j]), LineString):
                intersections[i].append(j)

    # get centers
    centers = data[["cx", "cy"]].values

    # find edges btw centers
    edges = []
    for c_idx, intr in intersections.items():
        for i in intr:
            edges.append([centers[c_idx], centers[i]])
    lc = mc.LineCollection(edges, colors='k', linewidths=1)

    # plot the graph
    # fig, ax = plt.subplots()
    # ax.add_collection(lc)
    # ax.scatter(centers[:, 0], centers[:, 1])
    # for i in range(len(centers)):
    #     ax.text(centers[i, 0], centers[i, 1], f"{data.index[i]}")
    # plt.show()

    # adjacency matrix
    n = len(centers)
    adj = np.eye(n)
    for c_idx, intr in intersections.items():
        adj[c_idx, intr] = 1

    # fig, ax = plt.subplots()
    # ax.imshow(adj)
    # plt.show()

    device = torch.device("cpu")
    epoch = 1000
    optimizer = "adam"
    criterion = nn.MSELoss()
    learning_rate = 0.01
    momentum = 0.07

    mlp_conf = {
        "input_dim": 2,
        "output_dim": 20,
        "hidden_dim": [100, 50],
        "num_layers": 3,
        "bias": True,
        "activations": ["relu", "relu", "softmax"],
    }

    model = MLP(**mlp_conf).to(device)

    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum)

    for i in range(epoch):
        adj_tensor = torch.from_numpy(adj)
        for c in centers:
            input_tensor = torch.from_numpy(c).to(device)
            pred = model(input_tensor)




if __name__ == '__main__':
    run()

