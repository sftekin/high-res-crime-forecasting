import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import collections as mc
from shapely.geometry import Polygon, Point, LineString
import torch
import torch.nn as nn
import torch.optim as optim


from models.mlp import MLP


def classes_to_adj(ind, in_shape):
    pos_grid = np.zeros(in_shape)
    for i in range(len(ind)):
        x, y = np.unravel_index(ind[i], shape=in_shape)
        pos_grid[x, y] = i + 1

    adj = grid_to_adj(in_grid=pos_grid)

    return adj


def grid_to_adj(in_grid):
    n = np.int(np.max(in_grid))
    adj = np.zeros((n, n))
    for i in range(in_grid.shape[0]):
        for j in range(in_grid.shape[1]):
            if in_grid[i, j] > 0:
                neighbours = search_neighbours(in_grid, idx=(i, j))
                for n in neighbours:
                    adj[int(n) - 1] = 1

    return adj


def search_neighbours(in_grid, idx):
    m, n = in_grid.shape
    l = idx[0], idx[1] - 1
    t = idx[0] - 1, idx[1]
    r = idx[0], idx[1] + 1
    d = idx[0] + 1, idx[1]
    neighbours = []
    for ind in [l, t, r, d]:
        if (0 <= ind[0] < m) & (0 <= ind[1] < n):
            if in_grid[ind] > 0:
                neighbours.append(in_grid[ind])
    return neighbours



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
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.scatter(centers[:, 0], centers[:, 1])
    for i in range(len(centers)):
        ax.text(centers[i, 0], centers[i, 1], f"{data.index[i]}")

    # adjacency matrix
    n = len(centers)
    adj = np.eye(n)
    for c_idx, intr in intersections.items():
        adj[c_idx, intr] = 1

    # fig, ax = plt.subplots()
    # ax.imshow(adj)
    # plt.show()

    device = torch.device("cpu")
    epoch = 10000
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
        # adj_target = torch.from_numpy(adj).float().to(device)
        adj_target = adj
        optimizer.zero_grad()
        preds = []
        for c in centers:
            input_tensor = torch.from_numpy(c).unsqueeze(dim=0).float().to(device)
            pred = model(input_tensor)
            preds.append(pred)
        pred_t = torch.cat(preds)
        ind = torch.argmax(pred_t, dim=1)
        adj_pred = classes_to_adj(ind, in_shape=(5, 4))
        diff_adj = np.abs(adj_target - adj_pred)
        weight_adj = np.sum(diff_adj, axis=1)
        weight_adj = torch.from_numpy(weight_adj).float().to(device)
        loss = torch.sum(pred_t * weight_adj.unsqueeze(dim=1))
        # adj_pred = torch.matmul(pred_t, pred_t.T)
        # loss = criterion(adj_target, adj_pred)
        loss.backward()
        optimizer.step()

        print(i, loss)

    print(adj_pred)




if __name__ == '__main__':
    run()

