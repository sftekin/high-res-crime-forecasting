import multiprocessing

import torch
import numpy as np
import matplotlib.collections
import matplotlib.pyplot as plt
from descartes import PolygonPatch


def plot_poly(in_poly, ax, face_color="b", edge_color="k", alpha=0.5):
    patch = PolygonPatch(in_poly, fc=face_color, ec=edge_color, alpha=alpha)
    ax.add_patch(patch)
    return ax


def plot_regions(polygons_list, coord_range, color="w"):
    fig, ax = plt.subplots(figsize=(10, 15))
    for i, poly in enumerate(polygons_list):
        plot_poly(poly, ax, face_color=color)
    plt.xlim(*coord_range[1])
    plt.ylim(*coord_range[0])
    plt.show()


def plot_graph(nodes, edges):
    edge_lines = []
    for c_idx, neigh in edges.items():
        for i in neigh:
            edge_lines.append([nodes[c_idx], nodes[i]])
    lc = matplotlib.collections.LineCollection(edge_lines, colors='k', linewidths=1)

    fig, ax = plt.subplots(figsize=(10, 15))
    ax.add_collection(lc)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=10)
    plt.show()


def flatten_labels(labels, regions):
    time_len = len(labels)
    flattened_labels = []
    for r, c in regions:
        flatten_arr = labels[:, r[0]:r[1], c[0]:c[1]].reshape(time_len, -1)
        flattened_labels.append(flatten_arr)
    f_labels = np.concatenate(flattened_labels, axis=1)
    return f_labels


def inverse_label(pred, label_shape, regions):
    batch_size = pred.shape[0]
    grid = torch.zeros(batch_size, *label_shape)
    prev_idx = 0
    for r, c in regions:
        row_count = r[1] - r[0]
        col_count = c[1] - c[0]
        cell_count = row_count * col_count
        grid[:, r[0]:r[1], c[0]:c[1]] = \
            pred[:, prev_idx:prev_idx + cell_count].reshape(-1, row_count, col_count)
        prev_idx += cell_count
    return grid


def get_probs(pred, node2cell):
    pred_mu, pred_sigma = pred
    batch_prob = []
    for batch_id in range(pred_mu.shape[0]):
        prob = []
        for node_id, cell_arr in node2cell.items():
            mu1, mu2 = pred_mu[batch_id, node_id]
            sigma1, sigma2 = pred_sigma[batch_id, node_id]
            p1 = __calc_prob(cell_arr[:, 0], mu1, sigma1)
            p2 = __calc_prob(cell_arr[:, 1], mu2, sigma2)
            prob.append(p1 * p2)
        prob = torch.cat(prob)
        batch_prob.append(prob)
    batch_prob = torch.stack(batch_prob)

    return batch_prob


def sample_dist(batch_dist, grid_shape, coord_range):
    batch_size = len(batch_dist)
    batch_grid = []
    for i in range(batch_size):
        samples = []
        dists = batch_dist[i]
        for j in range(len(dists)):
            samples.append(dists[j].sample((1000,)).cpu().numpy())
        samples = np.concatenate(samples)
        grid = convert_grid(samples, grid_shape, coord_range)
        batch_grid.append(grid)
    batch_grid = np.stack(batch_grid)
    return batch_grid


def get_grid_label(label, coord_range, grid_shape):
    batch_label = []
    for i in range(len(label)):
        if isinstance(label[i], torch.Tensor):
            y = label[i].detach().cpu().numpy()
        else:
            y = label[i]
        grid = convert_grid(y[:, :2],
                            coord_range=coord_range, grid_shape=grid_shape)
        grid = (grid > 0).astype(int)
        batch_label.append(grid)
    batch_label = np.stack(batch_label)
    return batch_label


def convert_grid(in_arr, grid_shape, coord_range):
    m, n = grid_shape
    x_ticks = np.linspace(coord_range[1][0], coord_range[1][1], n + 1)
    y_ticks = np.linspace(coord_range[0][0], coord_range[0][1], m + 1)

    arg_list = []
    for j in range(m):
        for i in range(n):
            arg_list.append([in_arr, x_ticks, y_ticks, (i, j)])

    with multiprocessing.Pool(processes=16) as pool:
        preds = list(pool.imap(upsample, arg_list))
        grid = np.flip(np.array(preds).reshape(m, n), axis=0)

    return grid


def upsample(args):
    in_arr, x_ticks, y_ticks, (i, j) = args
    lat_idx = (y_ticks[j] < in_arr[:, 1]) & (in_arr[:, 1] <= y_ticks[j + 1])
    lon_idx = (x_ticks[i] < in_arr[:, 0]) & (in_arr[:, 0] <= x_ticks[i + 1])
    sum = in_arr[lat_idx & lon_idx].sum()
    return sum


def __calc_prob(x, mu, sigma):
    x1 = (x[:, 0] - mu) / (sigma * 1.41)
    x2 = (x[:, 1] - mu) / (sigma * 1.41)
    prob = (torch.erf(x2) - torch.erf(x1)) * 0.5
    return prob
