import itertools
from tqdm import tqdm
import multiprocessing

import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.collections
import matplotlib.pyplot as plt
from descartes import PolygonPatch

from helpers.static_helper import bin_pred, f1_score, confusion_matrix, accuracy_score


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


def get_log_like(pred, node2cell):
    pred_mu, pred_sigma = pred
    device = pred_mu.device
    batch_prob = []
    for i in range(pred_mu.shape[0]):
        prob = []
        for j in range(pred_mu.shape[1]):
            mu = pred_mu[i, j]
            sigma = torch.eye(2).to(device) * pred_sigma[i, j]
            m = MultivariateNormal(mu.T, sigma)
            prob.append(m.log_prob(node2cell[j]))
        prob = torch.cat(prob)
        batch_prob.append(prob)
    batch_prob = torch.stack(batch_prob)

    return batch_prob


def _calc_prob(x, mu, sigma):
    x1 = (x[:, 0] - mu) / (sigma * 1.41)
    x2 = (x[:, 1] - mu) / (sigma * 1.41)
    prob = (torch.erf(x2) - torch.erf(x1)) * 0.5
    return prob


def get_graph_stats(pred_batches, label_batches, coord_range, grid_shape, num_process=16):
    arg_list = []
    for batch_id in range(len(pred_batches)):
        arg_list.append([pred_batches[batch_id], label_batches[batch_id],
                         coord_range, grid_shape])

    with multiprocessing.Pool(processes=num_process) as pool:
        grids = list(tqdm(pool.imap(_get_batch_stats, arg_list), total=len(arg_list)))

    grid_pred = np.concatenate([grids[i][0] for i in range(len(grids))])
    grid_label = np.concatenate([grids[i][1] for i in range(len(grids))])

    pred = bin_pred(grid_pred.flatten(), grid_label.flatten())
    f1 = f1_score(grid_label.flatten(), pred)

    tn, fn, fp, tp = confusion_matrix(y_true=grid_label.flatten(), y_pred=pred).flatten()
    acc = accuracy_score(y_true=grid_label.flatten(), y_pred=pred)

    print(f"Confusion Matrix = TN:{tn}, FN:{fn}, FP:{fp}, TP:{tp}")
    print(f"Accuracy = {acc:.4f}")

    return f1, grid_pred, grid_label


def _get_batch_stats(args):
    model_out, label, coord_range, grid_shape = args
    pred_mu, pred_sigma = model_out

    coords = create_coord_grid(coord_range, spatial_size=grid_shape).mean(axis=2).reshape(-1, 2)

    batch_loglike = []
    for i in range(pred_mu.shape[0]):
        log_likes = []
        for j in range(pred_mu.shape[1]):
            mu = torch.from_numpy(pred_mu[i, j])
            sigma = torch.eye(2) * 0.01
            m = MultivariateNormal(mu.T, sigma)
            log_like = m.log_prob(torch.from_numpy(coords).float())
            log_likes.append(log_like)
        log_likes, _ = torch.stack(log_likes, dim=1).max(dim=1)
        batch_loglike.append(log_likes)
    batch_loglike = torch.stack(batch_loglike).numpy()
    batch_loglike = batch_loglike.reshape(-1, *grid_shape)

    label_list = []
    for i in range(len(label)):
        l_arr = np.concatenate(list(label[i].values()))
        label_list.append(l_arr)

    grid_label = _get_grid_label(label_list, coord_range=coord_range, grid_shape=grid_shape)
    grids = [batch_loglike, grid_label]

    return grids


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

def _sample_dist(batch_dist, grid_shape, coord_range):
    batch_size = len(batch_dist)
    batch_grid = []
    for i in range(batch_size):
        samples = []
        dists = batch_dist[i]
        for j in range(len(dists)):
            samples.append(dists[j].sample((1000,)).cpu().numpy())
        samples = np.concatenate(samples)
        grid = _convert_grid(samples, grid_shape, coord_range)
        batch_grid.append(grid)
    batch_grid = np.stack(batch_grid)
    return batch_grid


def _get_grid_label(label, coord_range, grid_shape):
    batch_label = []
    for i in range(len(label)):
        if isinstance(label[i], torch.Tensor):
            y = label[i].detach().cpu().numpy()
        else:
            y = label[i]
        grid = _convert_grid(y[:, :2], coord_range=coord_range, grid_shape=grid_shape)
        grid = (grid > 0).astype(int)
        batch_label.append(grid)
    batch_label = np.stack(batch_label)
    return batch_label


def _convert_grid(in_arr, grid_shape, coord_range):
    m, n = grid_shape
    x_ticks = np.linspace(coord_range[1][0], coord_range[1][1], n + 1)
    y_ticks = np.linspace(coord_range[0][0], coord_range[0][1], m + 1)
    grid = np.zeros((m, n))
    for j in range(m):
        for i in range(n):
            lat_idx = (y_ticks[j] < in_arr[:, 1]) & (in_arr[:, 1] <= y_ticks[j + 1])
            lon_idx = (x_ticks[i] < in_arr[:, 0]) & (in_arr[:, 0] <= x_ticks[i + 1])
            grid[m - j - 1, i] = in_arr[lat_idx & lon_idx].sum()

    return grid
