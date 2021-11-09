import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def run():
    dataset_dir = "../dataset"
    save_path = os.path.join(dataset_dir, "simulation.csv")
    coord_range = [[0, 1], [0, 1]]
    node_count = 40
    group_count = 4
    pts = (0.8 - 0.2) * torch.rand(node_count, 2) + 0.2

    pts_arr = pts.numpy()
    groups = np.repeat(np.arange(group_count), node_count // group_count)

    plt.figure()
    for i, marker in enumerate(["o", "+", "x", "^"]):
        coords = pts_arr[groups == i]
        plt.scatter(coords[:, 0], coords[:, 1], marker=marker, label=f"group_{i}")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

    total_time_step = 3000
    num_events_per_step = create_ar_series(total_time_step)
    intensities = np.array([.1, .2, .3, .4])

    events_per_group = []
    for i in range(total_time_step):
        events_per_group.append((num_events_per_step[i] * intensities).astype(int))
    events_per_group = np.stack(events_per_group)

    group_sigma_val = np.array([0.008, 0.008, 0.008, 0.008])

    group_profiles = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

    # create distributions
    dists = []
    group_ids = []
    for group_id in range(group_count):
        group_dists = pts[groups == group_id]
        sigma = torch.eye(2) * group_sigma_val[group_id]
        for mu in group_dists:
            m = MultivariateNormal(mu.T, sigma)
            dists.append(m)
            group_ids.append(group_id)
    dists = np.array(dists)
    group_ids = np.array(group_ids)

    group_profiles = group_profiles.flatten()
    group_events = {key: [] for key in range(group_count)}
    for i in range(total_time_step):
        if i % 5 == 0:
            group_profiles = np.logical_not(group_profiles)
        selected_dists = dists[group_profiles]
        selected_ids = group_ids[group_profiles]

        events = [[] for g in range(group_count)]
        for dist, group_id in zip(selected_dists, selected_ids):
            event = dist.sample((events_per_group[i, group_id],)).clip(0, 1)
            events[group_id].append(event)

        for g in range(group_count):
            group_events[g].append(np.concatenate(events[g]))

    # create simulation csv
    dfs = []
    for t in range(total_time_step):
        for key, val in group_events.items():
            data = val[t]
            index = (np.ones(len(data)) * t).astype(int)
            df_t = pd.DataFrame(data, columns=["Latitude", "Longitude"], index=index)
            df_t["group"] = key
            dfs.append(df_t)
    all_df = pd.concat(dfs, axis=0)
    all_df.to_csv(save_path)
    print("data created")

    # for key, val in group_events.items():
    #     events = np.concatenate(val)
    #     grid = _convert_grid(events, (25, 25), coord_range)
    #     plt.figure()
    #     plt.imshow(grid)
    #     plt.title(f"{key}")
    #     plt.show()


def create_ar_series(total_len):
    series = []
    w = np.array([-0.01])
    y_t_1 = 0
    c = 200
    for i in range(total_len):
        e_t = np.random.normal(scale=10)
        y_t = c + e_t + (w[0] * y_t_1)
        y_t_1 = y_t
        series.append(y_t)
    series = np.array(series)
    return series


if __name__ == '__main__':
    run()




