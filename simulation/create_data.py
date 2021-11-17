import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from helpers.graph_helper import _convert_grid


def run():
    dataset_dir = "../dataset"
    save_path = os.path.join(dataset_dir, "simulation.csv")
    coord_range = [[0, 1], [0, 1]]
    node_count = 12
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

    group_sigma_val = np.array([0.02, 0.01, 0.005, 0.003])

    profile_1 = np.ones(node_count // group_count, dtype=int)
    profile_1[::2] = 0
    profile_2 = np.logical_not(profile_1).astype(int)

    profile_3 = np.ones(node_count // group_count, dtype=int)
    profile_3[::4] = 0
    profile_4 = np.logical_not(profile_3).astype(int)

    group_profiles = np.stack([profile_1, profile_2, profile_3, profile_4], axis=0)

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
    day = pd.to_datetime("2015-01-01")
    dfs = []
    for t in range(total_time_step):
        day += pd.DateOffset(hours=24)
        for key, val in group_events.items():
            data = val[t]
            idx = np.repeat(day, len(data))
            df_t = pd.DataFrame(data, columns=["Latitude", "Longitude"], index=idx)
            df_t["group"] = key
            dfs.append(df_t)
    all_df = pd.concat(dfs, axis=0)
    groups = pd.get_dummies(all_df["group"], prefix="group")
    all_df = pd.concat([all_df.drop(columns="group"), groups], axis=1)
    all_df.to_csv(save_path)
    print("data created")

    for key, val in group_events.items():
        events = np.concatenate(val)
        grid = _convert_grid(events, (25, 25), coord_range)
        plt.figure()
        plt.imshow(grid)
        plt.title(f"{key}")
        plt.show()


def create_ar_series(total_len):
    series = []
    w = np.array([-0.01])
    y_t_1 = 0
    c = 50
    for i in range(total_len):
        e_t = np.random.normal(scale=5)
        y_t = np.abs(c + e_t + (w[0] * y_t_1))
        y_t_1 = y_t
        series.append(y_t)
    series = np.array(series)
    plt.figure()
    plt.plot(series)
    plt.show()
    return series


if __name__ == '__main__':
    run()




