import math
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import collections as mc
from shapely.geometry import Polygon, LineString


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
    plt.show()

    # adjacency matrix
    n = len(centers)
    adj = np.eye(n)
    for c_idx, intr in intersections.items():
        adj[c_idx, intr] = 1

    # fig, ax = plt.subplots()
    # ax.imshow(adj)
    # plt.show()

    # graph2grid
    graph = intersections
    graph_drc = graph_with_drc(graph, centers)
    M, N = 5, 6
    grid = np.zeros((M, N))
    grid[:] = np.nan
    init_node = 7
    init_grid_idx = (2, 3)
    grid[init_grid_idx] = init_node

    contradictions = []
    placed = np.array([False for _ in range(len(graph.keys()))])
    placed[init_node] = True
    placements = [(init_node, init_grid_idx)]
    while not all(placed):
        new_placements = []
        for node_name, grid_idx in placements:
            new_p, contr = place_neighbours(grid, graph_drc, node_name, grid_idx, placed)
            new_placements += new_p
            contradictions += contr
            placed[node_name] = True
        placements = new_placements
    print("all placed")
    print(grid)

    # handle contradictions
    for idx in contradictions:
        directions = get_directions(idx)
        pool = []
        for drc_name, n_idx in directions.items():
            if not check_exists(n_idx, grid.shape):
                continue

            neigh_node = grid[n_idx]
            node_neighbours = graph_drc[neigh_node][inv_direction(drc_name)]
            node_neighbours = node_neighbours.tolist() if isinstance(node_neighbours, np.ndarray) else node_neighbours
            pool += node_neighbours
        counts = Counter(pool)
        selected_node = sorted(counts, key=counts.get, reverse=True)[0]
        grid[idx] = selected_node
    print(grid)


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


def graph_with_drc(in_graph, node_coords):
    directions = ["right", "top", "left", "bot"]
    direction_dict = {}
    for node_name, neigh_list in in_graph.items():
        direction_dict[node_name] = {drc: [] for drc in directions}
        origin_coord = node_coords[node_name]
        for n in neigh_list:
            n_coord = node_coords[n]
            drc_idx = calc_dir(origin_coord, n_coord)
            direction_dict[node_name][directions[drc_idx]].append(n)

        for drc, n_list in direction_dict[node_name].items():
            if len(n_list) > 1:
                direction_dict[node_name][drc] = order_nodes(node_name, n_list, node_coords)

    return direction_dict


def calc_dir(p1, p2):
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    if angle < 0:
        drc = np.array([0, 90, -180, -90])
    else:
        drc = np.array([0, 90, 180, 270])
    diff = np.abs(angle - drc)
    return np.argmin(diff)


def order_nodes(center_node, neigh_nodes, node_coords):
    dist = np.array([np.sum((node_coords[coord] - node_coords[center_node]) ** 2) for coord in neigh_nodes])
    ordered = np.argsort(dist)
    neigh_nodes = np.array(neigh_nodes)
    return neigh_nodes[ordered]


def place_neighbours(in_grid, in_graph, node_name, grid_idx, placed):
    directions = get_directions(in_idx=grid_idx)

    placed_nodes, contradictions = [], []
    for drc_name, idx in directions.items():
        if not check_exists(idx, in_grid.shape):
            continue

        if not np.isnan(in_grid[idx]) and placed[int(in_grid[idx])]:
            continue

        neigh_nodes = in_graph[node_name][drc_name]
        if len(neigh_nodes) == 0:
            continue

        if not np.isnan(in_grid[idx]) and in_grid[idx] != neigh_nodes[0]:
            contradictions.append(idx)
        in_grid[idx] = neigh_nodes[0]  # think about this
        placed_nodes.append((neigh_nodes[0], idx))

    return placed_nodes, contradictions


def get_directions(in_idx):
    directions = {
        "left": (in_idx[0], in_idx[1] - 1),
        "right": (in_idx[0], in_idx[1] + 1),
        "top": (in_idx[0] - 1, in_idx[1]),
        "bot": (in_idx[0] + 1, in_idx[1])
    }
    return directions


def check_exists(idx, in_shape):
    x, y = in_shape
    x_check = (0 <= idx[0]) & (idx[0] < x)
    y_check = (0 <= idx[1]) & (idx[1] < y)
    return x_check and y_check


def inv_direction(direction):
    inv_dir = {
        "left": "right",
        "right": "left",
        "top": "bot",
        "bot": "top",
    }
    return inv_dir[direction]


if __name__ == '__main__':
    run()
