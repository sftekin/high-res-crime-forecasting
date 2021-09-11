import math
from collections import Counter

import numpy as np


class GraphMapper:
    def __init__(self, grid_size, initial_node, initial_grid_loc):
        self.initial_node = initial_node
        self.initial_grid_loc = initial_grid_loc
        self.m, self.n = grid_size

        self.grid = np.zeros(grid_size)
        self.grid[:] = np.nan
        self.grid[initial_grid_loc] = initial_node

    def graph2grid(self, nodes, edges):
        # categorize every neighbour of each node under 4 directions
        graph_drc = self.__graph_with_drc(nodes, edges)

        # fill the grid
        contradicts = self.__fill_grid(graph_drc)

        # handle contradictions
        self.__solve_contradicts(contradicts, graph_drc)

        return self.grid

    def __graph_with_drc(self, nodes, edges):
        directions = ["right", "top", "left", "bot"]
        direction_dict = {}
        for node_name, neigh_list in edges.items():
            direction_dict[node_name] = {drc: [] for drc in directions}
            origin_coord = nodes[node_name]
            for n in neigh_list:
                n_coord = nodes[n]
                drc_idx = self.calc_dir(origin_coord, n_coord)
                direction_dict[node_name][directions[drc_idx]].append(n)

            for drc, n_list in direction_dict[node_name].items():
                if len(n_list) > 1:
                    direction_dict[node_name][drc] = self.order_nodes(node_name, n_list, nodes)

        return direction_dict

    def __fill_grid(self, graph_drc):
        # keep track of placed nodes
        node_count = len(graph_drc.keys())
        placed = np.zeros(node_count, dtype=bool)

        # set the initials
        placed[self.initial_node] = True
        placements = [(self.initial_node, self.initial_grid_loc)]

        contradictions = []
        while not all(placed):
            new_placements = []
            for node_name, grid_idx in placements:
                new_p, contr = self.__place_neighbours(graph_drc, node_name, grid_idx, placed)
                new_placements += new_p
                contradictions += contr
                placed[node_name] = True
            placements = new_placements

        return contradictions

    def __solve_contradicts(self, contradicts, graph_drc):
        for idx in contradicts:
            directions = self.get_directions(idx)
            pool = []
            for drc_name, n_idx in directions.items():
                if not self.check_exists(n_idx, self.grid.shape):
                    continue

                neigh_node = self.grid[n_idx]
                node_neighbours = graph_drc[neigh_node][self.inv_direction(drc_name)]
                node_neighbours = node_neighbours.tolist() if isinstance(node_neighbours,
                                                                         np.ndarray) else node_neighbours
                pool += node_neighbours
            counts = Counter(pool)
            selected_node = sorted(counts, key=counts.get, reverse=True)[0]
            self.grid[idx] = selected_node
        print(self.grid)

    def __place_neighbours(self, in_graph, node_name, grid_idx, placed):
        directions = self.get_directions(in_idx=grid_idx)

        placed_nodes, contradictions = [], []
        for drc_name, idx in directions.items():
            if not self.check_exists(idx, self.grid.shape):
                continue

            if not np.isnan(self.grid[idx]) and placed[int(self.grid[idx])]:
                continue

            neigh_nodes = in_graph[node_name][drc_name]
            if len(neigh_nodes) == 0:
                continue

            if not np.isnan(self.grid[idx]) and self.grid[idx] != neigh_nodes[0]:
                contradictions.append(idx)
            self.grid[idx] = neigh_nodes[0]  # think about this
            placed_nodes.append((neigh_nodes[0], idx))

        return placed_nodes, contradictions

    @staticmethod
    def calc_dir(p1, p2):
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        if angle < 0:
            drc = np.array([0, 90, -180, -90])
        else:
            drc = np.array([0, 90, 180, 270])
        diff = np.abs(angle - drc)
        return np.argmin(diff)

    @staticmethod
    def order_nodes(center_node, neigh_nodes, node_coords):
        dist = np.array([np.sum((node_coords[coord] - node_coords[center_node]) ** 2) for coord in neigh_nodes])
        ordered = np.argsort(dist)
        neigh_nodes = np.array(neigh_nodes)
        return neigh_nodes[ordered]

    @staticmethod
    def get_directions(in_idx):
        directions = {
            "left": (in_idx[0], in_idx[1] - 1),
            "right": (in_idx[0], in_idx[1] + 1),
            "top": (in_idx[0] - 1, in_idx[1]),
            "bot": (in_idx[0] + 1, in_idx[1])
        }
        return directions

    @staticmethod
    def check_exists(idx, in_shape):
        x, y = in_shape
        x_check = (0 <= idx[0]) & (idx[0] < x)
        y_check = (0 <= idx[1]) & (idx[1] < y)
        return x_check and y_check

    @staticmethod
    def inv_direction(direction):
        inv_dir = {
            "left": "right",
            "right": "left",
            "top": "bot",
            "bot": "top",
        }
        return inv_dir[direction]


