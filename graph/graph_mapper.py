import math
from collections import Counter

import numpy as np


class GraphMapper:
    def __init__(self, graph):
        self.input_graph = graph

        # create initial params for mapping
        self.m, self.n = self.calc_grid_size(graph)
        self.initial_node = graph.center_node
        self.init_m, self.init_n = self.calc_grid_center(grid_size=(self.m, self.n))

        # create the grid to be mapped
        self.grid = np.zeros((self.m, self.n))
        self.grid[:] = np.nan
        self.grid[self.init_m, self.init_n] = self.initial_node

    def map(self):
        # categorize every neighbour of each node under 4 directions
        graph_drc = self.__graph_with_drc()

        # fill the grid
        contradicts = self.__fill_grid(graph_drc)
        with open('downsampled_grid.npy', 'wb') as f:
            np.save(f, self.grid)

        # handle contradictions
        self.__solve_contradicts(contradicts, graph_drc)

        return self.grid

    def __graph_with_drc(self):
        nodes, edges = self.input_graph.nodes, self.input_graph.edges
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
        # set the initials
        placements = [(self.initial_node, (self.init_m, self.init_n))]

        contradictions = []
        while np.isnan(self.grid).sum() > 0:
            new_placements = []
            for node_name, grid_idx in placements:
                new_p, contr = self.__place_neighbours(graph_drc, node_name, grid_idx)
                new_placements += new_p
                contradictions += contr
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

    def __place_neighbours(self, in_graph, node_name, grid_idx):
        directions = self.get_directions(in_idx=grid_idx)

        placed_nodes, contradictions = [], []
        for drc_name, idx in directions.items():
            if not self.check_exists(idx, self.grid.shape):
                continue

            if not np.isnan(self.grid[idx]):
                continue

            neigh_nodes = in_graph[node_name][drc_name]
            if len(neigh_nodes) == 0:
                continue

            if not np.isnan(self.grid[idx]) and self.grid[idx] != neigh_nodes[0]:
                contradictions.append(idx)
            self.grid[idx] = neigh_nodes[0]  # think about this
            placed_nodes.append((neigh_nodes[0], idx))

        return placed_nodes, contradictions

    def continue_placing(self, placed, graph_drc):
        new_p = []
        not_placed_idx = np.squeeze(np.argwhere(~placed))
        for idx in not_placed_idx:
            for direction, neighbours in graph_drc[idx].items():
                for n in neighbours:
                    if placed[n] and n in self.grid:
                        n_ind = np.argwhere(self.grid == n)[0]
                        grid_idx = self.get_directions(n_ind)[direction]
                        if self.grid[grid_idx] == np.nan:
                            new_p.append((idx, grid_idx))

        return new_p

    @staticmethod
    def calc_grid_size(graph):
        regions_array = np.array(graph.regions)
        row_lines = regions_array[:, 0, :]
        col_lines = regions_array[:, 1, :]

        grid_size = []
        for interval in [col_lines, row_lines]:
            int_array = np.zeros(np.max(interval), dtype=int)
            for i in range(len(interval)):
                int_array[interval[i, 0]:interval[i, 1]] += 1

            grid_size.append(np.max(int_array))

        return grid_size

    @staticmethod
    def calc_grid_center(grid_size):
        dims = []
        for dim in grid_size:
            if dim % 2 == 0:
                center_idx = dim // 2
            else:
                center_idx = dim // 2 + 1
            dims.append(center_idx)

        return dims

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

    @staticmethod
    def plot_intervals(interval_unique):
        import matplotlib.collections
        import matplotlib.pyplot as plt
        lines = []
        for i in range(len(interval_unique)):
            p1 = np.array([interval_unique[i, 0], i])
            p2 = np.array([interval_unique[i, 1], i])
            lines.append([p1, p2])

        lc = matplotlib.collections.LineCollection(lines, colors='k', linewidths=1)

        fig, ax = plt.subplots(figsize=(10, 15))
        ax.add_collection(lc)
        ax.set_xlim(0, np.max(interval_unique))
        ax.set_ylim(0, len(lines))
        plt.show()


if __name__ == '__main__':
    import pickle as pkl
    from graph import Graph
    with open("graph.pkl", "rb") as f:
        graph = pkl.load(f)

    graph_mapper = GraphMapper(graph)
    grid = graph_mapper.map()


