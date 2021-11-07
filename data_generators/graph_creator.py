import os
from tqdm import tqdm
import pickle as pkl
import multiprocessing

import numpy as np
import pandas as pd

import itertools
from data_generators.data_creator import DataCreator
from shapely.geometry import Polygon, LineString, Point
from helpers.graph_helper import plot_regions, plot_graph
from helpers.plot_helper import plot_hist_dist
from concurrent.futures import ThreadPoolExecutor


class GraphCreator(DataCreator):
    def __init__(self, data_params, graph_params, save_dir):
        super(GraphCreator, self).__init__(data_params)
        self.threshold = graph_params["event_threshold"]
        self.include_side_info = graph_params["include_side_info"]
        self.grid_name = graph_params["grid_name"]
        self.min_cell_size = graph_params["min_cell_size"]
        self.normalize_coords = graph_params["normalize_coords"]
        self.k = graph_params["k_nearest"]

        self.node_features = None
        self.edge_index = None
        self.edge_weights = None
        self.regions = None
        self.labels = None
        self.node2cells = {}

        # create the data_dump directory
        self.graph_save_dir = os.path.join(self.temp_dir, "graph", save_dir,
                                           f"data_dump_{self.temp_res}_{self.threshold}")
        if not os.path.exists(self.graph_save_dir):
            os.makedirs(self.graph_save_dir)

        edge_index_path = os.path.join(self.graph_save_dir, "edge_index.pkl")
        node_features_path = os.path.join(self.graph_save_dir, "node_features.pkl")
        node2cells_path = os.path.join(self.graph_save_dir, "node2cells.pkl")
        regions_path = os.path.join(self.graph_save_dir, "regions.pkl")
        labels_path = os.path.join(self.graph_save_dir, "labels.pkl")
        edge_weight_path = os.path.join(self.graph_save_dir, "edge_weight.pkl")
        self.paths = [edge_index_path, node_features_path, node2cells_path,
                      regions_path, labels_path, edge_weight_path]

    def create_graph(self, grid, crime_type):
        crime_df = super().create()

        if self.normalize_coords:
            coord_arr = crime_df[["Latitude", "Longitude"]].values
            (min_lat, max_lat), (min_lon, max_lon) = self.coord_range
            coord_arr[:, 0] = (coord_arr[:, 0] - min_lat) / (max_lat - min_lat)
            coord_arr[:, 1] = (coord_arr[:, 1] - min_lon) / (max_lon - min_lon)
            crime_df.loc[:, ["Latitude", "Longitude"]] = coord_arr
            self.coord_range = [[0, 1], [0, 1]]

        # divide grid into regions (rectangles) according to threshold value
        m, n = grid.shape[1:3]
        init_r, init_c = (0, m), (0, n)
        grid_sum = np.squeeze(np.sum(grid, axis=0))
        regions = self.__divide_regions(grid_sum, threshold=self.threshold, r=init_r, c=init_c)

        # find each coordinate of the cells
        coord_grid = self.__create_coord_grid(m, n)  # M, N, 4, 2

        # convert each rectangle to polygon
        polygons = self.__region2polygon(coord_grid, regions)

        # create nodes
        nodes = np.concatenate([poly.centroid.coords.xy for poly in polygons], axis=1).T

        # create edges
        edges = self.get_intersections(polygons)

        # create graph parameters
        self.edge_index = self.create_edge_index(edges)
        self.edge_weights = self.create_edge_weights(nodes)
        self.node_features = self.__create_node_features(crime_df, nodes, polygons)
        self.regions = regions
        self.__create_node_cells(regions, coord_grid)
        self.labels = self.__create_labels(crime_df[crime_df[crime_type] == 1], nodes)

        if self.plot:
            plot_regions(polygons, coord_range=self.coord_range)
            plot_graph(nodes=nodes, edges=edges)
            node_sum = np.sum(self.node_features[:, :, -1], axis=0)
            zero_ratio = sum(node_sum == 0) / len(node_sum) * 100
            save_path = os.path.join(self.figures_dir, "nodes_hist.png")
            plot_hist_dist(node_sum,
                           x_label="Total Event per Node",
                           title=f"Zero Ratio {zero_ratio:.2f}",
                           save_path=save_path)

        # save created data
        self.__save_data()
        print(f"Data Creation finished, data saved under {self.graph_save_dir}")

    def __create_labels(self, crime_df, nodes):
        arg_list = []
        for t in self.date_r:
            arg_list.append((crime_df, t, nodes))

        with multiprocessing.Pool(processes=self.num_process) as pool:
            labels = list(tqdm(pool.imap(self.match_event_node, arg_list), total=len(arg_list)))

        return labels

    def match_event_node(self, args):
        crime_df, t, nodes = args
        label_arr = []
        t_1 = t + pd.DateOffset(hours=self.temp_res)
        cropped_df = crime_df.loc[(t <= crime_df.index) & (crime_df.index < t_1)]
        if not cropped_df.empty:
            locs = cropped_df[["Longitude", "Latitude"]].values
            dist_mat = self.calculate_distances(locs, nodes, self.num_process)
            node_contains = np.argsort(dist_mat, axis=1)[:, :self.k]
            label_arr = np.concatenate([locs, node_contains], axis=1)
        return label_arr

    def load(self):
        loaded = False
        if all([os.path.exists(path) for path in self.paths]):
            with open(self.paths[0], "rb") as f:
                self.edge_index = pkl.load(f)
            with open(self.paths[1], "rb") as f:
                self.node_features = pkl.load(f)
            with open(self.paths[2], "rb") as f:
                self.node2cells = pkl.load(f)
            with open(self.paths[3], "rb") as f:
                self.regions = pkl.load(f)
            with open(self.paths[4], "rb") as f:
                self.labels = pkl.load(f)
            with open(self.paths[5], "rb") as f:
                self.edge_weights = pkl.load(f)
            loaded = True
        return loaded

    def __create_node_features(self, crime_df, nodes, polygons):
        # create lat lon range for each polygon
        ranges = []
        for poly in polygons:
            lns, lats = poly.exterior.coords.xy
            lt_range = [min(lats), max(lats)]
            ln_range = [min(lns), max(lns)]
            ranges.append([lt_range, ln_range])

        # create node features
        time_len, num_nodes = len(self.date_r), len(nodes)
        if self.include_side_info:
            num_feats = crime_df.shape[1] + len(self.crime_types)  # categorical features + event_count + node_location
        else:
            num_feats = 2 + len(self.crime_types)  # event count + node_location
        node_features = np.zeros((time_len, num_nodes, num_feats))
        for n in range(len(nodes)):
            lt, ln = ranges[n]
            region_df = self.get_in_range(crime_df, lt, ln)
            node_features[:, n, :2] = nodes[n]  # first 2 features are the location of node
            if not region_df.empty:
                event_counts = []
                for crime in self.crime_types:
                    in_df = region_df[region_df[crime] == 1]
                    events = in_df.resample(f"{self.temp_res}H").size().reindex(self.date_r, fill_value=0)
                    event_counts.append(events.values)
                event_counts = np.stack(event_counts, axis=1)
                node_features[:, n, 2:len(self.crime_types) + 2] = event_counts  # append the event counts
                if self.include_side_info:
                    cat_df = region_df.resample(f"{self.temp_res}H").mean().reindex(self.date_r, fill_value=0)
                    cat_df = cat_df.fillna(0)
                    cat_df = cat_df.drop(columns=["Latitude", "Longitude"])
                    node_features[:, n, len(self.crime_types) + 2:] = cat_df.values

        return node_features

    def __create_node_cells(self, regions, coord_grid):
        for i, (r, c) in enumerate(regions):
            region_cells = coord_grid[r[0]:r[1], c[0]:c[1]].reshape(-1, 4, 2)
            region_cells = np.stack([np.min(region_cells, axis=1),
                                     np.max(region_cells, axis=1)], axis=-1)
            self.node2cells[i] = region_cells

    def __save_data(self):
        items = [self.edge_index, self.node_features, self.node2cells, self.regions, self.labels, self.edge_weights]
        for path, item in zip(self.paths, items):
            with open(path, "wb") as f:
                pkl.dump(item, f)

    def __divide_regions(self, in_grid, threshold, r, c):
        grid = in_grid[r[0]:r[1], c[0]:c[1]]

        if np.sum(grid) <= threshold or grid.shape <= self.min_cell_size:
            return [[r, c]]
        else:
            split_ids = self.split_regions(in_grid, r, c)
            region_ids = []
            for r, c in split_ids:
                region_id = self.__divide_regions(in_grid, threshold, r, c)
                region_ids += region_id
            return region_ids

    def __create_coord_grid(self, m, n):
        x = np.linspace(self.coord_range[1][0], self.coord_range[1][1], n + 1)
        y = np.linspace(self.coord_range[0][0], self.coord_range[0][1], m + 1)

        coord_grid = np.zeros((m, n, 4, 2))
        for j in range(m):
            for i in range(n):
                coords = np.array(list(itertools.product(x[i:i + 2], y[j:j + 2])))
                coords_ordered = coords[[0, 1, 3, 2], :]
                coord_grid[m - j - 1, i, :] = coords_ordered

        return coord_grid

    @staticmethod
    def create_edge_index(edges):
        edge_index = []
        for node_id, neighs in edges.items():
            for n in neighs:
                edge_index.append([node_id, n])
        edge_index = np.array(edge_index).T
        return edge_index

    def create_edge_weights(self, nodes):
        weights = []
        for i in range(self.edge_index.shape[1]):
            n1, n2 = self.edge_index[:, i]
            weights.append(np.sum((nodes[n1] - nodes[n2]) ** 2))
        weights = np.array(weights)
        return weights

    @staticmethod
    def get_in_range(cor_df, lt, ln):
        lat_idx = (lt[0] < cor_df["Latitude"]) & (cor_df["Latitude"] <= lt[1])
        lon_idx = (ln[0] < cor_df["Longitude"]) & (cor_df["Longitude"] <= ln[1])
        in_range_df = cor_df[lat_idx & lon_idx]
        return in_range_df

    @staticmethod
    def get_intersections(polygons_list):
        intersections = {}
        for i in range(len(polygons_list)):
            intersections[i] = []
            for j in range(len(polygons_list)):
                if i == j:
                    continue
                if polygons_list[i].intersects(polygons_list[j]) and \
                        isinstance(polygons_list[i].intersection(polygons_list[j]), LineString):
                    intersections[i].append(j)
        return intersections

    @staticmethod
    def __region2polygon(coord_grid, regions):
        polygons_list = []
        for r, c in regions:
            region_pts = coord_grid[r[0]:r[1], c[0]:c[1]]
            region_pts = region_pts.reshape(-1, 2)
            lon_min, lon_max = np.min(region_pts[:, 0]), np.max(region_pts[:, 0])
            lat_min, lat_max = np.min(region_pts[:, 1]), np.max(region_pts[:, 1])

            coords = np.array(list(itertools.product([lon_min, lon_max], [lat_min, lat_max])))
            coords_ordered = coords[[0, 1, 3, 2], :]
            polygon = Polygon(coords_ordered)
            polygons_list.append(polygon)

        return polygons_list

    @staticmethod
    def split_regions(in_grid, r, c):
        m, n = r[1] - r[0], c[1] - c[0]
        pos_m, pos_n = [m // 2], [n // 2]
        if m % 2 > 0:
            pos_m.append(m // 2 + 1)
        if n % 2 > 0:
            pos_n.append(n // 2 + 1)
        pos_m = [i + r[0] for i in pos_m]
        pos_n = [i + c[0] for i in pos_n]

        max_sum = -1
        best_indices = None
        r_i, r_j = r
        c_i, c_j = c
        for i, (m_id, n_id) in enumerate(itertools.product(pos_m, pos_n)):
            indices = [[(r_i, m_id), (c_i, n_id)],
                       [(r_i, m_id), (n_id, c_j)],
                       [(m_id, r_j), (c_i, n_id)],
                       [(m_id, r_j), (n_id, c_j)]]
            regions = [in_grid[x[0]:x[1], y[0]:y[1]] for x, y in indices]
            region_sums = [np.sum(r) for r in regions]

            if max_sum < max(region_sums):
                best_indices = indices
                max_sum = max(region_sums)

        return best_indices

    @staticmethod
    def calculate_distances(unk_locs, knw_locs, num_processes):
        """
        Calculates distance matrix.

        :param np.ndarray unk_locs: (N, 2)
        :param np.ndarray knw_locs: (M, 2)
        :param int num_processes:
        :return: distance matrix (M, N)
        :rtype: np.ndarray
        """

        def l2_dist(x, y):
            """
            calculates distance between vector and point as elementwise

            :param np.ndarray x: (2,)
            :param np.ndarray y: (N, 2)
            :return: distance vector (N,2)
            :rtype: np.ndarray
            """
            d = np.sum((y - x) ** 2, axis=1)
            return d

        num_unk_points = unk_locs.shape[0]
        arg_list = [(unk_locs[i], knw_locs) for i in range(num_unk_points)]
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            D = executor.map(lambda p: l2_dist(*p), arg_list)
            D = np.stack(list(D), axis=0)
        return D

