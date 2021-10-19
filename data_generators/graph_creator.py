import os
import pickle as pkl

import numpy as np
import pandas as pd

import itertools
from data_generators.grid_creator import GridCreator
from shapely.geometry import Polygon, LineString
from helpers.graph_helper import plot_regions, plot_graph
from helpers.plot_helper import plot_hist_dist
from concurrent.futures import ThreadPoolExecutor


class GraphCreator(GridCreator):
    def __init__(self, data_params, grid_params, graph_params):
        super(GraphCreator, self).__init__(data_params, grid_params)
        self.threshold = graph_params["event_threshold"]
        self.include_side_info = graph_params["include_side_info"]
        self.grid_name = graph_params["grid_name"]

        self.node_features = None
        self.edge_index = None
        self.labels = None

        # create the data_dump directory
        self.save_dir = os.path.join(self.temp_dir, "graph", f"data_dump_{self.temp_res}_{self.threshold}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        edge_index_path = os.path.join(self.save_dir, "edge_index.pkl")
        node_features_path = os.path.join(self.save_dir, "node_features.pkl")
        labels_path = os.path.join(self.save_dir, "labels.pkl")
        self.paths = [edge_index_path, node_features_path, labels_path]

    def create_graph(self):
        crime_df = super().create()
        if super().check_is_created():
            grid = super().load_grid(mode="all")
        else:
            grid = super().create_grid()

        # divide grid into regions (rectangles) according to threshold value
        init_r, init_c = (0, self.m), (0, self.n)
        self.regions = self.__divide_regions(self.grid, threshold=threshold, r=init_r, c=init_c)

        # find each coordinate of the cells
        self.coord_grid = self.__create_coord_grid()  # M, N, 4, 2

        # convert each rectangle to polygon
        self.region_poly = self.__region2polygon()

        # create nodes
        self.nodes = np.concatenate([poly.centroid.coords.xy for poly in self.region_poly], axis=1).T

        # create edges
        self.edges = self.get_intersections(self.region_poly)
        if plot:
            plot_graph(nodes=self.nodes, edges=self.edges)

        regions = self.__divide_into_regions(crime_df,
                                             lat_range=self.coord_range[0],
                                             lon_range=self.coord_range[1],
                                             threshold=self.threshold)
        # create nodes
        polygons = self.region2polygon(regions)
        nodes = np.concatenate([poly.centroid.coords.xy for poly in polygons], axis=1).T

        # create edges
        edges = self.get_intersections(polygons)

        # crea
        self.edge_index = self.create_edge_index(edges)
        self.node_features = self.__create_node_features(crime_df, nodes, regions)
        self.labels = self.__create_labels(crime_df, nodes)

        if self.plot:
            plot_regions(polygons, coord_range=self.coord_range)
            plot_graph(nodes=nodes, edges=edges)
            node_sum = np.sum(self.node_features[:, :, -1], axis=0)
            zero_ratio = sum(node_sum == 0) / len(node_sum) * 100
            save_path = os.path.join(self.figures_dir, "nodes_hist.png")
            plot_hist_dist(node_sum, x_label="Total Event per Node",
                           title=f"Zero Ratio {zero_ratio:.2f}",
                           save_path=save_path)

        # save created data
        self.__save_data()
        print(f"Data Creation finished, data saved under {self.save_dir}")

    def load(self):
        loaded = False
        if all([os.path.exists(path) for path in self.paths]):
            with open(self.paths[0], "rb") as f:
                self.edge_index = pkl.load(f)
            with open(self.paths[1], "rb") as f:
                self.node_features = pkl.load(f)
            with open(self.paths[2], "rb") as f:
                self.labels = pkl.load(f)

            loaded = True
        return loaded

    def __divide_into_regions(self, crime_df, lat_range, lon_range, threshold):
        cor_df = crime_df[["Latitude", "Longitude"]]
        region_count = len(self.get_in_range(cor_df, lat_range, lon_range))

        if region_count <= threshold:
            return [[lat_range, lon_range]]
        else:
            new_lats, new_lons = self.divide4(lat_range, lon_range)
            regions = []
            for lt_range, ln_range in itertools.product(new_lats, new_lons):
                region = self.__divide_into_regions(crime_df, lt_range, ln_range, threshold)
                regions += region
            return regions

    def __create_node_features(self, crime_df, nodes, regions):
        time_len, num_nodes = len(self.date_r), len(nodes)
        if self.include_side_info:
            num_feats = crime_df.shape[1] + 1  # categorical features + event_count + node_location
        else:
            num_feats = 3  # event count + node_location

        node_features = np.zeros((time_len, num_nodes, num_feats))
        for n in range(len(nodes)):
            lt, ln = regions[n]
            region_df = self.get_in_range(crime_df, lt, ln)
            node_features[:, n, :2] = nodes[n]  # first 2 features are the location of node
            if not region_df.empty:
                event_count = region_df.resample(f"{self.temp_res}H").size().reindex(self.date_r, fill_value=0)
                node_features[:, n, 2] = event_count.values  # third feature is the event count
                if self.include_side_info:
                    cat_df = region_df.resample(f"{self.temp_res}H").mean().reindex(self.date_r, fill_value=0)
                    cat_df = cat_df.fillna(0)
                    cat_df = cat_df.drop(columns=["Latitude", "Longitude"])
                    node_features[:, n, 3:] = cat_df.values

        return node_features

    def __create_labels(self, crime_df, nodes):
        arg_list = []
        for t in self.date_r:
            arg_list.append((crime_df, t, nodes))
        with ThreadPoolExecutor(max_workers=self.num_process) as executor:
            labels = executor.map(lambda p: self.__match_event_node(*p), arg_list)
            labels = list(labels)
        return labels

    def __match_event_node(self, crime_df, t, nodes):
        label_arr = []
        t_1 = t + pd.DateOffset(hours=self.temp_res)
        cropped_df = crime_df.loc[(t <= crime_df.index) & (crime_df.index < t_1)]
        if not cropped_df.empty:
            locs = cropped_df[["Longitude", "Latitude"]].values
            dist_mat = self.calculate_distances(locs, nodes, self.num_process)
            node_contains = np.argmin(dist_mat, axis=1).reshape(-1, 1)
            label_arr = np.concatenate([locs, node_contains], axis=1)
        return label_arr

    def __save_data(self):
        items = [self.edge_index, self.node_features, self.labels]
        for path, item in zip(self.paths, items):
            with open(path, "wb") as f:
                pkl.dump(item, f)

    @staticmethod
    def create_edge_index(edges):
        edge_index = []
        for node_id, neighs in edges.items():
            for n in neighs:
                edge_index.append([node_id, n])
        edge_index = np.array(edge_index).T
        return edge_index

    @staticmethod
    def get_in_range(cor_df, lt, ln):
        lat_idx = (lt[0] < cor_df["Latitude"]) & (cor_df["Latitude"] <= lt[1])
        lon_idx = (ln[0] < cor_df["Longitude"]) & (cor_df["Longitude"] <= ln[1])
        in_range_df = cor_df[lat_idx & lon_idx]
        return in_range_df

    @staticmethod
    def divide4(lt, ln):
        y_mid = lt[0] + abs(lt[1] - lt[0]) / 2
        x_mid = ln[0] + abs(ln[1] - ln[0]) / 2
        lts = [[lt[0], y_mid], [y_mid, lt[1]]]
        lns = [[ln[0], x_mid], [x_mid, ln[1]]]
        return lts, lns

    @staticmethod
    def region2polygon(regions):
        polygons_list = []
        for lt, ln in regions:
            coords = np.array(list(itertools.product(ln, lt)))
            coords_ordered = coords[[0, 1, 3, 2], :]
            polygon = Polygon(coords_ordered)
            polygons_list.append(polygon)
        return polygons_list

    @staticmethod
    def get_intersections(polygons_list):
        intersectons = {}
        for i in range(len(polygons_list)):
            intersectons[i] = []
            for j in range(len(polygons_list)):
                if i == j:
                    continue
                if polygons_list[i].intersects(polygons_list[j]) and \
                        isinstance(polygons_list[i].intersection(polygons_list[j]), LineString):
                    intersectons[i].append(j)
        return intersectons

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
