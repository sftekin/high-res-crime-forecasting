import numpy as np
import pandas as pd

import itertools
from data_generators.data_creator import DataCreator
from shapely.geometry import Polygon, LineString
from helpers.graph_helper import plot_regions, plot_graph


class GraphCreator(DataCreator):
    def __init__(self, data_params, graph_params):
        super(GraphCreator, self).__init__(data_params)
        self.threshold = graph_params["event_threshold"]
        self.include_side_info = graph_params["include_side_info"]

        self.regions = None
        self.node_features = None
        self.edge_idx = None
        self.labels = None

    def create(self):
        crime_df = super().create()
        regions = self.__divide_into_regions(crime_df,
                                             lat_range=self.coord_range[0],
                                             lon_range=self.coord_range[1],
                                             threshold=self.threshold)
        polygons = self.region2polygon(regions)
        if self.plot:
            plot_regions(polygons, coord_range=self.coord_range)

        # create nodes
        nodes = np.concatenate([poly.centroid.coords.xy for poly in polygons], axis=1).T

        # create edges
        edges = self.get_intersections(polygons)
        if self.plot:
            plot_graph(nodes=nodes, edges=edges)

        self.edge_idx = self.create_edge_idx(edges)
        self.node_features = self.create_node_features(crime_df, nodes, regions)
        self.labels = self.create_labels()

    def create_node_features(self, crime_df, nodes, regions):
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

    def create_labels(self):
        pass

    @staticmethod
    def create_edge_idx(edges):
        edge_index = []
        for node_id, neighs in edges.items():
            for n in neighs:
                edge_index.append([node_id, n])
        edge_index = np.array(edge_index)
        return edge_index

    def create_y(self):
        pass

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
