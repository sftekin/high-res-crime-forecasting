import numpy as np

import itertools
from data_generators.data_creator import DataCreator
from shapely.geometry import Polygon, LineString
from helpers.graph_helper import plot_regions, plot_graph


class GraphCreator(DataCreator):
    def __init__(self, data_params, graph_params):
        super(GraphCreator, self).__init__(data_params)
        self.threshold = graph_params["event_threshold"]

        self.regions = None
        self.polygons = None
        self.nodes = None
        self.edges = None

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

        self.regions = regions
        self.polygons = polygons
        self.nodes = nodes
        self.edges = edges

    def create_node_features(self):
        pass

    def create_edge_idx(self):
        pass

    def create_y(self):
        pass

    def __divide_into_regions(self, crime_df, lat_range, lon_range, threshold):
        cor_df = crime_df[["Latitude", "Longitude"]]
        region_count = self.get_count(cor_df, lat_range, lon_range)

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
    def get_count(cor_df, lt, ln):
        lat_idx = (lt[0] < cor_df["Latitude"]) & (cor_df["Latitude"] <= lt[1])
        lon_idx = (ln[0] < cor_df["Longitude"]) & (cor_df["Longitude"] <= ln[1])
        event_count = len(cor_df[lat_idx & lon_idx])
        return event_count

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
