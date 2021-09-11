import itertools

import numpy as np
from shapely.geometry import Polygon, LineString
from graph.graph_helper import plot_regions, plot_graph


class Graph:
    def __init__(self, grid, coord_range):
        self.grid = grid
        self.m, self.n = grid.shape
        self.coord_range = coord_range

        # graph elements
        self.nodes = {}
        self.edges = {}

        # some needed class attr
        self.regions = None
        self.coord_grid = None
        self.region_poly = None
        self.graph_built = False

    def create(self, threshold, plot=False):
        # divide grid into regions (rectangles) according to threshold value
        init_r, init_c = (0, self.m), (0, self.n)
        self.regions = self.__divide_regions(self.grid, threshold=threshold, r=init_r, c=init_c)

        # find each coordinate of the cells
        self.coord_grid = self.__create_coord_grid()  # M, N, 4, 2

        # convert each rectangle to polygon
        self.region_poly = self.__region2polygon()
        if plot:
            plot_regions(polygons_list=self.region_poly, coord_range=self.coord_range)

        # create nodes
        self.nodes = np.concatenate([poly.centroid.coords.xy for poly in self.region_poly], axis=1).T

        # create edges
        self.edges = self.get_intersections(self.region_poly)
        if plot:
            plot_graph(nodes=self.nodes, edges=self.edges)

        # graph is built
        self.graph_built = True

    def __divide_regions(self, in_grid, threshold, r, c):
        grid = in_grid[r[0]:r[1], c[0]:c[1]]

        if np.sum(grid) <= threshold or 1 in grid.shape:
            return [[r, c]]
        else:
            split_ids = self.split_regions(in_grid, r, c)
            region_ids = []
            for r, c in split_ids:
                region_id = self.__divide_regions(in_grid, threshold, r, c)
                region_ids += region_id
            return region_ids

    def __create_coord_grid(self):
        x = np.linspace(self.coord_range[1][0], self.coord_range[1][1], self.n + 1)
        y = np.linspace(self.coord_range[0][0], self.coord_range[0][1], self.m + 1)

        coord_grid = np.zeros((self.m, self.n, 4, 2))
        for j in range(self.m):
            for i in range(self.n):
                coords = np.array(list(itertools.product(x[i:i + 2], y[j:j + 2])))
                coords_ordered = coords[[0, 1, 3, 2], :]
                coord_grid[self.m - j - 1, i, :] = coords_ordered

        return coord_grid

    def __region2polygon(self):
        polygons_list = []
        for r, c in self.regions:
            region_pts = self.coord_grid[r[0]:r[1], c[0]:c[1]]
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

    # def __remove_node(self, v_id):
    #     n_nodes = self.edges[v_id][0]
    #     # remove prev node from lists of neighbours
    #     for n in n_nodes:
    #         neigh, dists = self.edges[n]
    #         if v_id in neigh:
    #             idx = neigh.index(v_id)
    #             neigh.remove(v_id)
    #             self.edges[n][1] = np.delete(dists, idx)
    #
    #     del self.nodes[v_id]
    #     del self.edges[v_id]
    #
    # @staticmethod
    # def _eu_distance(p1, p2):
    #     return np.sqrt(np.sum((p1 - p2) ** 2))
