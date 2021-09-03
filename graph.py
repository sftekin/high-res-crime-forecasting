import numpy as np


class Graph:
    def __init__(self, grid, location_grid):
        self.grid = grid
        self.location_grid = location_grid

        self.vertices = {}
        self.edges = {}
        # self.adjacency = None

    def create_from_grid(self):
        m, n = self.grid.shape
        for j in range(m):
            for i in range(n):
                # create vertex
                ravel_idx = j * n + i
                v_location = self.location_grid[j, i]
                v_value = self.grid[j, i]
                self.vertices[ravel_idx] = [v_value, v_location]

                # create edge
                n_vertices = self._get_grid_neighbours(in_grid=self.grid, idx=(j, i))
                n_distances = self.__get_neigh_distances(v_location, n_vertices)
                self.edges[ravel_idx] = [n_vertices, n_distances]

    def merge_vertices(self, v1, v2):
        # merge vertex
        v_val = self.vertices[v1][0] + self.vertices[v2][0]
        v_loc = (self.vertices[v1][1] + self.vertices[v2][1]) / 2
        new_vertex = [v_val, v_loc]

        # merge edge
        n_vertices = self.edges[v1][0] + self.edges[v2][0]
        n_distances = self.__get_neigh_distances(v_loc, n_vertices)

        # remove prev and add new
        self.__remove_vertex(v1)
        self.__remove_vertex(v2)
        self.vertices[f"{v1}_{v2}"] = new_vertex
        self.edges[f"{v1}_{v2}"] = [n_vertices, n_distances]

    def __remove_vertex(self, v_id):
        del self.vertices[v_id]
        del self.edges[v_id]

    def __get_neigh_distances(self, v_loc, n_vertices):
        n_locations = [self.location_grid[np.unravel_index(n, shape=self.grid.shape)] for n in n_vertices]
        n_distances = [self._eu_distance(v_loc, n) for n in n_locations]
        n_values = [self.vertices[n][0] for n in n_vertices]
        concat_arr = np.concatenate([n_distances, n_values], axis=1)
        concat_arr = concat_arr[np.lexsort((concat_arr[:, 0], concat_arr[:, 1]))]

        return concat_arr

    @staticmethod
    def _get_grid_neighbours(in_grid, idx):
        m, n = in_grid.shape
        l = idx[0], idx[1] - 1
        t = idx[0] - 1, idx[1]
        r = idx[0], idx[1] + 1
        d = idx[0] + 1, idx[1]
        neighbours = []
        for ind in [l, t, r, d]:
            if (0 <= ind[0] < m) & (0 <= ind[1] < n):
                neighbours.append(ind[0] * n + ind[1])

        return neighbours

    @staticmethod
    def _eu_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
