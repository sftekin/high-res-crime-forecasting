import numpy as np


class Graph:
    def __init__(self, grid, location_grid):
        self.grid = grid
        self.location_grid = location_grid

        self.vertices = {}
        self.edges = {}
        self.graph_built = False
        # self.adjacency = None

    def create_from_grid(self):
        m, n = self.grid.shape
        for j in range(m):
            for i in range(n):
                # create vertex
                ravel_idx = j * n + i
                vertex_loc = self.location_grid[j, i]
                vertex_val = self.grid[j, i]
                self.vertices[ravel_idx] = [vertex_val, vertex_loc]

                # create edge
                neigh_vertices = self._get_cell_neighbours(in_grid=self.grid, idx=(j, i))
                neigh_dists = self.__get_dist2neigh(vertex_loc, neigh_vertices)
                neigh_vertices, n_distances = self.__order_neighbours(neigh_vertices, neigh_dists)
                self.edges[ravel_idx] = [neigh_vertices, n_distances]
        self.graph_built = True

    def get_closest_neighbour(self, idx):
        vertices, distances = self.edges[idx]
        return vertices[0]

    def merge_vertices(self, v1, v2):
        # merge vertex values, new value and new location
        vertex_val = self.vertices[v1][0] + self.vertices[v2][0]
        vertex_loc = (self.vertices[v1][1] + self.vertices[v2][1]) / 2
        new_vertex = [vertex_val, vertex_loc]

        # merge edges, new neighbours
        neigh_vertices = self.edges[v1][0] + self.edges[v2][0]
        neigh_vertices = list(dict.fromkeys(neigh_vertices))  # drop duplicates
        if v1 in neigh_vertices:
            neigh_vertices.remove(v1)
        neigh_vertices.remove(v2)

        neigh_dists = self.__get_dist2neigh(vertex_loc, neigh_vertices)
        neigh_vertices, neigh_dists = self.__order_neighbours(neigh_vertices, neigh_dists)
        new_edge = [neigh_vertices, neigh_dists]

        # remove prev vertices
        self.__remove_vertex(v1)
        self.__remove_vertex(v2)

        self.vertices[v1] = new_vertex
        self.edges[v1] = new_edge

        return vertex_val

    def __get_dist2neigh(self, v_loc, n_vertices):
        neigh_locs = []
        for n in n_vertices:
            if self.graph_built:
                loc = self.vertices[n][1]
            else:
                loc = self.location_grid[np.unravel_index(n, shape=self.grid.shape)]
            neigh_locs.append(loc)
        neigh_dists = np.array([self._eu_distance(v_loc, n) for n in neigh_locs])
        return neigh_dists

    def __order_neighbours(self, n_vertices, n_distances):
        neigh_vals = []
        for n in n_vertices:
            if self.graph_built:
                val = self.vertices[n][0]
            else:
                val = self.grid[np.unravel_index(n, shape=self.grid.shape)]
            neigh_vals.append(val * -1)
        concat_arr = np.stack([n_distances, neigh_vals], axis=1)
        sort_idx = np.lexsort((concat_arr[:, 0], concat_arr[:, 1]))

        ordered_d = n_distances[sort_idx]
        ordered_n = [n_vertices[i] for i in sort_idx]
        return ordered_n, ordered_d

    def __remove_vertex(self, v_id):
        n_vertices = self.edges[v_id][0]
        # remove prev vertex from lists of neighbours
        for n in n_vertices:
            neigh, dists = self.edges[n]
            if v_id in neigh:
                idx = neigh.index(v_id)
                neigh.remove(v_id)
                self.edges[n][1] = np.delete(dists, idx)

        del self.vertices[v_id]
        del self.edges[v_id]

    @staticmethod
    def _get_cell_neighbours(in_grid, idx):
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
