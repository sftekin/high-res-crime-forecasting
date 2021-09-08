import numpy as np
from shapely.geometry import Polygon, LineString


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.graph_built = False
        # self.adjacency = None

    def create(self, poly_list):
        # create nodes

        # create edges
        edges = {}
        for i in range(len(poly_list)):
            edges[i] = []
            for j in range(len(poly_list)):
                if i == j:
                    continue
                if poly_list[i].intersects(poly_list[j]) and \
                        isinstance(poly_list[i].intersection(poly_list[j]), LineString):
                    edges[i].append(j)

    def get_closest_neighbour(self, idx):
        nodes, distances = self.edges[idx]
        return nodes[0]

    def __remove_node(self, v_id):
        n_nodes = self.edges[v_id][0]
        # remove prev node from lists of neighbours
        for n in n_nodes:
            neigh, dists = self.edges[n]
            if v_id in neigh:
                idx = neigh.index(v_id)
                neigh.remove(v_id)
                self.edges[n][1] = np.delete(dists, idx)

        del self.nodes[v_id]
        del self.edges[v_id]

    @staticmethod
    def _eu_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
