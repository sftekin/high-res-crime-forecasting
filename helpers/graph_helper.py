import matplotlib.collections
import matplotlib.pyplot as plt
from descartes import PolygonPatch


def plot_poly(in_poly, ax, face_color="b", edge_color="k", alpha=0.5):
    patch = PolygonPatch(in_poly, fc=face_color, ec=edge_color, alpha=alpha)
    ax.add_patch(patch)
    return ax


def plot_regions(polygons_list, coord_range, color="w"):
    fig, ax = plt.subplots(figsize=(10, 15))
    for i, poly in enumerate(polygons_list):
        plot_poly(poly, ax, face_color=color)
    plt.xlim(*coord_range[1])
    plt.ylim(*coord_range[0])
    plt.show()


def plot_graph(nodes, edges):
    edge_lines = []
    for c_idx, neigh in edges.items():
        for i in neigh:
            edge_lines.append([nodes[c_idx], nodes[i]])
    lc = matplotlib.collections.LineCollection(edge_lines, colors='k', linewidths=1)

    fig, ax = plt.subplots(figsize=(10, 15))
    ax.add_collection(lc)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=10)
    plt.show()
