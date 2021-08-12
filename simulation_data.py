from shapely.geometry import Polygon
from descartes import PolygonPatch
import matplotlib.pyplot as plt


def plot_poly(in_poly, ax, face_color="b", edge_color="k", alpha=0.5):
    patch = PolygonPatch(in_poly, fc=face_color, ec=edge_color, alpha=alpha)
    ax.add_patch(patch)
    return ax


def run():
    sample_polygon_coords = [
        [(1, 1), (1, 5), (3, 5), (3, 1)],
        [(3, 1), (3, 3), (4, 3), (4, 1)],
        [(3, 3), (3, 5), (4, 5), (4, 3)],
        [(4, 4), (4, 5), (5, 5), (5, 4)],
        [(5, 4), (5, 5), (6, 5), (6, 4)],
        [(6, 4), (6, 5), (7, 5), (7, 4)],
        [(4, 3), (4, 4), (5, 4), (5, 3)],
        [(5, 3), (5, 4), (6, 4), (6, 3)],
        [(6, 3), (6, 4), (7, 4), (7, 3)],
        [(4, 2), (4, 3), (5, 3), (5, 2)],
        [(5, 2), (5, 3), (6, 3), (6, 2)],
        [(6, 2), (6, 3), (7, 3), (7, 2)],
        [(4, 1), (4, 2), (5, 2), (5, 1)],
        [(5, 1), (5, 2), (6, 2), (6, 1)],
        [(6, 1), (6, 2), (7, 2), (7, 1)],
        [(7, 4), (7, 5), (10, 5), (10, 4)],
        [(7, 1), (7, 4), (10, 4), (10, 1)],
        [(1, 5), (1, 10), (6, 10), (6, 5)],
        [(6, 5), (6, 10), (10, 10), (10, 5)]
    ]

    poly_list = []
    for poly_coords in sample_polygon_coords:
        poly_list.append(Polygon(poly_coords))

    fig, ax = plt.subplots()
    for i, poly in enumerate(poly_list):
        color = f"C{i}"
        plot_poly(poly, ax, face_color=color)
    plt.xlim(1, 10)
    plt.ylim(1, 10)
    plt.show()


if __name__ == '__main__':
    run()
