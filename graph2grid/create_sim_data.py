import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from descartes import PolygonPatch


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

    centroids = [np.array(poly.centroid.coords.xy) for poly in poly_list]
    centroids = np.squeeze(np.array(centroids))
    for grid_idx, center in enumerate(centroids):
        plt.text(center[0], center[1], str(grid_idx + 1))
    plt.show()

    poly_coord_arr = np.array(sample_polygon_coords)
    x_coords = poly_coord_arr[:, :, 0]
    y_coords = poly_coord_arr[:, :, 1]

    data_values = np.concatenate([x_coords, y_coords, centroids], axis=1)
    columns = ["x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4", "cx", "cy"]
    data_df = pd.DataFrame(data_values, columns=columns)
    data_df.to_csv("simulation_data.csv")


if __name__ == '__main__':
    run()
