import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import collections as mc
from shapely.geometry import Polygon, Point, LineString


def run():
    data = pd.read_csv("simulation_data.csv")
    polygon_points = []
    for i in range(1, 5):
        corners = data[[f"x{i}", f"y{i}"]].values.T
        polygon_points.append(corners)
    polygon_points = np.stack(polygon_points, axis=0)

    polygons = []
    for i in range(len(data)):
        ply = Polygon(polygon_points[:, :, i])
        polygons.append(ply)

    intersections = {}
    for i in range(len(data)):
        intersections[i] = []
        for j in range(len(data)):
            if i == j:
                continue
            if polygons[i].intersects(polygons[j]) and \
                    isinstance(polygons[i].intersection(polygons[j]), LineString):
                intersections[i].append(j)

    centers = data[["cx", "cy"]].values

    edges = []
    for c_idx, intr in intersections.items():
        for i in intr:
            edges.append([centers[c_idx], centers[i]])
    lc = mc.LineCollection(edges, colors='k', linewidths=1)

    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.scatter(centers[:, 0], centers[:, 1])
    for i in range(len(centers)):
        ax.text(centers[i, 0], centers[i, 1], f"{data.index[i]}")

    plt.show()


if __name__ == '__main__':
    run()

