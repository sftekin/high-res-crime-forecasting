import itertools

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

from graph2grid.create_sim_data import plot_poly


def run():
    with open("grid.npy", "rb") as f:
        grid = np.load(f)
    grid_sum = np.sum(grid, axis=0)
    sorted_cells_idx = np.argsort(grid_sum.flatten())[::-1]
    sorted_cells = grid_sum.flatten()[sorted_cells_idx]

    cell_flags = np.zeros(len(sorted_cells), dtype=bool)

    threshold = 50
    regions = []
    for i in range(len(sorted_cells)):
        if cell_flags[i]:
            continue

        if sorted_cells[i] >= threshold:
            regions.append([sorted_cells_idx[i]])
            continue

        idx = np.unravel_index(sorted_cells_idx[i], shape=grid_sum.shape)
        count = sorted_cells[i]
        order = 1
        while count < threshold:
            count, neighbours = sum_neighbours(grid_sum, idx, order=order)
            order += 1

        cell_flags[neighbours] = True
        regions.append(neighbours.tolist())

    m, n = 100, 66
    coord_range = [[41.60, 42.05], [-87.9, -87.5]]
    x = np.linspace(coord_range[1][0], coord_range[1][1], n + 1)
    y = np.linspace(coord_range[0][0], coord_range[0][1], m + 1)

    coord_grid = np.zeros((m, n, 4, 2))
    for j in range(m):
        for i in range(n):
            coords = np.array(list(itertools.product(x[i:i+2], y[j:j+2])))
            coords_ordered = coords[[0, 1, 3, 2], :]
            coord_grid[m - j - 1, i, :] = coords_ordered

    polygons_list = []
    for region in regions:
        region_coords = []
        for idx in region:
            grid_idx = np.unravel_index(idx, shape=grid_sum.shape)
            region_coords.append(coord_grid[grid_idx])
        polygons = [Polygon(coord) for coord in region_coords]
        boundary = cascaded_union(polygons)
        polygons_list.append(boundary)

    fig, ax = plt.subplots()
    for i, poly in enumerate(polygons_list):
        color = f"C{i}"
        plot_poly(poly, ax, face_color=color)
    plt.xlim(*coord_range[1])
    plt.ylim(*coord_range[0])
    plt.show()


def sum_neighbours(in_grid, idx, order=1):
    kernel = np.zeros(in_grid.shape)
    i, j = idx
    kernel[i-order:i+order+1, j-order:j+order+1] = 1
    count = np.sum(kernel * in_grid)
    neighbours = np.where(kernel.flatten())[0]

    return count, neighbours


if __name__ == '__main__':
    run()
