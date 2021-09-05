import itertools

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

from graph2grid.create_sim_data import plot_poly
from graph import Graph


def run():
    coord_range = [[41.60, 42.05], [-87.9, -87.5]]
    with open("grid.npy", "rb") as f:
        grid = np.load(f)
        grid = np.squeeze(grid)
    grid_sum = np.sum(grid, axis=0)
    coord_grid = create_coord_grid(in_grid=grid_sum, coord_range=coord_range)  # M, N, 4, 2
    grid_center_locs = np.mean(coord_grid, axis=2)

    # create graph
    graph = Graph(grid_sum, grid_center_locs)
    graph.create_from_grid()

    sorted_cells_idx = np.argsort(grid_sum.flatten())[::-1]
    sorted_cells = grid_sum.flatten()[sorted_cells_idx]
    cell_flags = np.zeros(len(sorted_cells), dtype=bool)

    threshold = 50
    regions = []
    for i in range(len(sorted_cells)):
        print(f"{i/len(sorted_cells) * 100:.2f}")
        if cell_flags[i]:
            continue

        if sorted_cells[i] >= threshold:
            regions.append([sorted_cells_idx[i]])
            continue

        count = sorted_cells[i]
        region = []
        while count < threshold:
            neigh_vertex = graph.get_closest_neighbour(idx=i)
            count = graph.merge_vertices(v1=i, v2=neigh_vertex)
            cell_flags[neigh_vertex] = True
            region.append(neigh_vertex)
        cell_flags[i] = True
        regions.append(region)

    print()


def sum_neighbours(in_grid, idx, order=1):
    kernel = np.zeros(in_grid.shape)
    i, j = idx
    kernel[i-order:i+order+1, j-order:j+order+1] = 1
    count = np.sum(kernel * in_grid)
    neighbours = np.where(kernel.flatten())[0]

    return count, neighbours


def create_coord_grid(in_grid, coord_range):
    m, n = in_grid.shape
    x = np.linspace(coord_range[1][0], coord_range[1][1], n + 1)
    y = np.linspace(coord_range[0][0], coord_range[0][1], m + 1)

    coord_grid = np.zeros((m, n, 4, 2))
    for j in range(m):
        for i in range(n):
            coords = np.array(list(itertools.product(x[i:i + 2], y[j:j + 2])))
            coords_ordered = coords[[0, 1, 3, 2], :]
            coord_grid[m - j - 1, i, :] = coords_ordered

    return coord_grid


def plot_regions(regions, coord_grid, coord_range):
    polygons_list = []
    for region in regions:
        region_coords = []
        for idx in region:
            grid_idx = np.unravel_index(idx, shape=coord_grid.shape[:2])
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


if __name__ == '__main__':
    run()
