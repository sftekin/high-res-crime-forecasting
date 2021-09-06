import itertools

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

from graph2grid.create_sim_data import plot_poly
from graph import Graph


def divide_into_regions(in_grid, threshold, r, c):
    grid = in_grid[r[0]:r[1], c[0]:c[1]]

    if np.sum(grid) <= 200 or grid.shape <= (2, 2):
        return [[r, c]]
    else:
        split_ids = split_regions(in_grid, r, c)
        region_ids = []
        for r, c in split_ids:
            region_id = divide_into_regions(in_grid, threshold, r, c)
            region_ids += region_id

        return region_ids


def split_regions(in_grid, r, c):
    m, n = r[1] - r[0], c[1] - c[0]
    pos_m, pos_n = [m // 2], [n // 2]
    if m % 2 > 0:
        pos_m.append(m//2 + 1)
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


def run():
    coord_range = [[41.60, 42.05], [-87.9, -87.5]]

    with open("grid.npy", "rb") as f:
        grid = np.load(f)
        grid = np.squeeze(grid)
    grid_sum = np.sum(grid, axis=0)

    coord_grid = create_coord_grid(in_grid=grid_sum, coord_range=coord_range)  # M, N, 4, 2
    grid_center_locs = np.mean(coord_grid, axis=2)

    m, n = grid_sum.shape
    init_r, init_c = (0, m), (0, n)
    regions = divide_into_regions(grid_sum, threshold=50, r=init_r, c=init_c)
    plot_regions(regions, coord_grid, coord_range)
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


def plot_regions(regions, coord_grid, coord_range, colorize=False):
    polygons_list = []
    for r, c in regions:
        region_pts = coord_grid[r[0]:r[1], c[0]:c[1]]
        region_pts = region_pts.reshape(-1, 2)
        lon_min, lon_max = np.min(region_pts[:, 0]), np.max(region_pts[:, 0])
        lat_min, lat_max = np.min(region_pts[:, 1]), np.max(region_pts[:, 1])

        coords = np.array(list(itertools.product([lon_min, lon_max], [lat_min, lat_max])))
        coords_ordered = coords[[0, 1, 3, 2], :]
        polygon = Polygon(coords_ordered)
        polygons_list.append(polygon)

    fig, ax = plt.subplots()
    for i, poly in enumerate(polygons_list):
        if colorize:
            color = f"C{i}"
        else:
            color = "w"
        plot_poly(poly, ax, face_color=color)
    plt.xlim(*coord_range[1])
    plt.ylim(*coord_range[0])
    plt.show()


if __name__ == '__main__':
    run()
