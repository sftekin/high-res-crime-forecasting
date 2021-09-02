import numpy as np
import matplotlib.pyplot as plt


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

        filled_grid = np.zeros(len(sorted_cells))
        for i, region in enumerate(regions):
            filled_grid[np.array(region)] = i

    print()


def sum_neighbours(in_grid, idx, order=1):
    kernel = np.zeros(in_grid.shape)
    i, j = idx
    kernel[i-order:i+order+1, j-order:j+order+1] = 1
    count = np.sum(kernel * in_grid)
    neighbours = np.where(kernel.flatten())[0]
    return count, neighbours



if __name__ == '__main__':
    run()
