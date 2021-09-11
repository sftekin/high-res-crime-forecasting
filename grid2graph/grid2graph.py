

from graph.graph2grid import *
# from graph import Graph


def run():
    coord_range = [[41.60, 42.05], [-87.9, -87.5]]

    with open("grid.npy", "rb") as f:
        grid = np.load(f)
        grid = np.squeeze(grid)
    grid_sum = np.sum(grid, axis=0)


if __name__ == '__main__':
    run()
