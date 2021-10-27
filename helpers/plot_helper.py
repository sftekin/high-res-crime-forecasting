import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import matplotlib.cm as cm


def plot_hist_dist(in_arr, x_label="", title="", bins=25, save_path=None):
    plt.style.use('seaborn')
    plt.figure()
    sns.histplot(in_arr, bins=bins, kde=True)
    plt.xlabel(x_label)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def _plot_2d(in_grid, title, coord_range, grid_size):
    # sum_arr = np.sum(in_grid, axis=0)

    fig, ax = plt.subplots(figsize=(10, 15))
    _plot_background(ax=ax, coord_range=coord_range, grid_size=grid_size)

    # create your own custom color
    color_array = plt.cm.get_cmap('Reds')(range(1000))
    color_array[:, -1] = np.linspace(0.6, 1, 1000)
    map_object = LinearSegmentedColormap.from_list(name='fading_red', colors=color_array)
    plt.register_cmap(cmap=map_object)

    ax.imshow(X=in_grid,
              cmap="fading_red",
              interpolation='nearest',
              extent=[*coord_range[1], *coord_range[0]])
    ax.set_title(title, fontsize=22)

    plt.tight_layout()
    plt.show()

    # dir_path = os.path.join(self.figures_dir, "2d")
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)
    # save_path = os.path.join(dir_path, f"{title}.png")
    # plt.savefig(save_path, dpi=250, bbox_inches='tight')


def _plot_background(ax, coord_range, grid_size, tick_labels=False, remove_grid=True):
    m, n = grid_size
    background_path = "eda/background/chicago.png"
    x_ticks = np.linspace(coord_range[1][0], coord_range[1][1], n + 1)
    y_ticks = np.linspace(coord_range[0][0], coord_range[0][1], m + 1)
    ax.set_xticks(ticks=x_ticks)
    ax.set_yticks(ticks=y_ticks)

    if tick_labels:
        x_tick_labels = ["{:2.3f}".format(long) for long in x_ticks]
        y_tick_labels = ["{:2.3f}".format(lat) for lat in y_ticks]
        ax.set_xticklabels(labels=x_tick_labels, rotation=30, size=12)
        ax.set_yticklabels(labels=y_tick_labels, size=12)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    if remove_grid:
        plt.axis("off")

    img = mpimg.imread(background_path)
    ax.imshow(img, interpolation='bilinear', extent=[*coord_range[1], *coord_range[0]])
    ax.grid(True)

    return ax


def _plot_3d(in_grid, title, coord_range, grid_size):
    m, n = grid_size
    x = np.linspace(coord_range[1][0], coord_range[1][1], n)
    y = np.linspace(coord_range[0][0], coord_range[0][1], m)
    xx, yy = np.meshgrid(x, y)

    ax = Axes3D(plt.figure())
    ax.plot_surface(xx, yy, np.flip(in_grid, axis=1), cmap=plt.cm.viridis, cstride=1, rstride=1)
    ax.set_xticks(x[::10])
    ax.set_yticks(y[::10])
    ax.set_xticklabels(np.round(x[::10], decimals=2))
    ax.set_yticklabels(np.round(y[::10], decimals=2))
    ax.view_init(30, 90)

    # dir_path = os.path.join(self.figures_dir, "3d", "surface")
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)
    # save_path = os.path.join(dir_path, f"{title}.png")
    # plt.savefig(save_path, dpi=250, bbox_inches='tight')

