import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_hist_dist(in_arr, x_label="", title="", bins=25, save_path=None):
    plt.style.use('seaborn')
    plt.figure()
    sns.histplot(in_arr, bins=bins, kde=True)
    plt.xlabel(x_label)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_preds():
    plot_background(ax=ax[row_idx, col_idx], coord_range=coord_range, M=M, N=N)
    ax[row_idx, col_idx].imshow(X=grid_sum,
                                cmap=plt.cm.get_cmap('Reds', 10),
                                interpolation='nearest',
                                extent=[*coord_range[1], *coord_range[0]],
                                alpha=0.7)
    ax[row_idx, col_idx].set_title(crime_type, fontsize=22)


def plot_background(ax, coord_range, M=10, N=11):
    background_path = "background/chicago.png"
    x_ticks = np.linspace(coord_range[1][0], coord_range[1][1], N + 1)
    y_ticks = np.linspace(coord_range[0][0], coord_range[0][1], M + 1)

    x_tick_labels = ["{:2.3f}".format(long) for long in x_ticks]
    y_tick_labels = ["{:2.3f}".format(lat) for lat in y_ticks]

    ax.set_xticks(ticks=x_ticks)
    ax.set_xticklabels(labels=x_tick_labels,
                  rotation=30,
                  size=12)
    ax.set_yticks(ticks=y_ticks)
    ax.set_yticklabels(labels=y_tick_labels,
                    size=12)

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


    img = mpimg.imread(background_path)
    ax.imshow(img,
              interpolation='bilinear',
              extent=[*coord_range[1], *coord_range[0]])
    ax.grid(True)
    return ax


