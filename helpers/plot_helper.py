import seaborn as sns
import matplotlib.pyplot as plt


def plot_hist_dist(in_arr, x_label="", title="", bins=25, save_path=None):
    plt.style.use('seaborn')
    plt.figure()
    sns.histplot(in_arr, bins=bins, kde=True)
    plt.xlabel(x_label)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=200)
