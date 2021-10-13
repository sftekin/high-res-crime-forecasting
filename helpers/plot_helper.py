import seaborn as sns
import matplotlib.pyplot as plt


def plot_hist_dist(in_arr, x_label="", bins=25, save_path=None):
    plt.style.use('seaborn')
    plt.figure()
    sns.histplot(in_arr, bins=bins, kde=True)
    plt.xlabel(x_label)
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()





