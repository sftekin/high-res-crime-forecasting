import pandas as pd
import numpy as np

from minisom import MiniSom
import matplotlib.pyplot as plt


def run():
    dpi = 200
    first_neuron_c = 10
    second_neuron_c = 10

    data = pd.read_csv("data.csv")
    X = data[["x1", "x2"]].values
    y = data["y"].values
    y[y < 0] = 0

    som = MiniSom(first_neuron_c, second_neuron_c, 2, sigma=0.3, learning_rate=0.5)
    # som.pca_weights_init(X)
    som.train(X, 5000)

    plt.figure()
    plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
    plt.colorbar()

    markers = ['o', 'x']
    colors = ['C0', 'C1']
    for cnt, xx in enumerate(X):
        w = som.winner(xx)  # getting the winner
        # palce a marker on the winning position for the sample xx
        plt.plot(w[0] + .5, w[1] + .5, markers[y[cnt]], markerfacecolor='None',
                 markeredgecolor=colors[y[cnt] - 1], markersize=12, markeredgewidth=2)
    plt.savefig(f"figures/som_plot_{first_neuron_c}_{second_neuron_c}.png", dpi=dpi)
    plt.show()

    neurons = []
    for d in X:
        w = som.winner(d)
        neurons.append(w)
    neurons = np.array(neurons)

    plt.figure()

    plt.figure()
    plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
    plt.colorbar()

    for label in [0, 1]:
        y_idx = y == label

        plt.scatter(neurons[y_idx, 0] + .5 + (np.random.rand(np.sum(y_idx))-.5)*.8,
                    neurons[y_idx, 1] + .5 + (np.random.rand(np.sum(y_idx))-.5)*.8, s=50,
                    c=colors[label])
    plt.savefig(f"figures/som_scatter_{first_neuron_c}_{second_neuron_c}.png", dpi=dpi)
    plt.show()


if __name__ == '__main__':
    run()
