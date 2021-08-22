import itertools

import pandas as pd
import numpy as np

from minisom import MiniSom
import matplotlib.pyplot as plt


def run():
    data = pd.read_csv("simulation_data.csv")
    centers = data[["cx", "cy"]].values

    plt.figure()
    plt.scatter(centers[:, 0], centers[:, 1])
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()

    neuron_count_1 = 4
    neuron_count_2 = 5

    som = MiniSom(neuron_count_1, neuron_count_2, 2, sigma=0.3, learning_rate=0.1)
    # som.pca_weights_init(centers)
    # initialise the weights
    mins, maxs = centers.min(axis=0), centers.max(axis=0)
    y_axis = np.linspace(mins[1], maxs[1], neuron_count_2)
    x_axis = np.linspace(mins[0], maxs[0], neuron_count_1)
    init_points = np.array(list(itertools.product(x_axis, y_axis)))
    som._weights = init_points.reshape(neuron_count_1, neuron_count_2, 2)

    plt.figure()
    plt.scatter(init_points[:, 0], init_points[:, 1])
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()

    som.train(centers, 5000000, verbose=True)

    distances = som._distance_from_weights(centers)

    assigned_dict = {}
    sorted_distances = np.argsort(distances.ravel())
    for idx in sorted_distances:
        p_idx, n_idx = np.unravel_index(idx, shape=distances.shape)
        if p_idx not in assigned_dict.values():
            assigned_dict[n_idx] = p_idx

    unassigned_neurons = [i for i in range(neuron_count_1 * neuron_count_2) if i not in assigned_dict.keys()]
    for n in unassigned_neurons:
        assigned_dict[n] = np.argmin(distances[:, n])

    points = np.zeros((neuron_count_1, neuron_count_2))
    for n_idx, p_idx in assigned_dict.items():
        x_idx, y_idx = np.unravel_index(n_idx, shape=(neuron_count_1, neuron_count_2))
        points[x_idx, y_idx] = p_idx

    print(points)

    weights = som.get_weights().reshape(-1, 2)
    plt.figure()
    plt.scatter(weights[:, 0], weights[:, 1])
    for i in range(len(weights)):
        plt.text(weights[i, 0], weights[i, 1], f"{np.unravel_index(i, shape=(4, 5))}")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()


if __name__ == '__main__':
    run()
