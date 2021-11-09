import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def run():
    pts = torch.rand(20, 2)

    pts_arr = pts.numpy()
    groups = np.random.randint(low=0, high=4, size=len(pts_arr))
    plt.figure()
    for i, marker in enumerate(["o", "+", "x", "^"]):
        coords = pts_arr[groups == i]
        plt.scatter(coords[:, 0], coords[:, 1], marker=marker, label=f"group_{i}")
    plt.legend()
    plt.show()

    intensities = np.array([.1, .2, .3, .4])
    total_time_step = 3000
    num_events_per_step = np.random.randint(low=100, high=250, size=(total_time_step,))
    events_per_group = []
    for i in range(total_time_step):
        events_per_group.append(num_events_per_step[i] * intensities)
    events_per_group = np.stack(events_per_group)



if __name__ == '__main__':
    run()




