import torch
import numpy as np
from batch_generators.dataset import Dataset


class GraphDataset(Dataset):
    def __init__(self, in_data, labels, regions, edge_index,  window_in_len, window_out_len, batch_size, shuffle):
        super(GraphDataset, self).__init__(in_data, labels, window_in_len, window_out_len, batch_size, shuffle)
        self.regions = regions
        self.edge_index = edge_index
        self.labels = self.flatten_labels()

    def __next__(self):
        all_data = super().__next__()

        # generate
        for i in range(self.num_iter):
            node_features = torch.from_numpy(all_data[i][0])
            labels = torch.from_numpy(all_data[i][1])
            edge_index = torch.from_numpy(self.edge_index)
            yield node_features, labels, edge_index

    def flatten_labels(self):
        time_len = len(self.labels)
        flattened_labels = []
        for r, c in self.regions:
            flatten_arr = self.labels[:, r[0]:r[1], c[0]:c[1]].reshape(time_len, -1)
            flattened_labels.append(flatten_arr)
        f_labels = np.concatenate(flattened_labels, axis=1)
        return f_labels
