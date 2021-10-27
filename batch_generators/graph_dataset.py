import torch
from batch_generators.dataset import Dataset
from helpers.graph_helper import flatten_labels


class GraphDataset(Dataset):
    def __init__(self, in_data, labels, regions, edge_index,  window_in_len, window_out_len, batch_size, shuffle):
        super(GraphDataset, self).__init__(in_data, labels, window_in_len, window_out_len, batch_size, shuffle)
        self.regions = regions
        self.edge_index = edge_index
        self.labels = flatten_labels(labels, regions)

    def __next__(self):
        all_data = super().__next__()

        # generate
        for i in range(self.num_iter):
            node_features = torch.from_numpy(all_data[i][0])
            labels = torch.from_numpy(all_data[i][1]).squeeze()
            edge_index = torch.from_numpy(self.edge_index)
            yield node_features, labels, edge_index
