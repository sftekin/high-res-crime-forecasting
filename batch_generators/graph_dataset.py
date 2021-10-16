import torch
import numpy as np


class GraphDataset:
    def __init__(self, node_features, labels, edge_index,  window_in_len, window_out_len, batch_size, shuffle):
        self.node_features = node_features
        self.labels = labels
        self.edge_index = edge_index
        self.window_in_len = window_in_len
        self.window_out_len = window_out_len
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __next__(self):
        all_data = self.__create_buffer()
        self.num_iter = len(all_data)

        # generate
        for i in range(self.num_iter):
            node_features = torch.from_numpy(all_data[i][0])
            labels = torch.from_numpy(all_data[i][1])
            yield node_features, labels, self.edge_index

    def __create_buffer(self):
        total_frame = len(self.node_features)
        all_data, batch_node, batch_label = [], [], []
        j = 0
        for i in range(total_frame-self.window_in_len):
            if j < self.batch_size:
                batch_node.append(self.node_features[i:i+self.window_in_len])
                batch_label.append(self.labels[i+self.window_in_len])
                j += 1
            else:
                all_data.append([np.stack(batch_node, axis=0), batch_label])
                batch_node, batch_label = [], []
                j = 0
        return all_data
