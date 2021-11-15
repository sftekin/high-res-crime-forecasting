import numpy as np


class Dataset:
    def __init__(self, in_data, labels, window_in_len, window_out_len, batch_size, shuffle):
        self.in_data = in_data
        self.labels = labels
        self.window_in_len = window_in_len
        self.window_out_len = window_out_len
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.total_win_len = window_in_len + window_out_len
        self.num_iter = None

    def __next__(self):
        all_data = self._create_buffer()
        self.num_iter = len(all_data)
        return all_data

    def _create_buffer(self):
        total_frame = len(self.in_data)
        all_data, batch_node, batch_label = [], [], []
        j = 0
        for i in range(total_frame-self.total_win_len + 1):
            if j < self.batch_size:
                batch_node.append(self.in_data[i:i+self.window_in_len])
                batch_label.append(self.labels[i+self.window_in_len:i+self.total_win_len])
                j += 1
            else:
                all_data.append([np.stack(batch_node, axis=0),
                                 np.stack(batch_label, axis=0)])
                batch_node = [self.in_data[i:i+self.window_in_len]]
                batch_label = [self.labels[i+self.window_in_len:i+self.total_win_len]]
                j = 1
        if len(batch_node) > 0:
            all_data.append([np.stack(batch_node, axis=0),
                             np.stack(batch_label, axis=0)])
        return all_data
