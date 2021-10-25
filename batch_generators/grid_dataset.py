import torch
from batch_generators.dataset import Dataset


class GridDataset(Dataset):
    def __init__(self, in_data, labels, window_in_len, window_out_len, batch_size, shuffle):
        super(GridDataset, self).__init__(in_data, labels, window_in_len, window_out_len, batch_size, shuffle)

    def __next__(self):
        all_data = super().__next__()

        # generate
        for i in range(self.num_iter):
            in_data = torch.from_numpy(all_data[i][0])
            labels = torch.from_numpy(all_data[i][1])
            yield in_data, labels
