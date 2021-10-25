import numpy as np

from batch_generators.graph_dataset import GraphDataset
from batch_generators.grid_dataset import GridDataset
from normalizers.batch_normalizer import BatchNormalizer


class BatchGenerator:
    def __init__(self, in_data, labels, batch_gen_params, regions=None, edge_index=None):
        self.in_data = in_data
        self.labels = labels
        self.edge_index = edge_index
        self.regions = regions

        self.dataset_name = batch_gen_params["dataset_name"]
        self.test_size = batch_gen_params["test_size"]
        self.val_ratio = batch_gen_params["val_ratio"]
        self.window_in_len = batch_gen_params["window_in_len"]
        self.window_out_len = batch_gen_params["window_out_len"]
        self.batch_size = batch_gen_params["batch_size"]
        self.shuffle = batch_gen_params["shuffle"]
        self.normalize_flag = batch_gen_params["normalize_flag"]
        self.normalize_methods = batch_gen_params["normalize_methods"]
        self.normalization_dims = batch_gen_params["normalization_dims"]

        if self.normalize_flag:
            self.normalizer = BatchNormalizer(normalize_methods=self.normalize_methods,
                                              normalization_dims=self.normalization_dims)
            self.input_data = self.normalizer.norm(x=self.in_data)

        self.train_val_size = len(in_data) - self.test_size
        self.val_size = int(self.train_val_size * self.val_ratio)
        self.train_size = self.train_val_size - self.val_size

        self.data_ids = np.arange(len(in_data))
        self.data_dict = self.__split_data()
        self.dataset_dict = self.__create_sets()

    def __split_data(self):
        data_dict = {
            'train': self.data_ids[:self.train_size],
            'val': self.data_ids[self.train_size:self.train_val_size],
            "train_val": self.data_ids[:self.train_val_size],
            'test': self.data_ids[self.train_val_size:]
        }
        return data_dict

    def __create_sets(self):
        graph_dataset = {}
        for i in ['train', 'val', 'train_val', 'test']:
            data_ids = self.data_dict[i]
            if self.dataset_name == "graph":
                dataset = GraphDataset(in_data=self.in_data[data_ids],
                                       labels=self.labels[data_ids],
                                       regions=self.regions,
                                       edge_index=self.edge_index,
                                       window_in_len=self.window_in_len,
                                       window_out_len=self.window_out_len,
                                       batch_size=self.batch_size,
                                       shuffle=self.shuffle)
            elif self.dataset_name == "grid":
                dataset = GridDataset(in_data=self.in_data[data_ids],
                                      labels=self.labels[data_ids],
                                      window_in_len=self.window_in_len,
                                      window_out_len=self.window_out_len,
                                      batch_size=self.batch_size,
                                      shuffle=self.shuffle)
            else:
                raise RuntimeError("dataset name can only be 'graph' or 'grid'")
            graph_dataset[i] = dataset

        return graph_dataset

    def num_iter(self, dataset_name):
        return self.dataset_dict[dataset_name].num_iter

    def generate(self, dataset_name):
        selected_loader = self.dataset_dict[dataset_name]
        yield from selected_loader.__next__()
