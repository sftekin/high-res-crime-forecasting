import numpy as np

from batch_generators.graph_dataset import GraphDataset
from normalizers.batch_normalizer import BatchNormalizer


class GraphGenerator:
    def __init__(self, node_features, labels, regions, edge_index, batch_gen_params):
        self.node_features = node_features
        self.labels = labels
        self.edge_index = edge_index
        self.regions = regions

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
            self.input_data = self.normalizer.norm(x=self.node_features)

        self.train_val_size = len(node_features) - self.test_size
        self.val_size = int(self.train_val_size * self.val_ratio)
        self.train_size = self.train_val_size - self.val_size

        self.data_ids = np.arange(len(node_features))
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
            dataset = GraphDataset(node_features=self.node_features[data_ids],
                                   labels=self.labels[data_ids],
                                   regions=self.regions,
                                   edge_index=self.edge_index,
                                   window_in_len=self.window_in_len,
                                   window_out_len=self.window_out_len,
                                   batch_size=self.batch_size,
                                   shuffle=self.shuffle)
            graph_dataset[i] = dataset

        return graph_dataset

    def num_iter(self, dataset_name):
        return self.dataset_dict[dataset_name].num_iter

    def generate(self, dataset_name):
        selected_loader = self.dataset_dict[dataset_name]
        yield from selected_loader.__next__()
