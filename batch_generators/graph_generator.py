from batch_generators.graph_dataset import GraphDataset


class GraphGenerator:
    def __init__(self, node_features, labels, edge_index, batch_gen_params):
        self.node_features = node_features
        self.labels = labels
        self.edge_index = edge_index

        self.test_size = batch_gen_params["test_size"]
        self.val_ratio = batch_gen_params["val_ratio"]
        self.window_in_len = batch_gen_params["window_in_len"]
        self.window_out_len = batch_gen_params["window_out_len"]
        self.batch_size = batch_gen_params["batch_size"]
        self.shuffle = batch_gen_params["shuffle"]

        self.train_val_size = len(node_features) - self.test_size
        self.val_size = int(self.train_val_size * self.val_ratio)
        self.train_size = self.train_val_size - self.val_size

        self.data_dict = self.__split_data()
        self.dataset_dict = self.__create_sets()

    def __split_data(self):
        data_dict = {
            'train': (self.node_features[:self.train_size], self.labels[:self.train_size]),
            'val': (self.node_features[self.train_size:self.train_val_size],
                    self.labels[self.train_size:self.train_val_size]),
            "train_val": (self.node_features[:self.train_val_size], self.labels[:self.train_val_size]),
            'test': (self.node_features[self.train_val_size:], self.labels[self.train_val_size:])
        }
        return data_dict

    def __create_sets(self):
        graph_dataset = {}
        for i in ['train', 'val', 'train_val', 'test']:
            node_f, labels = self.data_dict[i]
            dataset = GraphDataset(node_features=node_f,
                                   labels=labels,
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
