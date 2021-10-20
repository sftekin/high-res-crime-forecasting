import torch
import torch.nn as nn
from torch_geometric_temporal.nn import GConvGRU


class GraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_layers, filter_sizes, bias, normalization, device):
        super(GraphModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.filter_sizes = filter_sizes
        self.bias = bias
        self.normalization = normalization
        self.device = device

        conv_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            conv_list += [GConvGRU(in_channels=cur_input_dim,
                                   out_channels=hidden_dims[i],
                                   K=filter_sizes[i],
                                   bias=bias,
                                   normalization=normalization)]
        self.memory_unit = nn.ModuleList(conv_list)
        self.mu = nn.Linear(in_features=hidden_dims[-1], out_features=2, bias=True)
        self.sigma = nn.Sequential(nn.Linear(in_features=hidden_dims[-1], out_features=2, bias=True),
                                   nn.Softplus())

    def forward(self, in_tensor, edge_index, edge_weight=None):
        # in_tensor has the shape of (B, T, M, D)
        # where M is node count, D is input dim,
        # B is batch size, and T is time step
        batch_size, window_in = in_tensor.shape[:2]

        # forward memory unit
        output_batch = []
        layer_output = None
        for i in range(batch_size):
            cur_input = in_tensor[i]
            for layer_idx in range(self.num_layers):
                h = None  # it will start zeros anyway
                h_inner = []
                for t in range(window_in):
                    h = self.memory_unit[layer_idx](X=cur_input[t], H=h,
                                                    edge_index=edge_index, edge_weight=edge_weight)
                    h_inner.append(h)
                layer_output = torch.stack(h_inner)
                cur_input = layer_output
            output_batch.append(layer_output[-1])
        output = torch.stack(output_batch)  # B, M, D'

        mu = self.mu(output)
        sigma = self.sigma(output)

        return mu, sigma

    def init_bias(self, bias_value):
        self.mu.bias = nn.Parameter(bias_value)


if __name__ == '__main__':
    import pickle as pkl
    import numpy as np

    with open("../temp/graph/data_dump_24_5000/node_features.pkl", "rb") as f:
        node_features = pkl.load(f)

    with open("../temp/graph/data_dump_24_5000/edge_index.pkl", "rb") as f:
        edge_index = pkl.load(f)

    in_tensor = []
    for i in range(5):
        in_tensor.append(node_features[i:i+10])
    in_tensor = np.stack(in_tensor)

    model = GraphModel(input_dim=3,
                       hidden_dims=[30, 20, 10],
                       num_layers=3,
                       filter_sizes=[3, 3, 3],
                       bias=True,
                       normalization="sym",
                       device="cpu")
    bias = torch.tensor([-87.52424242,  41.68325], requires_grad=True).float()
    model.init_bias(bias)

    in_tensor = torch.from_numpy(in_tensor).float()
    edge_index = torch.from_numpy(edge_index)
    mu, sigma = model(in_tensor, edge_index)
    print()

