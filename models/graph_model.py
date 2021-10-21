import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric_temporal.nn as pyg_t_nn
import torch.nn.functional as F


class GraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_layers, filter_sizes, bias, node_count, normalization, device):
        super(GraphModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.filter_sizes = filter_sizes
        self.bias = bias
        self.node_count = node_count
        self.normalization = normalization
        self.device = device

        conv_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            conv_list += [pyg_t_nn.GConvGRU(in_channels=cur_input_dim,
                                            out_channels=hidden_dims[i],
                                            K=filter_sizes[i],
                                            bias=bias,
                                            normalization=normalization)]
        self.memory_unit = nn.ModuleList(conv_list)

        mu_weights = []
        for i in range(node_count):
            mu_weights.append(nn.Linear(in_features=hidden_dims[-1], out_features=2, bias=True))
        self.mu_weights = nn.ModuleList(mu_weights)

        sigma_weights = []
        for i in range(node_count):
            sigma_weights.append(nn.Sequential(nn.Linear(in_features=hidden_dims[-1], out_features=2, bias=True),
                                               nn.Softplus()))
        self.sigma_weights = nn.ModuleList(sigma_weights)

    def forward(self, in_tensor, edge_index, edge_weight=None):
        # in_tensor has the shape of (B, T, M, D)
        # where M is node count, D is input dim,
        # B is batch size, and T is time step
        batch_size, window_in, node_count = in_tensor.shape[:3]

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

        # forward node linear layers
        mu_outputs = []
        sigma_outputs = []
        for i in range(self.node_count):
            mu = self.mu_weights[i](output[:, i])
            sigma = self.sigma_weights[i](output[:, i])
            mu_outputs.append(mu)
            sigma_outputs.append(sigma)

        # create mixing coefficients
        batch_idx = torch.ones((batch_size, node_count), dtype=int) * torch.arange(batch_size).unsqueeze(dim=1)
        mix_val = pyg_nn.global_mean_pool(output.view(-1, self.hidden_dims[-1]), batch_idx.flatten().to(self.device))
        mix_coeff = F.softmax(mix_val, dim=1)

        return mu_outputs, sigma_outputs, mix_coeff

    def init_bias(self, bias_values):
        for i in range(len(bias_values)):
            self.mu_weights[i].bias = nn.Parameter(bias_values[i])


if __name__ == '__main__':
    import pickle as pkl
    import numpy as np

    with open("../temp/graph/data_dump_24_5000/node_features.pkl", "rb") as f:
        node_features = pkl.load(f)

    with open("../temp/graph/data_dump_24_5000/edge_index.pkl", "rb") as f:
        edge_index = pkl.load(f)

    in_tensor = []
    for i in range(5):
        in_tensor.append(node_features[i:i + 10])
    in_tensor = np.stack(in_tensor)

    node_count = node_features.shape[1]
    model = GraphModel(input_dim=3,
                       hidden_dims=[30, 20, 10],
                       num_layers=3,
                       filter_sizes=[3, 3, 3],
                       bias=True,
                       node_count=node_features.shape[1],
                       normalization="sym",
                       device="cpu")

    bias_values = []
    for i in range(node_count):
        bias_values.append(torch.tensor([-87.52424242, 41.68325], requires_grad=True).float())
    model.init_bias(bias_values)

    in_tensor = torch.from_numpy(in_tensor).float()
    edge_index = torch.from_numpy(edge_index)
    mu, sigma, mix_coeff = model(in_tensor, edge_index)
    print()
