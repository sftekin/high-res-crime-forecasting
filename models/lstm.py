import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, drop_out, device, node_count=True):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.device = device
        self.node_count = node_count if node_count is not None else 1

        self.memory_unit = nn.LSTM(input_size=input_dim,
                                   hidden_size=hidden_dim,
                                   num_layers=num_layers,
                                   dropout=drop_out,
                                   batch_first=False)

        self.mu = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=64, bias=True),
                                nn.Linear(in_features=64, out_features=2, bias=True),
                                nn.Sigmoid())
        self.sigma = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=64, bias=True),
                                   nn.Linear(in_features=64, out_features=2, bias=True),
                                   nn.Sigmoid())

    def forward(self, in_tensor, **kwargs):
        batch_size = in_tensor.shape[0]

        batch_out = []
        for i in range(batch_size):
            node_out, _ = self.memory_unit(in_tensor[i])
            batch_out.append(node_out[-1])
        output = torch.stack(batch_out)

        # forward node linear layers
        mu_outputs = []
        sigma_outputs = []
        for batch_id in range(batch_size):
            mu_outputs.append(self.mu(output[batch_id]))
            sigma_outputs.append(self.sigma(output[batch_id]) * 0.5)
        mu_outputs = torch.stack(mu_outputs)
        sigma_outputs = torch.stack(sigma_outputs)

        return mu_outputs, sigma_outputs
