import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, bias, activations):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bias = bias
        self.activations = activations
        self.input_dim = self.input_dim

        self.activation_functions = {
            "relu": torch.relu,
            "tanh": torch.tanh,
            "softplus": self.softplus,
            "sigmoid": F.sigmoid,
            "softmax": nn.Softmax(dim=1)
        }

        self.mlp = []
        for i in range(num_layers-1):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            self.mlp.append(nn.Linear(cur_input_dim, self.hidden_dim[i], self.bias))
        self.mlp.append(nn.Linear(self.hidden_dim[-1], self.output_dim, self.bias))
        self.mlp = nn.ModuleList(self.mlp)

    def forward(self, x):
        batch_size = x.shape[0]
        out = x.view(batch_size, -1)
        for i in range(self.num_layers):
            out = self.activation_functions[self.activations[i]](self.mlp[i](out))
        out = out.unsqueeze(1)
        return out

    @staticmethod
    def softplus(x):
        return torch.log(1 + torch.exp(x))
