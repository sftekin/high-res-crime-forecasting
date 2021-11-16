import torch
import torch.nn as nn


class FCLSTM(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, drop_out, device, node_count=True):
        super(FCLSTM, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.device = device
        self.node_count = node_count if node_count is not None else 1

        self.memory_unit = nn.LSTM(input_size=input_dim,
                                   hidden_size=hidden_dim,
                                   num_layers=num_layers,
                                   dropout=drop_out,
                                   batch_first=True)

    def forward(self, in_tensor):
        batch_size, window_in_len, height, width, feat_dim = in_tensor.shape
        in_tensor = in_tensor.reshape(batch_size, window_in_len, -1)

        output, (hn, cn) = self.memory_unit(in_tensor)
        output = output[:, [-1]]
        output = output.reshape(batch_size, 1, height, width, feat_dim)

        return output
