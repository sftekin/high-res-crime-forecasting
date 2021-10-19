import torch.nn as nn
from torch_geometric_temporal.nn import GConvGRU
from torch.nn import Linear
import torch.functional as F


class GraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, filter_size, bias, normalization, window_out, device):
        super(GraphModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.filter_size = filter_size
        self.bias = bias
        self.normalization = normalization
        self.window_out = window_out
        self.device = device

        self.gconv_gru = GConvGRU()
