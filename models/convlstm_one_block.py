import torch
import torch.nn as nn

from models.convlstm import ConvLSTMCell


class ConvLSTMOneBlock(nn.Module):
    def __init__(self, input_dim, input_size, hidden_dims, kernel_sizes, window_in, window_out, num_layers, device,
                 bias):
        super(ConvLSTMOneBlock, self).__init__()
        self.device = device
        self.input_size = input_size
        self.height, self.width = self.input_size

        self.window_in = window_in
        self.window_out = window_out
        self.num_layers = num_layers

        # Defining block
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list += [ConvLSTMCell(input_size=(self.height, self.width),
                                       input_dim=cur_input_dim,
                                       hidden_dim=hidden_dims[i],
                                       kernel_size=kernel_sizes[i],
                                       bias=bias,
                                       device=self.device,
                                       peephole_con=False)]
        self.block = nn.ModuleList(cell_list)

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.block[i].init_hidden(batch_size))
        return init_states

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        b, t, d, m, n = x.shape
        hidden = self.init_hidden(batch_size=b)

        layer_output_list = []
        layer_state_list = []
        cur_layer_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden[layer_idx]
            output_inner = []
            for t in range(self.window_out):
                h, c = self.block[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                             cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            layer_state_list.append([h, c])

        block_output = layer_output_list[-1]

        output = []
        for i in range(self.window_out):
            out = torch.sigmoid(block_output[:, i])
            output.append(out)
        output = torch.stack(output, dim=1)
        output = output.permute(0, 1, 3, 4, 2)

        return output
