# Define networks
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PosWLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b``

    Same as nn.Linear, except that weight matrix is constrained to be non-negative
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(PosWLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.with_Tanh = kwargs.get('with_Tanh', False)
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # weight is non-negative
                # weight is non-negative
        output = F.linear(input, torch.abs(self.weight), self.bias)
        if self.with_Tanh:
            output = F.linear(input, F.relu(self.weight), self.bias)
        return output
    
    
class EIRecLinear(nn.Module):
    r"""
    Recurrent E-I Linear transformation with flexible Dale and block constraints.

    Args:
        hidden_size: int, total hidden neurons
        e_prop: float in (0,1), proportion of excitatory units
        mode: 'none', 'dense', 'block'
        block_groups: int, only used if mode='block'
    """
    __constants__ = ['bias', 'hidden_size', 'e_prop']

    def __init__(self, hidden_size, e_prop, mode='none', block_groups=2, bias=True, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.e_prop = e_prop
        self.e_size = int(e_prop * hidden_size)
        self.i_size = hidden_size - self.e_size
        self.mode = mode
        self.block_groups = block_groups

        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        
        # Dale mask
        if self.mode in ['dense', 'block']:
            mask = np.tile([1]*self.e_size+[-1]*self.i_size, (hidden_size,1))
            np.fill_diagonal(mask, 0)
            self.register_buffer('dale_mask', torch.tensor(mask, dtype=torch.float32))
        else:
            mask = np.ones((hidden_size, hidden_size))
            np.fill_diagonal(mask, 0)
            self.register_buffer('dale_mask', torch.tensor(mask, dtype=torch.float32))
        # Block mask
        if self.mode == 'block':
            group_size = self.e_size // self.block_groups
            block_mask = torch.ones(hidden_size, hidden_size)
            # self.register_buffer('block_mask', torch.ones(hidden_size, hidden_size))

            for g in range(self.block_groups):
                start = g * group_size
                end = (g+1)*group_size if g < self.block_groups-1 else self.e_size
                block_mask[start:end, :start] = 0
                block_mask[start:end, end:self.e_size] = 0
            # allow I connections everywhere
            block_mask[:, self.e_size:] = 1
            block_mask[self.e_size:, :] = 1
            self.register_buffer('block_mask', block_mask)
        else:
            self.register_buffer('block_mask', torch.ones(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.mode in ['dense', 'block']:
            # Scale E columns by E/I ratio
            self.weight.data[:, :self.e_size] /= (self.e_size / self.i_size)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def effective_weight(self):
        if self.mode == 'none':
            return self.weight
        else:
            return torch.abs(self.weight) * self.dale_mask* self.block_mask

    def forward(self, input):
        return F.linear(input, self.effective_weight(), self.bias)


class EIRNN(nn.Module):
    """E-I RNN.
    
    Reference:
        Song, H.F., Yang, G.R. and Wang, X.J., 2016.
        Training excitatory-inhibitory recurrent neural networks
        for cognitive tasks: a simple and flexible framework.
        PLoS computational biology, 12(2).

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: (seq_len, batch, input_size)
        hidden: (batch, hidden_size)
        e_prop: float between 0 and 1, proportion of excitatory neurons
    """

    def __init__(self, input_size, hidden_size, dt=None,
                 e_prop=0.8, sigma_rec=0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.e_size = int(hidden_size * e_prop)
        self.i_size = hidden_size - self.e_size
        self.num_layers = 1
        self.tau = 100
        self.mode = kwargs.get('mode', 'none')
        self.noneg = kwargs.get('noneg', True)
        self.with_Tanh = kwargs.get('with_Tanh', False)
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        # Recurrent noise
        self._sigma_rec = np.sqrt(2*alpha) * sigma_rec
        if self.noneg and self.mode != 'none':
            self.input2h = PosWLinear(input_size, hidden_size, with_Tanh=self.with_Tanh)
        else:
            self.input2h = nn.Linear(input_size, hidden_size)

        if self.mode == 'dense':
            self.h2h = EIRecLinear(hidden_size, e_prop=e_prop, mode='dense')
        elif self.mode == 'block':
            self.h2h = EIRecLinear(hidden_size, e_prop=e_prop, mode='block', block_groups=2)
        else:  # 'none'
            self.h2h = EIRecLinear(hidden_size, e_prop=e_prop, mode='none')
        # self.h2h = EIRecLinear(hidden_size, e_prop=0.8)

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return (torch.zeros(batch_size, self.hidden_size).to(input.device),
                torch.zeros(batch_size, self.hidden_size).to(input.device))

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        state, output = hidden
        total_input = self.input2h(input) + self.h2h(output)
        state = state * self.oneminusalpha + total_input * self.alpha
        state += self._sigma_rec * torch.randn_like(state)
        output = torch.relu(state)
        return state, output

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden[1])

        output = torch.stack(output, dim=0)
        return output, hidden


class Net(nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Excitatory-inhibitory RNN
        self.mode = kwargs.get('mode', 'none')
        self.noneg = kwargs.get('noneg', True)
        self.with_Tanh = kwargs.get('with_Tanh', False)
        self.rnn = EIRNN(input_size, hidden_size, **kwargs)

        if self.noneg and self.mode != 'none':
            self.fc = PosWLinear(self.rnn.e_size, output_size, with_Tanh=self.with_Tanh)
        else:
            if self.mode == 'none':
                # If no Dale's law, use all hidden neurons
                self.fc = nn.Linear(self.rnn.hidden_size, output_size)
            else:
                # If Dale's law, use only excitatory neurons
                self.fc = nn.Linear(self.rnn.e_size, output_size)

        # self.fc = PosWLinear(self.rnn.e_size, output_size)
        # self.fc = nn.Linear(self.rnn.e_size, output_size)

    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        if self.mode == 'none':
            # If no Dale's law, use all hidden neurons
            rnn_e = rnn_activity
        else:
            rnn_e = rnn_activity[:, :, :self.rnn.e_size]
        out = self.fc(rnn_e)
        return out, rnn_activity


## utility functions
def compute_loss(y_pred, y_true, mask):
    return ((y_pred - y_true) ** 2 * mask).sum() / mask.sum()

def accuracy(y_pred, y_true, mask):
    # 只看决策期的平均输出
    pred = torch.argmax((y_pred * mask).mean(dim=0), dim=1)
    true = torch.argmax((y_true * mask).mean(dim=0), dim=1)
    return (pred == true).float().mean().item()

