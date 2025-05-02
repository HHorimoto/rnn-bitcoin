import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, rnn_name, in_size=4, out_size=2, hidden_size=32):
        super(Net, self).__init__()

        self.rnn_name = rnn_name

        if self.rnn_name == "RNN":
            self.rnn = nn.RNNCell(input_size=in_size, hidden_size=hidden_size)
        elif rnn_name == "LSTM":
          self.rnn = nn.LSTMCell(input_size=in_size, hidden_size=hidden_size)
        elif rnn_name == "GRU":
          self.rnn = nn.GRUCell(input_size=in_size, hidden_size=hidden_size)
        
        self.linear = nn.Linear(in_features=hidden_size, out_features=out_size)

    def forward(self, x, hx, cx):
        if self.rnn_name == "RNN":
            hx = self.rnn(x, hx)
            h = self.linear(hx)
            return h, hx, cx
        elif self.rnn_name == "LSTM":
            hx, cx = self.rnn(x, (hx, cx))
            h = self.linear(hx)
            return h, hx, cx
        elif self.rnn_name == "GRU":
            hx = self.rnn(x, hx)
            h = self.linear(hx)
            return h, hx, cx