from __future__ import print_function
import torch
import torch.utils.data
from torch import nn


# LSTM
# ref: https://www.curiousily.com/posts/time-series-forecasting-with-lstm-for-daily-coronavirus-cases/

# LSTM
# ref: https://pytorch.org/docs/stable/nn.html
class LSTM(nn.Module):

    def __init__(self, input_dim=34, hidden_dim=300, output_dim=1, seq_len=7, num_layers=3, dropout=0.2):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=dropout)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input):
        # Reset the hidden state
        self.hidden = (
            torch.zeros(self.num_layers, len(input), self.hidden_dim).cuda(),
            torch.zeros(self.num_layers, len(input), self.hidden_dim).cuda()
        )
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        # Get Only the Last Step
        last_time_step = lstm_out.view(len(input), self.seq_len, self.hidden_dim)[:, -1, :]
        y = self.linear(last_time_step)
        return y


