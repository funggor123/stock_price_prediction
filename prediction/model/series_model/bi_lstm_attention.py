import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils


# Encode the Text into a context vector
# https://www.aclweb.org/anthology/P16-2034
class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=300, output_dim=1, num_layers=1, dropout=0.2):
        super(BiLSTMAttention, self).__init__()

        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True,
                            dropout=dropout, num_layers=num_layers)

        self.attention_linear = nn.Linear(hidden_dim, 1)
        self.linear_hidden = nn.Linear(self.hidden_size, output_dim)

    # Model Structure
    # Embedding -> LSTM -> Attention -> Linear
    def forward(self, input):
        # Reset the hidden state
        self.hidden = (
            torch.zeros(self.num_layers*2, len(input), self.hidden_dim).cuda(),
            torch.zeros(self.num_layers*2, len(input), self.hidden_dim).cuda()
        )
        out, self.hidden = self.lstm(input, self.hidden)
        out = self.attention(out, self.attention_linear)
        context_vector = self.linear_hidden(out)
        return context_vector

    def attention(self, x, attention_linear):
        hidden_states = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:]
        hidden_states = torch.tanh(hidden_states)
        e = attention_linear(hidden_states)
        a = F.softmax(e, dim=1)
        out = torch.sum(torch.mul(hidden_states, a), 1)
        return out
