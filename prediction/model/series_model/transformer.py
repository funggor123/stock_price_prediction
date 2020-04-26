import torch.nn as nn
from torch.nn import Transformer
import torch
import math


# Transformer Encoder
# ref: https://zhuanlan.zhihu.com/p/48508221

# TimeSeries Transformer Architecture reference from this paper
# https://arxiv.org/pdf/2001.08317.pdf

# TimeSeries Transformer code ref from
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

# Time Series Transformer
class TimeSeriesTransformer(nn.Module):

    def __init__(self, out_dim=1, d_input=34, dropout=0.2, num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=64, activation="gelu",
                 seq_length=7):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        print(seq_length)
        print("sefsefsef")
        # Better to be divisible by 2
        self.d_model = 16
        # Must be divisible by embedding dim
        self.nhead = 4

        # Embedding Module
        self.embedding = nn.Linear(d_input, self.d_model)

        # Add Positional Encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, seq_length)

        # Encoding and Decoding Stack
        self.transformer_model = nn.Transformer(d_model=self.d_model, nhead=self.nhead,
                                                num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation)

        # Output Module
        self.last_fn = nn.Linear(self.d_model, out_dim)

    def forward(self, src, tgt):
        src, tgt = self.embedding(src) * math.sqrt(self.d_model), self.embedding(tgt) * math.sqrt(
            self.d_model)
        src, tgt = self.pos_encoder(src), self.pos_encoder(tgt)
        out = self.transformer_model(src, tgt)
        out = self.last_fn(out)
        return out.view(-1, 1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=7):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
