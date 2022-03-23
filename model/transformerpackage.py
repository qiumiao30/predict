# -*- coding: utf-8 -*-
# @Time    : 2021/11/25 15:43
# @File    : transformerpackage.py
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, num_hiddens).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, num_hiddens, 2).float() * -(math.log(10000.0) / num_hiddens)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(self.pe[:, :x.size(1)])

class TransformerEncoderPackage(nn.Module):
    def __init__(self, n_features, d_model, num_layers, nhead):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=d_model, kernel_size=3, padding=1)
        self.position = PositionalEncoding(num_hiddens=d_model, dropout=0.3)
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers)
        self.gru = nn.GRU(d_model, d_model // 2, batch_first=True)
        self.linear = nn.Linear(d_model // 2, n_features)

    def forward(self, x):
        x = self.position(x) + self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.transformer_encoder(x)
        _, h = self.gru(out)
        h = h[-1, :, :]
        out = self.linear(h)

        return out
