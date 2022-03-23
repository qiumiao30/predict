# -*- coding: utf-8 -*-
# @Time    : 2021/9/1 15:01
# @File    : transformer.py
import torch
import torch.nn as nn
import numpy as np
import time
import math
from .attention import MultiHeadAttention
# from .modules import FeatureAttentionLayer, GRULayer, Forecasting_Model
from args import get_parser
parser = get_parser()
args = parser.parse_args()

#@save
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

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x
class ValueEmbedding(nn.Module):
    def __init__(self, input, hid_dim):
        super().__init__()
        self.valueGRU = nn.GRU(input_size=input, hidden_size=hid_dim, batch_first=True)

    def forward(self, x):
        x = self.valueGRU(x)
        return x

#@save
class PositionWiseFFN(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(num_hiddens, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

#@save
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

#@save
class EncoderBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, num_heads, ffn_num_hiddens, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(num_hiddens, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X, X, X))
        return self.addnorm2(Y, self.ffn(Y))

#@save
class TransformerEncoder(nn.Module):
    def __init__(self, n_features, num_hiddens, norm_shape, num_heads, num_layers, ffn_num_hiddens, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.value_embedding = TokenEmbedding(c_in=n_features, d_model=num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i), EncoderBlock(num_hiddens, norm_shape, num_heads, ffn_num_hiddens,
                                                              dropout, use_bias))

        self.gru = nn.GRU(num_hiddens, num_hiddens//2, batch_first=True)
        self.linear = nn.Linear(num_hiddens//2, n_features)

    def forward(self, X, *args):
        # 因为位置编码值在 -1 和 1 之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(X * math.sqrt(self.num_hiddens)) + self.value_embedding(X)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        _, h = self.gru(X)
        h = h[-1, :, :].view(X.shape[0], -1)
        X = self.linear(h)

        return X


#
# encoder = TransformerEncoder(n_features=38, num_hiddens=512, norm_shape=[100, 512],
#                  num_heads=8, num_layers=2, ffn_num_hiddens=1024, dropout=0.5)
# encoder.eval()
# print(encoder)
# print(encoder(torch.randn((256, 100, 38))).shape)
#
# encoder = GATEncoder(num_hiddens=38, norm_shape=[150, 38],
#     ffn_num_input=38, ffn_num_hiddens=1024, num_layers=2, dropout=0.5)
# encoder.eval()
# print(encoder)
# print(encoder(torch.randn((256, 150, 38))).shape)