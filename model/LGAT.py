# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 20:09
# @File    : LGAT.py
import torch
import torch.nn as nn
import numpy as np
import time
import math
from args import get_parser
parser = get_parser()
args = parser.parse_args()

from .modules import (
    ConvLayer,
    GRULayer,
    Forecasting_Model,
    FeatureAttentionLayer,
)

class LGAT(nn.Module):
    def __init__(self,
                 n_features,
                 window_size,
                 out_dim,
                 kernel_size=7,
                 feat_gat_embed_dim=None,
                 use_gatv2=True,
                 gru_n_layers=1,
                 gru_hid_dim=128,
                 forecast_n_layers=1,
                 forecast_hid_dim=150,
                 dropout=0.3,
                 alpha=0.2
                 ):
        super(LGAT, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.gru = GRULayer(2*n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        # self.temp_gru = nn.GRU(n_features, n_features, num_layers=2, batch_first=True)


    def forward(self, x):
        x = self.conv(x)
        h_feat = self.feature_gat(x)
        # out, _ = self.temp_gru(x)

        h_cat = torch.cat([x, h_feat], dim=2)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)

        predictions = self.forecasting_model(h_end)

        return predictions

