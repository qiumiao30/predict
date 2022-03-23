# -*- coding: utf-8 -*-
# @Time    : 2021/7/13 15:54
# @File    : cnn_gru.py
import torch
import torch.nn as nn

from .modules import (
    ConvLayer,
    GRULayer,
    Forecasting_Model,
)

class Action(nn.Module):
    def __init__(self, n_segment=32, shift_div=8):
        super(Action, self).__init__()
        self.n_segment = n_segment
        self.in_channels = 38
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.reduced_channels = self.in_channels // 3
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fold = self.in_channels // shift_div

        # shifting
        self.action_shift = nn.Conv1d(
            self.in_channels, self.in_channels,
            kernel_size=3, padding=1, groups=self.in_channels,
            bias=False)
        self.action_shift.weight.requires_grad = True
        self.action_shift.weight.data.zero_()
        self.action_shift.weight.data[:self.fold, 0, 2] = 1  # shift left
        self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1  # shift right

        if 2 * self.fold < self.in_channels:
            self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1  # fixed

        # motion excitation
        self.pad = (0, 1)
        self.action_p3_squeeze = nn.Conv1d(self.in_channels, self.reduced_channels, kernel_size=1, stride=1,
                                           bias=False, padding=0)
        self.action_p3_bn1 = nn.BatchNorm1d(self.reduced_channels)
        self.action_p3_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3,
                                         stride=1, bias=False, padding=1, groups=self.reduced_channels)
        self.action_p3_expand = nn.Conv1d(self.reduced_channels, self.in_channels, kernel_size=1, stride=1,
                                          bias=False, padding=0)
        print('=> Using ACTION')

    def forward(self, x):
        x_shift = x.permute(0, 2, 1)

        # # 1D convolution: motion excitation
        x3 = self.action_p3_squeeze(x_shift)
        x3 = self.action_p3_bn1(x3)
        n, c, t = x3.size()
        x3_plus0, x_left = x3.split([t - 1, 1], dim=2)
        x3_plus1 = self.action_p3_conv1(x3)

        _, x3_plus1 = x3_plus1.split([1, t - 1], dim=2)
        x_p3 = x3_plus1 - x3_plus0
        # x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)
        x_p3 = torch.cat([x_p3, x_left], 2)
        # x_p3 = self.avg_pool(x_p3.view(n, c, t))
        x_p3 = self.action_p3_expand(x_p3)
        x_p3 = self.sigmoid(x_p3)
        x_p3 = x_shift * x_p3 + x_shift

        # out1 = x_p2
        out = x_p3.permute(0, 2, 1)

        return out

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x + x

class CNN_GRU(nn.Module):
    """ CNN-GRU model class.

    :param n_features: Number of input features
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param dropout: dropout rate
    """

    def __init__(
        self,
        n_features,
        out_dim,
        kernel_size=7,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        dropout=0.2,
    ):
        super(CNN_GRU, self).__init__()
        self.MotionBlocks = Action()
        self.se = SEModule(n_features)
        self.gru_hid_dim = gru_hid_dim
        self.conv = ConvLayer(n_features, kernel_size)
        self.gru1 = GRULayer(n_features, gru_hid_dim, gru_n_layers, dropout)
        self.gru2 = GRULayer(gru_hid_dim, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)

    def forward(self, x):
        # x shape (b, n, k): b---batch_size, n---window_size, k---number of features
        batch_size, seq_len, _ = x.shape

        x_e = self.se(x.permute(0,2,1)).permute(0,2,1)

        x_m = self.MotionBlocks(x)

        x_c = self.conv(x)

        output, h_end = self.gru1(x_e)
        output, h_end = self.gru2(output)
        h_end = h_end[-1, :, :]
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp

        # output = output.view(batch_size, seq_len, self.gru_hid_dim)[:, -1]

        predictions = self.forecasting_model(h_end) # [batch, features], and DATA:MSL/SMAP-->[batch, 1]

        return predictions