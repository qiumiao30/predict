# -*- coding: utf-8 -*-
# @Time    : 2022/1/2 21:59
# @File    : test_action.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import pdb
#
#
# class Action(nn.Module):
#     def __init__(self, n_segment=3, shift_div=8):
#         super(Action, self).__init__()
#         # self.net = net
#         self.n_segment = n_segment
#         self.in_channels = 32
#         self.out_channels = 16
#         self.kernel_size = 3
#         self.stride = 1
#         self.padding = 1
#         self.reduced_channels = self.in_channels // 16
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
#         self.fold = self.in_channels // shift_div
#
#         # shifting
#         self.action_shift = nn.Conv1d(
#             self.in_channels, self.in_channels,
#             kernel_size=3, padding=1, groups=self.in_channels,
#             bias=False)
#         self.action_shift.weight.requires_grad = True
#         self.action_shift.weight.data.zero_()
#         self.action_shift.weight.data[:self.fold, 0, 2] = 1  # shift left
#         self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1  # shift right
#
#         if 2 * self.fold < self.in_channels:
#             self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1  # fixed
#
#         # # # spatial temporal excitation
#         # self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3),
#         #                                  stride=(1, 1, 1), bias=False, padding=(1, 1, 1))
#
#         # # channel excitation
#         self.action_p2_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1, 1),
#                                            bias=False, padding=(0, 0))
#         self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1,
#                                          bias=False, padding=1,
#                                          groups=1)
#         self.action_p2_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1),
#                                           bias=False, padding=(0, 0))
#
#         # motion excitation
#         self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
#         self.action_p3_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1, 1),
#                                            bias=False, padding=(0, 0))
#         self.action_p3_bn1 = nn.BatchNorm2d(self.reduced_channels)
#         self.action_p3_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 3),
#                                          stride=(1, 1), bias=False, padding=(1, 1), groups=self.reduced_channels)
#         self.action_p3_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1),
#                                           bias=False, padding=(0, 0))
#         print('=> Using ACTION')
#
#     def forward(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#
#         x_shift = x.view(n_batch, self.n_segment, c, h, w)
#         x_shift = x_shift.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
#         x_shift = x_shift.contiguous().view(n_batch * h * w, c, self.n_segment)
#         x_shift = self.action_shift(x_shift)  # (n_batch*h*w, c, n_segment)
#         x_shift = x_shift.view(n_batch, h, w, c, self.n_segment)
#         x_shift = x_shift.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
#         x_shift = x_shift.contiguous().view(nt, c, h, w)
#
#         # 3D convolution: c*T*h*w, spatial temporal excitation
#         # nt, c, h, w = x_shift.size()
#         # x_p1 = x_shift.view(n_batch, self.n_segment, c, h, w).transpose(2, 1).contiguous()
#         # x_p1 = x_p1.mean(1, keepdim=True)
#         # x_p1 = self.action_p1_conv1(x_p1)
#         # x_p1 = x_p1.transpose(2, 1).contiguous().view(nt, 1, h, w)
#         # x_p1 = self.sigmoid(x_p1)
#         # x_p1 = x_shift * x_p1 + x_shift
#
#         # 2D convolution: c*T*1*1, channel excitation
#         x_p2 = self.avg_pool(x_shift)
#         x_p2 = self.action_p2_squeeze(x_p2)
#         nt, c, h, w = x_p2.size()
#         x_p2 = x_p2.view(n_batch, self.n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2, 1).contiguous()
#         x_p2 = self.action_p2_conv1(x_p2)
#         x_p2 = self.relu(x_p2)
#         x_p2 = x_p2.transpose(2, 1).contiguous().view(-1, c, 1, 1)
#         x_p2 = self.action_p2_expand(x_p2)
#         x_p2 = self.sigmoid(x_p2)
#         x_p2 = x_shift * x_p2 + x_shift
#
#         # # 2D convolution: motion excitation
#         x3 = self.action_p3_squeeze(x_shift)
#         x3 = self.action_p3_bn1(x3)
#         nt, c, h, w = x3.size()
#         x3_plus0, _ = x3.view(n_batch, self.n_segment, c, h, w).split([self.n_segment - 1, 1], dim=1)
#         x3_plus1 = self.action_p3_conv1(x3)
#
#         _, x3_plus1 = x3_plus1.view(n_batch, self.n_segment, c, h, w).split([1, self.n_segment - 1], dim=1)
#         x_p3 = x3_plus1 - x3_plus0
#         x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)
#         x_p3 = self.avg_pool(x_p3.view(nt, c, h, w))
#         x_p3 = self.action_p3_expand(x_p3)
#         x_p3 = self.sigmoid(x_p3)
#         x_p3 = x_shift * x_p3 + x_shift
#
#         out1 = x_p2
#         out2 = x_p3
#
#         # out = x_p1 + x_p2 + x_p3
#         return out1, out2
#
#
# model = Action()
# x = torch.rand(12, 32, 4, 5)
# out1, out2 = model(x)
# print(out1.shape, out2.shape)



# -*- coding: utf-8 -*-
# @Time    : 2022/1/2 21:59
# @File    : test_action.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb


class Action(nn.Module):
    def __init__(self, n_segment=32, shift_div=8):
        super(Action, self).__init__()
        self.n_segment = n_segment
        self.in_channels = 38
        self.out_channels = 16
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.reduced_channels = self.in_channels // 16
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
        self.pad = (0,1)
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
        x3_plus0, _ = x3.split([t - 1, 1], dim=2)
        x3_plus1 = self.action_p3_conv1(x3)

        _, x3_plus1 = x3_plus1.split([1, t - 1], dim=2)
        x_p3 = x3_plus1 - x3_plus0
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)
        x_p3 = self.avg_pool(x_p3.view(n, c, t))
        x_p3 = self.action_p3_expand(x_p3)
        x_p3 = self.sigmoid(x_p3)
        x_p3 = x_shift * x_p3 + x_shift

        # out1 = x_p2
        out = x_p3.permute(0, 2, 1)

        return out


model = Action()
x = torch.rand(64, 32, 38)
out = model(x)
print(out.shape)