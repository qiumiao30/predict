# -*- coding: utf-8 -*-
# @Time    : 2021/11/1 21:22
# @File    : LSTNet.py
__author__ = "Guan Song Wang"
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTNet(nn.Module):
    def __init__(self, args):
        super(LSTNet, self).__init__()
        self.P = args.lookback
        self.m = 38
        self.hidR = args.gru_hid_dim
        self.hidC = 100
        self.hidS = 50
        self.Ck = 3
        self.skip = 20

        self.hw = 20
        self.conv1 = nn.Conv1d(self.m, self.hidC, kernel_size=self.Ck)
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if (self.skip > 0):
            self.pt = (self.P - self.Ck) / self.skip
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, 38)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, self.m)
        self.output = None

        self.linear2 = nn.Linear(self.m, 38)

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.permute(0, 2, 1)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        # c = c.permute(0, 2, 1)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)

        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn

        if (self.skip > 0):
            self.pt = int(self.pt)
            s = c[:, :, int(-self.pt * self.skip):].contiguous()

            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            z = self.linear2(z)
            res = res+z

        return res

