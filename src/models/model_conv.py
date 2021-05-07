#!/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: romanshen 
@file: model_conv.py
@time: 2021/05/07
@contact: xiangqing.shen@njust.edu.cn
"""


from torch import nn
import torch
import torch.nn.functional as F


class ConvClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        out_features,
    ):
        super().__init__()

        self.first_conv = nn.Conv2d(in_channels, 32, 3, 1, 2)
        self.max_pool = nn.MaxPool2d(2)
        self.second_conv = nn.Conv2d(32, 64, 3, 1, 1)
        self.third_conv = nn.Conv2d(64, 128, 3, 1, 1)
        self.fc1 = nn.Linear(128 * 3 * 3, 64)
        self.last_layer = nn.Linear(64, out_features)

    def forward(self, x):
        x = self.first_conv(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = F.relu(self.second_conv(x))
        x = self.max_pool(x)
        x = F.relu(self.third_conv(x))
        x = self.max_pool(x)
        s = x.shape
        x = torch.reshape(x, (s[0], s[1], -1))
        x = F.relu(self.fc1(x))
        x = self.last_layer(x)
        logx = F.log_softmax(x, dim=-1)
        return x, logx
