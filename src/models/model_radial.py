#!/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: romanshen 
@file: model_radial.py
@time: 2021/05/07
@contact: xiangqing.shen@njust.edu.cn
"""


import torch
from torch import nn

from src.models.variational_bayes import SVIConv2D, SVIMaxPool2D, SVI_Linear


class SVIConvClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        out_features,
        initial_rho,
        initial_mu_std,
        variational_distribution,
        prior,
    ):
        super().__init__()

        self.first_conv = SVIConv2D(
            in_channels,
            32,
            [3, 3],
            variational_distribution,
            prior,
            initial_rho,
            initial_mu_std,
            padding=2,
        )
        self.max_pool = SVIMaxPool2D((2, 2))

        self.second_conv = SVIConv2D(
            32,
            64,
            [3, 3],
            variational_distribution,
            prior,
            initial_rho,
            initial_mu_std,
            padding=1,
        )

        self.third_conv = SVIConv2D(
            64,
            128,
            [3, 3],
            variational_distribution,
            prior,
            initial_rho,
            initial_mu_std,
            padding=1,
        )

        self.fc1 = SVI_Linear(
            128 * 3 * 3,
            64,
            initial_rho,
            initial_mu_std,
            variational_distribution,
            prior,
        )

        self.last_layer = SVI_Linear(
            64,
            out_features,
            initial_rho,
            initial_mu_std,
            variational_distribution,
            prior,
        )

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
        x = F.log_softmax(self.last_layer(x), dim=-1)
        return x
