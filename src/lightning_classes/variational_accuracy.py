#!/usr/bin/env python
# -*- coding:utf-8 -*-  
""" 
@author: romanshen 
@file: variational_accuracy.py 
@time: 2021/05/08
@contact: xiangqing.shen@njust.edu.cn
"""


from torchmetrics import Metric
import torch


class VariationalAccuracy(Metric):
    
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        preds = torch.exp(preds)
        class_prediction = torch.argmax(preds, dim=2)
        modal_prediction, _ = torch.mode(class_prediction, dim=1)
        self.correct += torch.sum(modal_prediction == targets)
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total
