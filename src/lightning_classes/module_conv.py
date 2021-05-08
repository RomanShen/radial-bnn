#!/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: romanshen 
@file: module_conv.py 
@time: 2021/05/07
@contact: xiangqing.shen@njust.edu.cn
"""


import pytorch_lightning as pl
import torch
from torchmetrics.classification.accuracy import Accuracy

from src.models.model_conv import ConvClassifier


class ConvModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.model = ConvClassifier(
            in_channels=cfg.in_channels, out_features=cfg.out_features
        )

        self.criterion = torch.nn.NLLLoss()

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, y = batch
        xx, logxx = self.forward(x)
        loss = self.criterion(logxx, y)
        preds = torch.argmax(xx, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            amsgrad=self.hparams.amsgrad,
        )
