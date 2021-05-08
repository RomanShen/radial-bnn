#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: romanshen 
@file: module_radial.py 
@time: 2021/05/08
@contact: xiangqing.shen@njust.edu.cn
"""


import logging

import pytorch_lightning as pl
import torch

from src.models.model_radial import SVIConvClassifier
from src.loss import Elbo
from src.lightning_classes.variational_accuracy import VariationalAccuracy


logger = logging.getLogger(__name__)


class SVIConvModule(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.model = SVIConvClassifier(
            in_channels=self.hparams.in_channels,
            out_features=self.hparams.out_features,
            initial_rho=self.hparams.initial_rho,
            initial_mu_std=self.hparams.initial_mu_std,
            variational_distribution=self.hparams.variational_distribution,
            prior=self.hparams.prior
        )

        self.criterion = Elbo(self.hparams.loss.binary, self.hparams.loss.regression)
        self.criterion.set_model(self.model, self.hparams.batch_size)

        self.epoch_variational_samples = None
        self.in_pretraining = False

        self.train_accuracy = VariationalAccuracy()
        self.val_accuracy = VariationalAccuracy()
        self.test_accuracy = VariationalAccuracy()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.expand((-1, self.epoch_variational_samples, -1, -1, -1))
        return self.model(x)

    def step(self, batch):
        x, y = batch
        logits = self.forward(x)
        return logits, y

    def training_step(self, batch, batch_idx):
        logits, y = self.step(batch)
        nll_loss, kl_term = self.criterion.compute_loss(logits, y)
        loss = nll_loss + kl_term  # already take mean
        acc = self.train_accuracy(logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        logits, y = self.step(batch)
        nll_loss, kl_term = self.criterion.compute_loss(logits, y)
        loss = nll_loss + kl_term
        acc = self.val_accuracy(logits, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        logits, y = self.step(batch)
        nll_loss, kl_term = self.criterion.compute_loss(logits, y)
        loss = nll_loss + kl_term
        acc = self.test_accuracy(logits, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_train_epoch_start(self):
        self.criterion.set_num_batches(self.hparams.num_train_batches)
        self.epoch_variational_samples = self.hparams.variational_train_samples
        if self.hparams.pretrain_epochs is not None:
            if self.current_epoch < self.hparams.pretrain_epochs:
                self.model.pretraining(True)
                logger.info("This epoch is with pretraining")
                self.in_pretraining = True
                self.epoch_variational_samples = 1
            elif self.current_epoch == self.hparams.pretrain_epochs:
                logger.info("Turning off pretraining now")
                self.model.pretraining(False)
                self.in_pretraining = False

    def on_validation_epoch_start(self):
        self.criterion.set_num_batches(self.hparams.num_val_batches)
        self.epoch_variational_samples = self.hparams.variational_val_samples
        if self.in_pretraining:
            self.epoch_variational_samples = 1

    def on_test_epoch_start(self):
        self.criterion.set_num_batches(self.hparams.num_test_batches)
        self.epoch_variational_samples = self.hparams.variational_val_samples
        if self.in_pretraining:
            self.epoch_variational_samples = 1

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            amsgrad=self.hparams.amsgrad,
        )
