#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: romanshen 
@file: run_radial.py 
@time: 2021/05/08
@contact: xiangqing.shen@njust.edu.cn
"""


import logging
import os
import datetime

from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Timer

from src.lightning_classes.module_radial import SVIConvModule
from src.lightning_classes.datamodule_mnist import MNISTDataModule


logger = logging.getLogger(__name__)


def run(cfg):
    pl.seed_everything(cfg.seed)

    output_dir = os.getcwd()
    output_dir = os.path.join(output_dir, "lightning_logs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(hydra.utils.get_original_cwd())

    # lightningmodule checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",
        dirpath=output_dir,
        filename="{epoch:02d}-{val/acc:.2f}",
        save_top_k=3,
        mode="max",
    )

    # early stopping
    early_stopping_callback = EarlyStopping(
        monitor="val/acc", min_delta=0.0, patience=5, verbose=False, mode="max"
    )

    # timer
    timer = Timer()

    callbacks = [
        early_stopping_callback,
        checkpoint_callback,
        timer
    ]

    # logger
    wandb_logger = WandbLogger(
        name="radial_cnn_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        save_dir=output_dir,
        project="radial-bnn",
    )

    # initialize trainer
    trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, **cfg.trainer)

    # datamodule
    dm = MNISTDataModule(cfg=cfg.datamodule)
    dm.prepare_data()
    dm.setup()

    # model
    lightningmodule_cfg = cfg.lightningmodule
    num_batch = OmegaConf.create({
        "num_train_batches": len(dm.train_dataloader()),
        "num_val_batches": len(dm.val_dataloader()),
        "num_test_batches": len(dm.test_dataloader())
    })
    new_cfg = OmegaConf.merge(num_batch, lightningmodule_cfg)
    model = SVIConvModule(cfg=new_cfg)

    # train model
    trainer.fit(model=model, datamodule=dm)

    # test model
    trainer.test(datamodule=dm, ckpt_path="best")

    # training time
    logger.info("{} elapsed in training".format(timer.time_elapsed("train")))


@hydra.main(config_path="conf", config_name="config_radial")
def run_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    run_model()