from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from config.config import BaseConfig as config
from util.train import get_total_train_steps
from util.data import get_dataset, Loader
from config.private import PrivateConfig
from model.rgbd_fr import PtlRgbdFr
from util.env import seed_all

import pytorch_lightning as pl
import wandb
import os


if __name__ == '__main__':
    print(f'pid={os.getpid()}')
    seed_all(config.random_seed)
    conf_dict = {}
    for k, v in config.__dict__.items():
        if "__" not in k:
            conf_dict[k] = v

    train_dataset, \
    valid_gallery_dataset, \
    valid_probe_dataset = get_dataset()
    dataloader = Loader({
            "train": train_dataset,
            "valid_probe": valid_probe_dataset,
        }
    )

    total_steps = get_total_train_steps(
        train_dataset,
        len(config.gpu),
        config.batch_size,
        config.epoch
    )

    model = PtlRgbdFr(total_steps=total_steps, gallery=valid_gallery_dataset)

    wandb.login(key=PrivateConfig.wandb_key)
    logger = WandbLogger(
        project=PrivateConfig.wandb_project_name,
        name=PrivateConfig.wandb_run_name,
        entity=PrivateConfig.entity,
        config=conf_dict
    )
    logger.watch(model.rgbd_fr, log="all", log_freq=1)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=config.epoch,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=config.gpu,
        logger=logger,
        precision=config.precision,
        strategy="ddp_find_unused_parameters_false",
        callbacks=[lr_monitor],
        val_check_interval=config.valid_check_interval
    )
    trainer.fit(model=model, datamodule=dataloader)