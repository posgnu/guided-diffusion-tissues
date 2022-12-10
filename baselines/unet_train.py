from argparse import ArgumentParser
from typing import List, Optional

import h5py

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from baselines.unet_module import UNetDataModule, UNetTrainingModule

DEFAULT_OPTIONS = {
    "epoch_length": 1024,

    "valid_samples": ["Slide011-6.tif","Slide019-3.tif","Slide022-1.tif","Slide022-3.tif","Slide023-3.tif","Slide025-1.tif","Slide028-1.tif","Slide029-3.tif"],
}


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("data_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--upsampled_source", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1_000)

    return parser.parse_args()


def main(
        data_dir: str,
        batch_size,
        patch_size,
        upsampled_source,
        learning_rate,
        logdir: str,
        name: Optional[str],
        gpus: int,
        epochs: int
):
    options = DEFAULT_OPTIONS.copy()

    options["gpus"] = gpus
    options["data_dir"] = data_dir
    options["patch_size"] = patch_size
    options["batch_size"] = batch_size
    options["learning_rate"] = learning_rate
    options["upsampled_source"] = upsampled_source

    data_module = UNetDataModule(options)
    network_module = UNetTrainingModule(options)

    logger = TensorBoardLogger(save_dir=logdir, name=name)

    trainer = pl.Trainer(
        logger=logger,
        gpus=gpus,
        # weights_summary='full',
        max_epochs=epochs,
        precision=16,
        val_check_interval=0.25
    )

    trainer.fit(network_module, data_module)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(**arguments.__dict__)
