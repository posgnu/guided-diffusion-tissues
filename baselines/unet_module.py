from typing import Tuple, Dict, Any

import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader
from guided_diffusion.image_datasets import image_files_train_valid_test_split
from glob import glob
import numpy as np

from os import path
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import pytorch_lightning as pl

from baselines.unet import UNet
from guided_diffusion.image_datasets import PreloadedImageDataset
from torchmetrics.functional import structural_similarity_index_measure as ssim


class UNetTrainingModule(pl.LightningModule):
    def __init__(self, options: Dict[str, Any]):
        super(UNetTrainingModule, self).__init__()

        self.save_hyperparameters(options)
        self.options = options
        self.unet = UNet()

    def forward(self, low_res_image):
        return self.unet(low_res_image)

    def training_step(self, batch, batch_idx):
        high_res_image, low_res_image, upscaled_image = batch
        if self.options["upsampled_source"]:
            low_res_image = upscaled_image

        high_res_image = (high_res_image - 0.5) * 2
        low_res_image = (low_res_image- 0.5) * 2

        batch_size = high_res_image.shape[0]
        combined_images = torch.cat((low_res_image, high_res_image), dim=0)

        reconstructed_image, reconstructed_features = self.unet(combined_images)

        high_res_features = [feature[batch_size:].detach() for feature in reconstructed_features]
        high_res_reconstruction = reconstructed_image[batch_size:].detach()

        reconstructed_features = [feature[:batch_size] for feature in reconstructed_features]
        reconstructed_image = reconstructed_image[:batch_size]

        # self.eval()
        # with torch.no_grad():
        #     _, high_res_features = self.unet(high_res_image)
        #
        # self.train()

        mse = ((reconstructed_image - high_res_image) ** 2).mean()
        for reconstructed_feature, high_res_feature in zip(reconstructed_features, high_res_features):
            mse = mse + ((reconstructed_feature - high_res_feature) ** 2).mean()

        ssim_score = ssim(reconstructed_image.to(torch.float), high_res_image)

        self.log("train_loss", mse)
        self.logger.experiment.add_scalar("MSE", mse, self.global_step)
        self.logger.experiment.add_scalar("SSIM", ssim_score, self.global_step)

        return mse

    def validation_step(self, batch, batch_idx):
        high_res_image, low_res_image, upscaled_image = batch
        if self.options["upsampled_source"]:
            low_res_image = upscaled_image

        reconstructed_image, _ = self.unet((low_res_image - 0.5) * 2)
        reconstructed_image = reconstructed_image / 2 + 0.5

        mse = ((reconstructed_image - high_res_image) ** 2).mean()

        self.log("val_loss", mse)

        if batch_idx == 0:
            images = torch.cat((low_res_image, high_res_image, reconstructed_image), dim=3)
            images = images[:16]
            images = images.permute(1, 0, 2, 3)
            images = images.reshape(3, -1, images.shape[-1])
            images = images.permute(0, 2, 1)
            self.logger.experiment.add_image("generated_images", images, self.global_step)

        return mse

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.options['learning_rate'])
        return optimizer


class UNetDataModule(pl.LightningDataModule):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.batch_size = options["batch_size"]

    def setup(self, stage=None):
        train_paths, valid_paths = image_files_train_valid_test_split(
            self.options["data_dir"],
            self.options["valid_samples"],
        )
        test_paths = valid_paths

        # noinspection PyAttributeOutsideInit
        self.training_dataset = PreloadedImageDataset(
            image_paths=train_paths,
            patch_size=self.options["patch_size"],
            epoch_length=self.options["epoch_length"] * 16,
        )

        # noinspection PyAttributeOutsideInit
        self.validation_dataset = PreloadedImageDataset(
            image_paths=valid_paths,
            patch_size=self.options["patch_size"],
            epoch_length=self.options["epoch_length"],
            deterministic=True
        )

        # noinspection PyAttributeOutsideInit
        self.testing_dataset = PreloadedImageDataset(
            image_paths=test_paths,
            patch_size=self.options["patch_size"],
            epoch_length=self.options["epoch_length"],
            deterministic=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.options["gpus"] > 0,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            pin_memory=self.options["gpus"] > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testing_dataset,
            batch_size=self.batch_size,
            pin_memory=self.options["gpus"] > 0,
        )