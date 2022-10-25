"""
Train a super-resolution model.
"""

import argparse

from mpi4py import MPI
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter

import os
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import (
    load_superres_data,
    image_files_train_valid_test_split,
    load_data,
)
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import datetime


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.log(f"Worker {dist.get_rank()}: started")

    full_log_dir = logger.configure(dir=args.log_dir)
    log_sub_dir = os.path.join(
        full_log_dir, "runs", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )
    if dist.get_rank() == 0:
        # Set summarywriter only for the thread 0
        tb = SummaryWriter(log_dir=log_sub_dir)
        logger.log(f"Worker {dist.get_rank()}: Creating logger")
    else:
        tb = None

    logger.log(f"Worker {dist.get_rank()}: Creating model and diffusion...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    if args.model_path:
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        logger.log(f"Worker {dist.get_rank()}: Model loaded")

    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f"Worker {dist.get_rank()}: Creating data loader...")
    if not args.pre_train:
        """
        val_files = [
            "Slide002-2.tif",
            "Slide003-2.tif",
            "Slide005-1.tif",
            "Slide008-1.tif",
            "Slide008-2.tif",
            "Slide010-1.tif",
            "Slide011-1.tif",
            "Slide011-5.tif",
        ]
        """
        val_files = [
            "Slide011-6.tif",
            "Slide019-3.tif",
            "Slide022-1.tif",
            "Slide022-3.tif",
            "Slide023-3.tif",
            "Slide025-1.tif",
            "Slide028-1.tif",
            "Slide029-3.tif",
        ]
        train_paths, valid_paths = image_files_train_valid_test_split(
            args.data_dir, val_files, 0.3
        )

        logger.log(
            f"Worker {dist.get_rank()}: The size of training set {len(train_paths)}"
        )
        logger.log(
            f"Worker {dist.get_rank()}: The size of validation set {len(valid_paths)}"
        )
        train_data = load_superres_data(
            paths=train_paths,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            deterministic=False,
        )

        logger.log(f"Worker {dist.get_rank()}: Training data loaded")
        # This prevent dataloading stuck
        new_valid_paths = []
        for path in valid_paths:
            for _ in range(MPI.COMM_WORLD.Get_size()):
                new_valid_paths.append(path)

        if dist.get_rank() == 0:
            valid_kwargs = load_data_for_validation(
                new_valid_paths, args.tb_valid_im_num, args.patch_size
            )
        else:
            valid_kwargs = None
    else:
        logger.log(f"Worker {dist.get_rank()}: Pre-training mode")
        train_data = load_superres_data_old(
            args.data_dir,
            args.batch_size,
            large_size=args.large_size,
            small_size=args.small_size,
        )
        logger.log(f"Worker {dist.get_rank()}: Training data loaded")
        if dist.get_rank() == 0:
            valid_kwargs = load_data_for_validation_old(
                args.val_data_dir,
                args.tb_valid_im_num,
                args.large_size,
                args.small_size,
            )
        else:
            valid_kwargs = None

    if dist.get_rank() == 0:
        # Store the hyperparameters setting
        logger.save_parameters(args=args, dir=log_sub_dir)

    logger.log(f"Worker {dist.get_rank()}: Training started")

    training_loop = TrainLoop(
        tb=tb,
        model=model,
        diffusion=diffusion,
        data=train_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        total_steps=args.total_steps,
        valid_kwargs=valid_kwargs,
        tb_valid_im_num=args.tb_valid_im_num,
    )
    logger.log(f"Worker {dist.get_rank()}: Training loop constructed")
    training_loop.run_loop()

    if dist.get_rank() == 0 and tb:
        tb.close()


def load_superres_data_old(
    data_dir, batch_size, large_size, small_size, deterministic=False
):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=False,
        deterministic=deterministic,
        random_crop=True,
    )
    for large_batch, model_kwargs in data:
        low_res = F.interpolate(large_batch, small_size, mode="area")

        # Upsampling
        model_kwargs["low_res"] = F.interpolate(
            low_res, (large_size, large_size), mode="bilinear"
        )

        yield large_batch, model_kwargs


def load_data_for_validation_old(data_dir, tb_valid_im_num, large_size, small_size):
    # Utilize load_superres_data function to preprocess the image data
    dataloader = load_superres_data_old(
        data_dir,
        batch_size=tb_valid_im_num,
        large_size=large_size,
        small_size=small_size,
        deterministic=True,
    )

    data = next(dataloader)

    batch, model_kwargs = data
    model_kwargs["high_res"] = batch

    return model_kwargs


def load_data_for_validation(
    valid_paths, tb_valid_im_num, patch_size,
):
    # Utilize load_superres_data function to preprocess the image data
    dataloader = load_superres_data(
        paths=valid_paths,
        batch_size=tb_valid_im_num,
        patch_size=patch_size,
        deterministic=True,
        random_crop=False,
    )

    data = next(dataloader)

    batch, model_kwargs = data
    model_kwargs["high_res"] = batch

    return model_kwargs


def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="",
        val_data_dir="",
        schedule_sampler="uniform",
        patch_size=128,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        tb_valid_im_num=10,
        total_steps=1050e3,
        model_path="",
        pre_train=False,
        large_size=256,
        small_size=64,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
