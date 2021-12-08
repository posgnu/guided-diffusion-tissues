"""
Train a super-resolution model.
"""

import argparse

import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import os
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets_multiple import load_superres_data, _list_image_files
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion_multiple,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import datetime

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    print(dist.get_rank()) 
    if dist.get_rank() == 0:
        full_log_dir = logger.configure(dir=args.log_dir)
        tb = SummaryWriter(log_dir=os.path.join(full_log_dir, 'runs', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

        logger.log("creating model and diffusion...")
    else:
        tb=None
    print(args)
    model, diffusion = sr_create_model_and_diffusion_multiple(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys()), n_illum=args.n_illum
    )
    if args.resume_checkpoint is not None:
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    train_paths = _list_image_files(
        args.data_dir,
        args.train_samples,
        args.samples_to_exclude)

    valid_paths = _list_image_files(
        args.data_dir,
        args.valid_samples,
        args.samples_to_exclude)

    test_paths = _list_image_files(
        args.data_dir,
        args.test_samples,
        args.samples_to_exclude)

    print(len(train_paths))
    print(len(valid_paths))

    train_data = load_superres_data(
        paths = train_paths,
        n_illum = args.n_illum,
        batch_size = args.batch_size,
        patch_size = args.patch_size,
    )
    valid_kwargs = load_data_for_validation(
            valid_paths,
            args.n_illum,
            len(valid_paths),
            args.patch_size)
    
    if dist.get_rank() == 0:
        logger.save_parameters(args=args, dir=full_log_dir)
        logger.log("training...")
    
    TrainLoop(
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
        n_illum=args.n_illum,
    ).run_loop()

    if dist.get_rank() == 0:
        tb.close()

def load_data_for_validation(
    valid_paths,
    n_illum,
    tb_valid_im_num,
    patch_size,
        ):
    data = next(load_superres_data(
        paths = valid_paths,
        n_illum = n_illum,
        batch_size = tb_valid_im_num,
        patch_size = patch_size
            ))
    batch, model_kwargs = data
    model_kwargs["high_res"] = batch

    return model_kwargs

def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="",
        schedule_sampler="uniform",
        patch_size=128,
        n_illum=15,
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
        train_samples = ['Slide001-1_Trim', 'Slide001-2_Trim', 'Slide001-3_Trim', 'Slide001-4_Trim', 'Slide002-1_Trim', 'Slide002-3_Trim', 'Slide003-1_Trim', 'Slide004-1_Trim', 'Slide004-2_Trim', 'Slide004-3_Trim', 'Slide004-4_Trim', 'Slide005-2_Trim', 'Slide007-1_Trim', 'Slide007-2_Trim', 'Slide009-1_Trim', 'Slide009-2_Trim', 'Slide009-3_Trim', 'Slide009-4_Trim', 'Slide010-2_Trim', 'Slide011-2_Trim', 'Slide011-3_Trim', 'Slide012-1_Trim', 'Slide013-1_Trim', 'Slide014-1_Trim', 'Slide015-1_Trim', 'Slide015-2_Trim', 'Slide015-3_Trim', 'Slide015-4_Trim', 'Slide016-1_Trim', 'Slide016-2_Trim', 'Slide016-3_Trim', 'Slide017-1_Trim', 'Slide017-2_Trim', 'Slide018-1_Trim', 'Slide018-2_Trim', 'Slide020-1_Trim', 'Slide020-2_Trim', 'Slide020-3_Trim', 'Slide022-2_Trim', 'Slide023-1_Trim', 'Slide023-2_Trim', 'Slide024-1_Trim', 'Slide024-2_Trim', 'Slide025-2_Trim', 'Slide026-1_Trim', 'Slide026-2_Trim', 'Slide027-1_Trim', 'Slide027-2_Trim', 'Slide028-2_Trim', 'Slide028-3_Trim', 'Slide029-1_Trim', 'Slide029-2_Trim', 'Slide030-2_Trim', 'Slide031-2_Trim', 'Slide031-3_Trim', 'Slide031-4_Trim', 'Slide032-1_Trim', 'Slide032-2_Trim', 'Slide033-1_Trim', 'Slide033-2_Trim', 'Slide033-3_Trim', 'Slide034-1_Trim', 'Slide034-2_Trim', 'Slide036-3_Trim', 'Slide037-1_Trim', 'Slide037-3_Trim', 'Slide037-4_Trim', 'Slide038-1_Trim', 'Slide038-2_Trim', 'Slide038-3_Trim', 'Slide039-2_Trim', 'Slide039-3_Trim', 'Slide041-1_Trim', 'Slide041-2_Trim', 'Slide042-2_Trim', 'Slide042-3_Trim', 'Slide043-1_Trim', 'Slide043-2_Trim', 'Slide043-3_Trim', 'Slide044-1_Trim', 'Slide046-1_Trim', 'Slide046-2_Trim', 'Slide047-1_Trim', 'Slide048-1_Trim', 'Slide048-2_Trim', 'Slide049-1_Trim', 'Slide049-2_Trim', 'Slide050-1_Trim', 'Slide050-2_Trim', 'Slide050-3_Trim', 'Slide051-1_Trim', 'Slide051-2_Trim', 'Slide052-1_Trim', 'Slide053-2_Trim', 'Slide054-1_Trim', 'Slide054-2_Trim', 'Slide054-3_Trim'],
        valid_samples = ['Slide002-2_Trim', 'Slide003-2_Trim', 'Slide005-1_Trim', 'Slide008-1_Trim', 'Slide008-2_Trim', 'Slide010-1_Trim', 'Slide011-1_Trim', 'Slide011-5_Trim', 'Slide011-6_Trim', 'Slide019-3_Trim', 'Slide022-1_Trim', 'Slide022-3_Trim', 'Slide023-3_Trim', 'Slide025-1_Trim', 'Slide028-1_Trim', 'Slide029-3_Trim', 'Slide030-1_Trim', 'Slide032-3_Trim', 'Slide036-1_Trim', 'Slide036-2_Trim', 'Slide037-2_Trim', 'Slide039-1_Trim', 'Slide042-1_Trim', 'Slide044-3_Trim', 'Slide046-3_Trim', 'Slide047-2_Trim', 'Slide053-1_Trim'],
        test_samples = ['Slide008-3_Trim', 'Slide011-4_Trim', 'Slide013-2_Trim', 'Slide014-2_Trim', 'Slide019-1_Trim', 'Slide019-2_Trim', 'Slide022-4_Trim', 'Slide031-1_Trim', 'Slide034-3_Trim', 'Slide035-1_Trim', 'Slide044-2_Trim', 'Slide045-1_Trim', 'Slide045-2_Trim', 'Slide045-3_Trim', 'Slide052-2_Trim'],
        samples_to_exclude = ['Slide002-1_Trim', 'Slide033-1_Trim', 'Slide029-1_Trim', 'Slide018-2_Trim', 'Slide033-1_Trim', 'Slide033-3_Trim', 'Slide041-2_Trim', 'Slide043-3_Trim', 'Slide044-1_Trim', 'Slide046-2_Trim', 'Slide022-1_Trim', 'Slide022-3_Trim', 'Slide042-1_Trim', 'Slide046-3_Trim', 'Slide019-2_Trim', 'Slide035-1_Trim',  'Slide044-2_Trim'],
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
