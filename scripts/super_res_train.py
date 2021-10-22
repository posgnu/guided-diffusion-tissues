"""
Train a super-resolution model.
"""

import argparse

import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import os
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_superres_data, _list_image_files_train_valid_test
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
    print(dist.get_rank()) 
    if dist.get_rank() == 0:
        full_log_dir = logger.configure(dir=args.log_dir)
        tb = SummaryWriter(log_dir=os.path.join(full_log_dir, 'runs', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

        logger.log("creating model and diffusion...")
    else:
        tb=None
    print(args)
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    if args.resume_checkpoint is not None:
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    
    train_paths, valid_paths, test_paths = _list_image_files_train_valid_test(
        args.data_dir,
        args.valid_samples,
        args.test_samples)
    print(len(train_paths))
    print(len(valid_paths))

    train_data = load_superres_data(
        paths = train_paths,
        batch_size = args.batch_size,
        patch_size = args.patch_size,
    )
    valid_kwargs = load_data_for_validation(
            valid_paths,
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
    ).run_loop()

    if dist.get_rank() == 0:
        tb.close()

def load_data_for_validation(
    valid_paths,
    tb_valid_im_num,
    patch_size,
        ):
    data = next(load_superres_data(
        paths = valid_paths,
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
        valid_samples=['Slide002-2.tif', 'Slide003-2.tif', 'Slide005-1.tif', 'Slide008-1.tif', 'Slide008-2.tif', 'Slide010-1.tif', 'Slide011-1.tif', 'Slide011-5.tif', 'Slide011-6.tif', 'Slide019-3.tif', 'Slide022-1.tif', 'Slide022-3.tif', 'Slide023-3.tif', 'Slide025-1.tif', 'Slide028-1.tif', 'Slide029-3.tif', 'Slide030-1.tif', 'Slide032-3.tif', 'Slide036-1.tif', 'Slide036-2.tif', 'Slide037-2.tif', 'Slide039-1.tif', 'Slide042-1.tif', 'Slide044-3.tif', 'Slide046-3.tif', 'Slide047-2.tif', 'Slide053-1.tif'],
        test_samples=['Slide008-3.tif', 'Slide011-4.tif', 'Slide013-2.tif', 'Slide014-2.tif', 'Slide019-1.tif', 'Slide019-2.tif', 'Slide022-4.tif', 'Slide031-1.tif', 'Slide034-3.tif', 'Slide035-1.tif', 'Slide044-2.tif', 'Slide045-1.tif', 'Slide045-2.tif', 'Slide045-3.tif', 'Slide052-2.tif'], 
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
