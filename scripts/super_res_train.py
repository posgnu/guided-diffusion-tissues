"""
Train a super-resolution model.
"""

import argparse

import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
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
    if dist.get_rank() == 0:
        full_log_dir = logger.configure(dir=args.log_dir)
        tb = SummaryWriter(log_dir=os.path.join(full_log_dir, 'runs', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

        logger.log("creating model and diffusion...")
    else:
        tb=None
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    if args.resume_checkpoint is not None:
        dist_util.load_state_dict(args.model_path, map_location="cpu
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_superres_data(
        args.data_dir,
        args.batch_size,
        test_samples=args.test_samples,
        patch_size=args.patch_size,
        class_cond=args.class_cond,
    )
    test_kwargs = load_data_for_testing(
            args.data_dir,
            args.tb_test_im_num,
            test_samples=args.test_samples,
            patch_size=args.patch_size,
            class_cond=args.class_cond)
    
    if dist.get_rank() == 0:
        logger.save_parameters(args=args, dir=full_log_dir)
        logger.log("training...")
    
    TrainLoop(
        tb=tb,
        model=model,
        diffusion=diffusion,
        data=data,
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
        test_kwargs=test_kwargs,
    ).run_loop()

    if dist.get_rank() == 0:
        tb.close()

def load_data_for_testing(
        ):
    data = next(load_superres_data(
            ))
    batch, model_kwargs = data
    model_kwargs["high_res"] = batch
    return model_kwargs

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
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
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
