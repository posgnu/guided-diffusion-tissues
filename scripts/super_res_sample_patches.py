"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os
import time
import torch

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
import datetime
from PIL import Image
import torchvision.transforms as T
import random


from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import is_black, is_white

PAD_VALUE = 205

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

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

    val_files = [
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide001-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide002-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide003-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide004-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide005-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide007-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide008-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide009-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide010-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide012-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide013-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide014-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide023-3.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide028-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide029-3.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide025-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide011-6.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide019-3.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide022-1.tif",
            "/baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide022-3.tif",
    ]

    # Parameters
    args.patch_size = 256
    args.batch_size = 12

    patch_arr, patch_target_arr, width, height = decompose_biomedical_image(
        val_files, args.patch_size
    )
    data = load_data_for_worker(patch_arr, args.batch_size)
    num_samples = len(patch_arr)

    all_images = sample_patches(args, model, diffusion, data, num_samples)
    # only idx % 8 == 0 are matched 
    # Averaging over images
    if dist.get_rank() == 0:
        for idx, (sr_patch, lr_patch, hr_patch) in enumerate(zip(all_images, patch_arr, patch_target_arr)):
            if idx % 8 != 0:
                continue
            tb.add_images(f"super resolution patches", sr_patch.to(torch.uint8).unsqueeze(0), idx)
            tb.add_images(f"high resolution patches", torch.from_numpy(hr_patch).permute(2, 0, 1).to(torch.uint8).unsqueeze(0), idx)
            tb.add_images(f"low resolution patches", torch.from_numpy(lr_patch).permute(2, 0, 1).to(torch.uint8).unsqueeze(0), idx)
            import time
            time.sleep(5)

    dist.barrier()    
    logger.log("sampling complete")

def sample_patches(args, model, diffusion, data, num_samples):
    logger.log("creating samples...")
    all_images = []
    # FIXME num_samples is not needed since all the samples should be used
    while len(all_images) < num_samples:
        model_kwargs, is_white_black = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        start_time = time.time()
        # FIXME: Make inference faster by excluding white and black patches
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.patch_size, args.patch_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        logger.log(f"finish! took {time.time() - start_time}")

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)

        # Replace white and black samples
        
        for i, is_replace in enumerate(is_white_black):
            if is_replace:
                sample[i] = torch.full((3, args.patch_size, args.patch_size), 255).to(
                    torch.uint8
                )
        
        sample = sample.contiguous()
        
        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample)  # gather not supported with NCCL

        # Reorder patches
        num_ranks = dist.get_world_size()
        arr = [None] * len(all_samples) * args.batch_size
        for rank in range(num_ranks):
            arr[rank::num_ranks] = all_samples[rank].cpu()

        all_images += arr

        logger.log(f"created {len(all_samples) * args.batch_size} samples")

    all_images = all_images[:num_samples]
    return all_images

def decompose_biomedical_image(image_paths, patch_size=256):
    arr_image_list, arr_target_list = pad_image_as_multiples(image_paths)
    height, width, _ = arr_image_list[0].shape

    patch_arr, patch_target_arr = decompose_into_patches(patch_size, arr_image_list, arr_target_list, width, height)
    return patch_arr, patch_target_arr, width, height

def decompose_into_patches(patch_size, arr_image_list, arr_target_list, width, height, num_patches=9):
    patch_arr = []
    target_path_arr = []
    for arr_image, target_arr_image in zip(arr_image_list, arr_target_list):
        num_patch_generated = 0
        while True:
            if num_patch_generated >= num_patches:
                break

            crop_y = random.randrange(height - patch_size + 1)
            crop_x = random.randrange(width - patch_size + 1)
            arr_crop = arr_image[crop_y : crop_y + patch_size, crop_x : crop_x + patch_size]
            target_crop = target_arr_image[crop_y : crop_y + patch_size, crop_x : crop_x + patch_size]

            if is_white(arr_crop) or is_black(arr_crop):
                continue
            
            patch_arr.append(arr_crop)
            target_path_arr.append(target_crop)

            num_patch_generated += 1

    logger.log(f"{len(patch_arr)} patches are generated")
    return patch_arr, target_path_arr

def pad_image_as_multiples(image_paths):
    arr_image_list = []
    arr_target_list = []
    for image_path in image_paths:
        target_image_path = image_path.replace("low_res", "high_res")

        with bf.BlobFile(image_path, "rb") as f:
            low_res_pil_image = Image.open(f)
            low_res_pil_image.load()
        low_res_pil_image = low_res_pil_image.convert("RGB")
        arr_image = np.array(low_res_pil_image)
        arr_image_list.append(arr_image)

        with bf.BlobFile(target_image_path, "rb") as f:
            high_res_pil_image = Image.open(f)
            high_res_pil_image.load()
        high_res_pil_image = high_res_pil_image.convert("RGB")
        target_arr_image = np.array(high_res_pil_image)
        arr_target_list.append(target_arr_image)

    return arr_image_list, arr_target_list


def compose_patches(patch_arr, tb, width, height, patch_size=256, circle=False):
    assert tb is not None

    composed_image = stitch_patches(patch_arr, width, height, patch_size, circle)
    
    # Remove margin
    result_image = composed_image[:, -2727:, -3636:]
    
    return result_image
    

def compose_patches_misaligned(patch_arr, tb, width, height, patch_size=256, circle=False):
    assert tb is not None
    assert patch_size % 2 == 0

    composed_image = stitch_patches(patch_arr, width, height, patch_size, circle)

    pad_size = patch_size // 2
    result_image = composed_image[:, pad_size:-pad_size, pad_size:-pad_size][:, -2727:, -3636:]
    
    return result_image

def stitch_patches(patch_arr, width, height, patch_size: int, circle=False):
    composed_image = torch.zeros((3, width, height)).to(torch.int)
    idx = 0
    for row in range(0, width, patch_size):
        for col in range(0, height, patch_size):
            if circle:
                template = tailor_patch(patch_arr[idx], patch_size)

                composed_image[
                    :, row : row + patch_size, col : col + patch_size
                ] = template
            else:
                composed_image[
                    :, row : row + patch_size, col : col + patch_size
                ] = patch_arr[idx]

            idx += 1
    return composed_image

# Circle
def tailor_patch(patch, patch_size):
    
    template = torch.full((3, patch_size, patch_size), fill_value=-1)
    x = torch.arange(0, patch_size)
    y = torch.arange(0, patch_size)
    cx = patch_size / 2
    cy = patch_size /2
    r = patch_size /2
    mask = (x[None,:]-cx)**2 + (y[:, None]-cy)**2 <= (r**2 + 100)
    mask = torch.stack([mask, mask, mask])

    template[mask] = patch[mask].to(int)

    return template
    

def load_data_for_worker(image_arr, batch_size):
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    is_white_black = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if is_white(image_arr[i]) or is_black(image_arr[i]):
                is_white_black.append(True)
            else:
                is_white_black.append(False)

            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)

                yield res, is_white_black
                buffer = []
                is_white_black = []


def create_argparser():
    defaults = dict(
        log_dir="",
        clip_denoised=True,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
        large_size=0,
        small_size=0,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
