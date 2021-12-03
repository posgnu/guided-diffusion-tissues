import math
import random
import os

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import cv2
import glob
from tqdm import tqdm
def load_superres_data(
    *,
    paths,
    n_illum,
    batch_size,
    patch_size,
    deterministic=False,
    random_crop=True,
    random_flip=True,
    ):
    if not paths:
        raise ValueError("unspecified paths")
    dataset = SuperresImageDataset(
                n_illum,
                patch_size,
                paths,
                shard=MPI.COMM_WORLD.Get_rank(),
                num_shards=MPI.COMM_WORLD.Get_size(),
                random_crop=random_crop,
                random_flip=random_flip,
                )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

class SuperresImageDataset(Dataset):
    def __init__(
            self,
            n_illum, 
            patch_size,
            paths,
            shard=0,
            num_shards=1,
            random_crop=True,
            random_flip=True,
            ):
        super().__init__()
        self.n_illum = n_illum
        self.patch_size = patch_size
        self.local_images = paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        high_res_path = self.local_images[idx]
        lr_root_dir = '/'.join(high_res_path.split('/')[:-1]).replace('Registered Images', 'Upscale')
        lr_images = [os.path.join(lr_root_dir, file) for file in sorted(bf.listdir(lr_root_dir)) if not file.startswith('.')]
        range_images = np.arange(0,len(lr_images),len(lr_images)//self.n_illum+1)

        with bf.BlobFile(high_res_path, "rb") as f:
            high_res_pil_image = Image.open(f)
            high_res_pil_image.load()
            high_res_pil_image = high_res_pil_image.convert("RGB")
            high_res_arr = np.array(high_res_pil_image)

        low_res_arr = []
        for i in range_images:
            with bf.BlobFile(lr_images[i], "rb") as f:
                low_res_i_pil_image = Image.open(f)
                low_res_i_pil_image.load()
                low_res_i_pil_image = low_res_i_pil_image.convert("RGB")

                if high_res_pil_image.size != low_res_i_pil_image.size:
                    return self.__getitem__(idx+1)

                low_res_i_arr = np.array(low_res_i_pil_image)
                low_res_arr.append(low_res_i_arr)

        low_res_arr = np.concatenate(low_res_arr, axis=2)

        if self.random_crop:
            high_res_arr, low_res_arr = random_crop_arr_input_target(
                        high_res_arr,
                        low_res_arr,
                        self.patch_size)
            #if is_white(high_res_arr) or is_black(high_res_arr):
                #return self.__getitem__(idx)

        if self.random_flip and random.random() < 0.5:
            high_res_arr = high_res_arr[:, ::-1]
            low_res_arr = low_res_arr[:, ::-1]

        high_res_arr = high_res_arr.astype(np.float32) / 127.5 - 1
        low_res_arr = low_res_arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        out_dict["low_res"] = np.transpose(low_res_arr, [2, 0, 1])
        return np.transpose(high_res_arr, [2, 0, 1]), out_dict

def is_white(arr):
    white_fraction = np.median(arr) / 255
    if white_fraction >= 0.95 :
        return True
    else:
        return False

def is_black(arr):
    black_fraction = np.median(arr) / 255
    if black_fraction <= 0.05 :
        return True
    else:
        return False
 
def _file_name(file_path):
    basename = os.path.basename(file_path)
    file_name = os.path.splitext(basename)[0]
    return file_name

def _list_image_files(data_dir_hr, samples, samples_to_exclude):
    paths = []
    for entry in sorted(bf.listdir(data_dir_hr)):
         #if len(paths) < 2: #This is just for faster pipeline testing, comment during actual training
            if not entry.startswith('.'):
                if entry in samples and entry not in samples_to_exclude:
                    full_path = bf.join(data_dir_hr, entry)
                    try:
                        hr_image = glob.glob(os.path.join(full_path, '*fit_mapped*'))[0]
                    except:
                        hr_image = glob.glob(os.path.join(full_path, '*5x_mapped*'))[0]
                        # BrownLab/Ptychography/Registered Images/Slide012-1_Trim/aligned_1_slide12_5x_mapped.tif
                    paths.append(hr_image)

         #else:
             #break
    
    return paths

def random_crop_arr_input_target(pil_image_target, pil_image_input, patch_size):

    arr_target = np.array(pil_image_target)
    arr_input = np.array(pil_image_input) 
    crop_y = random.randrange(arr_target.shape[0] - patch_size + 1)
    crop_x = random.randrange(arr_target.shape[1] - patch_size + 1)
    arr_target_crop = arr_target[crop_y : crop_y + patch_size, crop_x : crop_x + patch_size]
    arr_input_crop = arr_input[crop_y : crop_y + patch_size, crop_x : crop_x + patch_size]

    if is_white(arr_target_crop) or is_black(arr_target_crop):
        return random_crop_arr_input_target(pil_image_target, pil_image_input, patch_size)
    return arr_target_crop, arr_input_crop

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
