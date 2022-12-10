from typing import Tuple, Union, List

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

from os import path
from glob import glob
from tqdm import tqdm
import hashlib



def load_preloaded_superres_downsampled_data(
    *,
    paths,
    batch_size,
    patch_size,
    deterministic=False,
    random_crop=True,
    random_flip=True,
):
    if not paths:
        raise ValueError("unspecified paths")

    dataset = PreloadedImageDataset(
        epoch_length=len(paths),
        image_paths=paths,
        patch_size=patch_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        compatible=True,
        swap_low_res=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )

    while True:
        yield from loader


def load_preloaded_superres_data(
    *,
    paths,
    batch_size,
    patch_size,
    deterministic=False,
    random_crop=True,
    random_flip=True,
):
    if not paths:
        raise ValueError("unspecified paths")

    dataset = PreloadedImageDataset(
        epoch_length=len(paths),
        image_paths=paths,
        patch_size=patch_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        compatible=True,
        swap_low_res=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )

    while True:
        yield from loader


class PreloadedImageDataset(Dataset):
    def __init__(
        self,
        epoch_length: int,
        image_paths: Union[str, List[str]],
        patch_size: Union[int, Tuple[int, int]],
        shard=0,
        num_shards=1,
        random_crop=True,
        random_flip=True,
        compatible: bool = False,
        swap_low_res: bool = False,
        deterministic: bool = False,
    ):
        super().__init__()

        # Format inputs to be consistent
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        patch_size = np.array(patch_size)
        self.patch_size = patch_size
        self.epoch_length = epoch_length
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.compatible = compatible
        self.swap_low_res = swap_low_res
        self.deterministic = deterministic

        if isinstance(image_paths, str):
            image_paths = sorted(glob(f"{image_paths}/high_res/*.tif"))

        image_paths = image_paths[shard:][::num_shards]

        # Load from cache if we already did this.
        hashing_function = lambda x: int(
            hashlib.sha256(x.encode()).hexdigest(), base=16
        )
        path_hash = sum(map(hashing_function, sorted(image_paths))) % (2 ** 48)
        base_dir = "/".join(image_paths[0].split("/")[:-2])
        cache_file = f"{base_dir}/{path_hash}.cache"
        if path.isfile(cache_file):
            print("Loading images from cache file.")
            (
                high_res_images,
                low_res_images,
                upscaled_images,
                index_sizes,
                images_sizes,
            ) = th.load(cache_file)

        else:
            high_res_images = []
            low_res_images = []
            upscaled_images = []
            index_sizes = []
            images_sizes = []

            print("Loading images from disk.")
            for image_path in tqdm(image_paths):
                high_res_image = self.load_image(image_path)
                low_res_image = self.load_image(
                    image_path.replace("high_res", "low_res")
                )

                # Make an alternative low-res image bby downscaling and upscaling the original.
                upscaled_image = high_res_image.float().unsqueeze(0)
                upscaled_image = F.interpolate(
                    upscaled_image, scale_factor=0.5, mode="bilinear"
                )
                upscaled_image = F.interpolate(
                    upscaled_image, size=high_res_image.shape[1:], mode="bilinear"
                )
                upscaled_image = upscaled_image.squeeze(0).round().byte()

                current_image_size = np.array(high_res_image.shape[1:])
                index_size = np.array(current_image_size) - patch_size + 1

                high_res_images.append(high_res_image)
                low_res_images.append(low_res_image)
                upscaled_images.append(upscaled_image)
                index_sizes.append(index_size)
                images_sizes.append(current_image_size)

            high_res_images = th.concat(
                [image.reshape(3, -1) for image in high_res_images], dim=1
            )
            low_res_images = th.concat(
                [image.reshape(3, -1) for image in low_res_images], dim=1
            )
            upscaled_images = th.concat(
                [image.reshape(3, -1) for image in upscaled_images], dim=1
            )
            index_sizes = np.array(index_sizes)
            images_sizes = np.array(images_sizes)

            th.save(
                f=cache_file,
                obj=(
                    high_res_images,
                    low_res_images,
                    upscaled_images,
                    index_sizes,
                    images_sizes,
                ),
            )

        self.high_res_images = high_res_images
        self.low_res_images = low_res_images
        self.upscaled_images = upscaled_images
        self.index_sizes = index_sizes
        self.images_sizes = images_sizes

        self.flat_image_sizes = np.cumsum([0] + list(map(np.prod, images_sizes)))
        self.num_patches_per_image = list(map(np.prod, index_sizes))
        self.patch_idx = np.cumsum([0] + self.num_patches_per_image)[:-1]

    @staticmethod
    def load_image(path: str):
        image_array = np.array(Image.open(path).convert("RGB"))
        return th.from_numpy(image_array).permute(2, 0, 1)

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        idx = idx % len(self.images_sizes)

        rng = np.random
        if self.deterministic:
            rng = np.random.RandomState(idx)

        # Convert flat image into proper image shape
        image_lower = self.flat_image_sizes[idx]
        image_upper = self.flat_image_sizes[idx + 1]
        image_size = (3, *self.images_sizes[idx])

        high_res_image = self.high_res_images[:, image_lower:image_upper].reshape(
            image_size
        )
        low_res_image = self.low_res_images[:, image_lower:image_upper].reshape(
            image_size
        )
        upscaled_image = self.upscaled_images[:, image_lower:image_upper].reshape(
            image_size
        )
        index_size = self.index_sizes[idx]

        # Find a random patch and extract that patch from all images
        if self.random_crop:
            valid_patch = False

            while not valid_patch:
                patch_lower = rng.randint(np.zeros_like(index_size), index_size)
                patch_upper = patch_lower + self.patch_size

                patch_high_res_image = high_res_image[
                    :, patch_lower[0] : patch_upper[0], patch_lower[1] : patch_upper[1]
                ]
                patch_low_res_image = low_res_image[
                    :, patch_lower[0] : patch_upper[0], patch_lower[1] : patch_upper[1]
                ]
                patch_upscaled_image = upscaled_image[
                    :, patch_lower[0] : patch_upper[0], patch_lower[1] : patch_upper[1]
                ]

                if not is_white(np.array(patch_high_res_image)) and not is_black(np.array(patch_high_res_image)):
                    valid_patch = True
                

            high_res_image = patch_high_res_image
            low_res_image = patch_low_res_image
            upscaled_image = patch_upscaled_image

        high_res_image = high_res_image.float() / 255
        low_res_image = low_res_image.float() / 255
        upscaled_image = upscaled_image.float() / 255

        # Possibly flip image
        if self.random_flip and rng.random() < 0.5:
            high_res_image = th.flip(high_res_image, dims=(1,))
            low_res_image = th.flip(low_res_image, dims=(1,))
            upscaled_image = th.flip(upscaled_image, dims=(1,))

        if self.swap_low_res:
            low_res_image, upscaled_image = upscaled_image, low_res_image

        # Old style return
        if self.compatible:
            return high_res_image, {"low_res": low_res_image}

        # Return the full resolution and two types of downscales.
        return high_res_image, low_res_image, upscaled_image


def load_superres_downsampled_data(
    *,
    paths,
    batch_size,
    patch_size,
    deterministic=False,
    random_crop=True,
    random_flip=True,
):
    if not paths:
        raise ValueError("unspecified paths")
    dataset = SuperresDownImageDataset(
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


class SuperresDownImageDataset(Dataset):
    def __init__(
        self,
        patch_size,
        paths,
        shard=0,
        num_shards=1,
        random_crop=True,
        random_flip=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.local_images = paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        high_res_path = self.local_images[idx]
        low_res_path = high_res_path.replace("high_res", "low_res")

        with bf.BlobFile(high_res_path, "rb") as f:
            high_res_pil_image = Image.open(f)
            high_res_pil_image.load()
        with bf.BlobFile(low_res_path, "rb") as f:
            low_res_pil_image = Image.open(f)
            low_res_pil_image.load()
        if high_res_pil_image.size != low_res_pil_image.size:
            return self.__getitem__(idx + 1)

        high_res_pil_image = high_res_pil_image.convert("RGB")
        low_res_pil_image = low_res_pil_image.convert("RGB")

        if self.random_crop:
            high_res_arr, _ = random_crop_arr_input_target(
                high_res_pil_image, low_res_pil_image, self.patch_size
            )
            # if is_white(high_res_arr) or is_black(high_res_arr):
            # return self.__getitem__(idx)

        if self.random_flip and random.random() < 0.5:
            high_res_arr = high_res_arr[:, ::-1]
            # low_res_arr = low_res_arr[:, ::-1]

        high_res_arr = high_res_arr.astype(np.float32) / 127.5 - 1
        # low_res_arr = low_res_arr.astype(np.float32) / 127.5 - 1

        out_dict = {}

        new_hight, new_width = self.patch_size // 4, self.patch_size // 4
        downsampled = cv2.resize(high_res_arr, (new_width, new_hight))
        upsampled = cv2.resize(downsampled, (self.patch_size, self.patch_size))

        high_res = np.transpose(high_res_arr, [2, 0, 1])
        low_res = np.transpose(upsampled, [2, 0, 1])

        # downsampled = F.interpolate(high_res, (new_hight, new_width), mode="bilinear")
        # upsampled = F.interpolate(downsampled, (self.patch_size, self.patch_size), mode="bilinear")
        out_dict["low_res"] = low_res
        return high_res, out_dict


def load_superres_data(
    *,
    paths,
    batch_size,
    patch_size,
    deterministic=False,
    random_crop=True,
    random_flip=True,
):
    if not paths:
        raise ValueError("Nothing to load")
    dataset = SuperresImageDataset(
        patch_size,
        paths,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )

    if deterministic:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,
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
        patch_size,
        paths,
        shard=0,
        num_shards=1,
        random_crop=True,
        random_flip=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.local_images = paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        high_res_path = self.local_images[idx]
        low_res_path = high_res_path.replace("high_res", "low_res")

        with bf.BlobFile(high_res_path, "rb") as f:
            high_res_pil_image = Image.open(f)
            high_res_pil_image.load()
        with bf.BlobFile(low_res_path, "rb") as f:
            low_res_pil_image = Image.open(f)
            low_res_pil_image.load()
        if high_res_pil_image.size != low_res_pil_image.size:
            return self.__getitem__(idx + 1)

        high_res_pil_image = high_res_pil_image.convert("RGB")
        low_res_pil_image = low_res_pil_image.convert("RGB")

        if self.random_crop:
            high_res_arr, low_res_arr = random_crop_arr_input_target(
                high_res_pil_image, low_res_pil_image, self.patch_size
            )
        else:
            high_res_arr, low_res_arr = center_crop_arr_input_target(
                high_res_pil_image, low_res_pil_image, self.patch_size
            )

        if self.random_flip and random.random() < 0.5:
            high_res_arr = high_res_arr[:, ::-1]
            low_res_arr = low_res_arr[:, ::-1]

        high_res_arr = high_res_arr.astype(np.float32) / 127.5 - 1
        low_res_arr = low_res_arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        out_dict["low_res"] = np.transpose(low_res_arr, [2, 0, 1])
        return np.transpose(high_res_arr, [2, 0, 1]), out_dict


def is_white(arr):
    white_fraction = np.mean(arr) / 255
    if white_fraction >= 0.99:
        return True
    else:
        return False


def is_black(arr):
    black_fraction = np.mean(arr) / 255
    if black_fraction <= 0.01:
        return True
    else:
        return False


def _file_name(file_path):
    basename = os.path.basename(file_path)
    file_name = os.path.splitext(basename)[0]
    return file_name


def image_files_train_valid_test_split(
    data_dir: str, val_files: list, valid_sample_rate: float = 0.2
):
    assert (valid_sample_rate) < 1, "Invalid train_validation split ratio"
    train_sample_rate = 1 - valid_sample_rate

    paths = []
    for entry in sorted(bf.listdir(data_dir)):
        # if len(paths) < 2: #This is just for faster pipeline testing, comment during actual training
        if not entry.startswith("."):
            full_path = bf.join(data_dir, entry)
            ext = entry.split(".")[-1]
            if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "tif"]:
                paths.append(full_path)
            elif bf.isdir(full_path):
                print("Recursive data loading is not supported")
                # paths.extend(_list_image_files_train_valid_test(full_path))
    valid_paths = []
    train_paths = []

    for path in paths:
        for val_file in val_files:
            if path == bf.join(data_dir, val_file):
                valid_paths.append(path)
                break
        else:
            train_paths.append(path)

    return train_paths, valid_paths


def random_crop_arr_input_target(pil_image_target, pil_image_input, patch_size):

    arr_target = np.array(pil_image_target)
    arr_input = np.array(pil_image_input)
    crop_y = random.randrange(arr_target.shape[0] - patch_size + 1)
    crop_x = random.randrange(arr_target.shape[1] - patch_size + 1)

    arr_target_crop = arr_target[
        crop_y : crop_y + patch_size, crop_x : crop_x + patch_size
    ]
    arr_input_crop = arr_input[
        crop_y : crop_y + patch_size, crop_x : crop_x + patch_size
    ]
    if is_white(arr_target_crop) or is_black(arr_target_crop):
        return random_crop_arr_input_target(
            pil_image_target, pil_image_input, patch_size
        )
    return arr_target_crop, arr_input_crop


def center_crop_arr_input_target(pil_image_target, pil_image_input, patch_size):

    arr_target = np.array(pil_image_target)
    arr_input = np.array(pil_image_input)
    crop_y = (arr_target.shape[0] - patch_size) // 2
    crop_x = (arr_target.shape[1] - patch_size) // 2

    arr_target_crop = arr_target[
        crop_y : crop_y + patch_size, crop_x : crop_x + patch_size
    ]
    arr_input_crop = arr_input[
        crop_y : crop_y + patch_size, crop_x : crop_x + patch_size
    ]
    if is_white(arr_target_crop) or is_black(arr_target_crop):
        return random_crop_arr_input_target(
            pil_image_target, pil_image_input, patch_size
        )
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
