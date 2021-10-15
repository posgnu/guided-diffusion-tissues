import math
import random
import os

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

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
        raise ValueError("unspecified paths")
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
            patch_size,
            paths,
            shard=0,
            num_shards=1,
            random_crop=True,
            random_flip=True,
            ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

        def __len__(self):
            return len(self.local_images)

        def __getitem__(self, idx):
            high_res_path = self.local_images[idx]
            low_res_path = high_res_path.replace('high_res', 'low_res')

            with bf.BlobFile(high_res_path, "rb") as f:
                high_res_pil_image = Image.open(f)
            with bf.BlobFile(low_res_path, "rb") as f:
                low_res_pil_image = Image.open(f)
            
            if high_res_pil_image.size != low_res_pil_image:
                return self.__getitem__(self, idx+1)

            high_res_pil_image.load()
            low_res_pil_image.load()

            high_res_pil_image = high_res_pil_image.convert("RGB")
            low_res_pil_image = low_res_pil_image.convert("RGB")

            if self.random_crop:
                high_res_arr, low_res_arr = random_crop_arr_input_target(
                        high_res_pil_image,
                        low_res_pil_image,
                        patch_size)
                if is_white(high_res_arr):
                    return self.__getitem__(self, idx)

            if self.random_flip and random.random() < 0.5:
                high_res_arr = high_res_arr[:, ::-1]
                low_res_arr = low_res_arr[:, ::-1]

            high_res_arr = high_res_arr.astype(np.float32) / 127.5 - 1
            low_res_arr = low_res_arr.astype(np.float32) / 127.5 - 1

            out_dict = {}
            out_dict["low_res"] = np.transpose(low_res_arr, [2, 0, 1])
            return np.transpose(high_res_arr, [2, 0, 1]), out_dict

def is_white(arr):
    number_of_white_pix = np.sum(arr = [255, 255, 255])
    all_pix = arr,shape[0] * arr.shape[1]
    white_fraction = number_of_white_pix / all_pix
    if white_fraction >= 0.95 :
        return True
    else:
        return False
    
def _file_name(file_path):
    basename = os.path.basename(file_path)
    file_name = os.path.splitext(basename)[0]
    return file_name

def _list_image_files_train_valid_test(data_dir, valid_samples, test_samples):
    paths = []
    for entry in sorted(bf.listdir(data_dir)):
        if len(paths) < 2: % This is just for faster pipeline testing, comment during actual training
            full_path = bf.join(data_dir, entry)
            ext = entry.split(".")[-1]
            if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "tif"]:
                paths.append(full_path)
            elif bf.isdir(full_path):
                paths.extend(_list_image_files_train_valid_test(full_path)) 
        else:
            break

    test_paths = [path for path in paths if _file_name(path) in test_samples]
    valid_paths = [path for path in paths if _file_name(path) in valid_samples]
    train_paths = [path for path in paths if (path not in test_paths) and (path not in valid_paths)]
    
    return train_paths, valid_paths, test_paths

def random_crop_arr_input_target(pil_image_target, pil_image_input, patch_size):

    arr_target = np.array(pil_image_target)
    arr_input = np.array(pil_image_input) 
    crop_y = random.randrange(arr_target.shape[0] - patch_size + 1)
    crop_x = random.randrange(arr_target.shape[1] - patch_size + 1)
    return arr_target[crop_y : crop_y + patch_size, crop_x : crop_x + patch_size],
            arr_input[crop_y : crop_y + patch_size, crop_x : crop_x + patch_size]

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
