# guided-diffusion-tissues

This is the codebase for the project on improving the resolution of tissue images. 

We aim to try various CNN models, including recently reliazed guided diffusion models. We heavily rely on the codebase for [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233) [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

## Requirements
* Python 3.9

## Prerequisites
```shellscript
pip install -e .
```
```shellscript
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib
```

## MPI
For running with `mpi`, you need to install it. I did `conda install -c conda-forge mpi4py`. 

# Data exploration

Check *notebooks*. 
We decided to divide the dataset to the following groups: 70% training, 20% validation, 10% testing.
Validation dataset will be used to visualize intermediate results, testing will be held out until the very end.

Validation samples:
```
'Slide002-2.tif', 'Slide003-2.tif', 'Slide005-1.tif', 'Slide008-1.tif', 'Slide008-2.tif', 'Slide010-1.tif', 'Slide011-1.tif', 'Slide011-5.tif', 'Slide011-6.tif', 'Slide019-3.tif', 'Slide022-1.tif', 'Slide022-3.tif', 'Slide023-3.tif', 'Slide025-1.tif', 'Slide028-1.tif', 'Slide029-3.tif', 'Slide030-1.tif', 'Slide032-3.tif', 'Slide036-1.tif', 'Slide036-2.tif', 'Slide037-2.tif', 'Slide039-1.tif', 'Slide042-1.tif', 'Slide044-3.tif', 'Slide046-3.tif', 'Slide047-2.tif', 'Slide053-1.tif'
```

Testing samples:
```
'Slide008-3.tif', 'Slide011-4.tif', 'Slide013-2.tif', 'Slide014-2.tif', 'Slide019-1.tif', 'Slide019-2.tif', 'Slide022-4.tif', 'Slide031-1.tif', 'Slide034-3.tif', 'Slide035-1.tif', 'Slide044-2.tif', 'Slide045-1.tif', 'Slide045-2.tif', 'Slide045-3.tif', 'Slide052-2.tif'
```

The IDs are already included as agruments in `scripts/super_res_training.py` 

# Training guided-diffusion model

You need to specify number of GPUs `-n`, location for storing models and all logger files `--logdir`. Other parameters were taken from the paper mentioned above.
For trying a model, I suggest to change the following arguments:
`--log_interval 1`
`--save_interval 1`
`--tb_valid_im_num 1`

After each `--save_interval`, the validation set is visualized to tensorboard. `--tb_valid_im_num` defines number of validation patches displayed in tensorboard. I wanted to add also assessment of the validation losses, but for some reason I have a CUDA error (will deal with this later).

Training from the scratch
```sh 
NCCL_P2P_DISABLE=1 mpiexec -n 8 python3 scripts/super_res_train.py \
--model_path log/model240000.pt \
--patch_size 256 \
--data_dir /baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/high_res \
--log_dir log \
--diffusion_steps 1000 \
--noise_schedule "linear" \
--num_channels 192 \
--num_res_blocks 2 \
--num_head_channels 64 \
--attention_resolutions "32,16,8" \
--lr 1e-4 \
--log_interval 1000 --save_interval 10000 \
--batch_size 4 --tb_valid_im_num 8
```

Pre-training
```sh
NCCL_P2P_DISABLE=1 mpiexec -n 8 python3 scripts/super_res_train.py \
--patch_size 256 \
--data_dir /home/kgw/guided-diffusion-tissues/ILSVRC/Data/CLS-LOC/test \
--val_data_dir /home/kgw/guided-diffusion-tissues/ILSVRC/Data/CLS-LOC/val \
--log_dir pre-train-log-256 \
--diffusion_steps 1000 \
--noise_schedule "linear" \
--num_channels 192 \
--num_res_blocks 2 \
--num_head_channels 64 \
--attention_resolutions "32,16,8" \
--lr 1e-4 \
--log_interval 100 --save_interval 10000 \
--batch_size 2 --tb_valid_im_num 8 \
--pre_train True
```

Resume training
```sh
NCCL_P2P_DISABLE=1 mpiexec -n 8 python3 scripts/super_res_train.py \
--resume_checkpoint log-fine-tune/model640000.pt \
--patch_size 256 \
--data_dir /baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/high_res \
--log_dir log-fine-tune \
--diffusion_steps 1000 \
--noise_schedule "linear" \
--num_channels 192 \
--num_res_blocks 2 \
--num_head_channels 64 \
--attention_resolutions "32,16,8" \
--lr 1e-4 \
--log_interval 100 --save_interval 10000 \
--batch_size 14 --tb_valid_im_num 8
```

Inference
```sh
NCCL_P2P_DISABLE=1 mpiexec -n 8 python3 scripts/super_res_sample.py \
--model_path log-fine-tune/model670000.pt \
--diffusion_steps 1000 \
--noise_schedule "linear" \
--num_channels 192 \
--num_res_blocks 2 \
--num_head_channels 64 \
--attention_resolutions "32,16,8" \
--base_samples /baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide022-1.tif \
--log_dir test-result-256
```

Sample patches
```
NCCL_P2P_DISABLE=1 mpiexec -n 8 python3 scripts/super_res_sample_patches.py \
--patch_size 256 \
--model_path log/model080000.pt \
--diffusion_steps 1000 \
--noise_schedule "linear" \
--num_channels 192 \
--num_res_blocks 2 \
--num_head_channels 64 \
--attention_resolutions "32,16,8" \
--base_samples /baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/low_res/Slide022-1.tif \
--log_dir test-result-256
```

UNet training
```
python3 unet_train.py /baldig/bioprojects2/BrownLab/Ptychography/Registered_Images2/high_res --gpus 1 --epochs 10000
```
> Things to consider:
>
> Batch size is small because CUDA runs out of memory. Needs to be trained on larger nodes.
>
> Patch size is also small, however, if making it larger, the batch size should be even smaller.
>
> For now loss is just MSE. Maybe later we can add variational lower bound to the loss function.

# Running tensorboard
Go to the folder, which you indicated in `--log_dir`
```sh
tensorboard --logdir runs/ --port=6009 --bind_all
```
The first raw of images correspond to low resolution input, the second row - high resolution target, the third row = prediction.


