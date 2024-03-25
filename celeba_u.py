from train_utils import *
from unet_ddpm import UNet
from celeba_dataset import CelebaWrapper

BASE_NUM_STEPS = 1024
BASE_TIME_SCALE = 1

def make_model(vae_name=None):
    if vae_name in ['ft-mse', 'v1-4']:
        net = UNet(
            in_channel = 4,
            out_channel = 4,
            channel = 128-16,
            channel_multiplier = [1, 2, 2, 4, 4],
            n_res_blocks = 2,
            attn_strides = [8, 16],
            attn_heads = 4,
            use_affine_time = True,
            dropout = 0,
            fold = 1
        )
        net.image_size = [1, 4, 32, 32]
    else:
        net = UNet(
            in_channel = 3,
            out_channel = 3,
            channel = 128-16,
            channel_multiplier = [1, 2, 2, 4, 4],
            n_res_blocks = 2,
            attn_strides = [8, 16],
            attn_heads = 4,
            use_affine_time = True,
            dropout = 0,
            fold = 1
        )
        net.image_size = [1, 3, 256, 256]
    return net

def make_dataset(train=True, vae_handler=None):
    return CelebaWrapper(dataset_dir="./data/celeba_256/", resolution=256, vae_handler=vae_handler)

