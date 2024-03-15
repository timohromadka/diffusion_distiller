from train_utils import *
from unet_ddpm import UNet
from mnist_dataset import MNISTWrapper

BASE_NUM_STEPS = 1024
BASE_TIME_SCALE = 1

def make_model():
    net = UNet(in_channel=1,  # Grayscale images for MNIST
               channel=128-16,
               channel_multiplier=[1, 2, 2, 4, 4],
               n_res_blocks=2,
               attn_strides=[8, 16],
               attn_heads=4,
               use_affine_time=True,
               dropout=0,
               fold=1)
    net.image_size = [1, 1, 28, 28]  # Adjusted for MNIST's 28x28 images
    return net

def make_dataset(train=True):
    return MNISTWrapper(dataset_dir="./data/mnist/", train=train)
