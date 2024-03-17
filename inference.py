import argparse
import copy
import torch
from tqdm import tqdm
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
import importlib
import os
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from train_utils import make_visualization_timestep, InfinityDataset
from v_diffusion import make_beta_schedule


def load_checkpoint(checkpoint_path, device, make_diffusion, make_model):
    ckpt = torch.load(checkpoint_path, map_location=device)
    ema_model = make_model().to(device)
    ema_model.load_state_dict(ckpt['G'])
    
    n_timesteps = ckpt["n_timesteps"]
    time_scale = ckpt["time_scale"]
    
    ema_diffusion_model = make_diffusion(ema_model, n_timesteps, time_scale, device)
    return ema_diffusion_model

def make_argument_parser():
    parser = argparse.ArgumentParser(description="Model Inference and FID Calculation")  
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--name", help="Experiment name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--dname", help="Distillation name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    
    parser.add_argument("--batch_size", help="Batch size for the testset loading.", type=int, default=64)
    parser.add_argument("--num_workers", help="Num workers for dataset loading.", type=int, default=2)
    
    parser.add_argument("--checkpoint_path", required=True, type=str, help="Path to the model checkpoint.")
    parser.add_argument("--num_fid_samples", default=50, type=int, help="Number of images to generate for FID calculation.")
    parser.add_argument("--sampling_timesteps", nargs="+", type=int, default=[256, 512, 1024], help="Timesteps for image sampling.")
    parser.add_argument("--num_images_to_log", type=int, default=16, help="How many images to save for logging purposes?.")
    return parser
    
def run_inference(args, make_model, make_dataset):
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    
    checkpoints_dir = os.path.join("checkpoints", args.name, args.dname)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
    def make_diffusion(model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("v_diffusion")
        D = getattr(M, "GaussianDiffusionDefault")
        return D(model, betas, time_scale=time_scale)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    ema_model = make_model().to(device)
    image_size = ema_model.image_size
    ema_model.load_state_dict(ckpt['G'])
    
    n_timesteps = ckpt["n_timesteps"]
    time_scale = ckpt["time_scale"]
    
    ema_diffusion_model = make_diffusion(ema_model, n_timesteps, time_scale, device)
    if 'mnist' in args.checkpoint_path:
        args.module = 'mnist_model'
    elif 'cifar10' in args.checkpoint_path:
        args.module = 'cifar10_model'
    
    tensorboard = SummaryWriter(os.path.join(checkpoints_dir, "tensorboard"))
    
    # we only need args.num_fid_samples from the dataset
    test_dataset = InfinityDataset(make_dataset(train=False), args.num_fid_samples)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    # First, initialize FID objects with the real images from the test set
    base_fid = FrechetInceptionDistance(normalize=True, reset_real_features=False)
    base_fid.to(device)
    base_fid.persistent(mode=True)
    base_fid.requires_grad_(False)
    
    for images, _ in tqdm(test_loader, desc='Loading desired amount of test set onto FID object'):
        base_fid.update(images.to(device), real=True)
    
    fids = {}
    for sampling_timestep in args.sampling_timesteps:
        fid = copy.deepcopy(base_fid)
        fids[sampling_timestep] = fid
    
    # Now, generate images all at once, but we save intermediate steps for efficiency
    num_batches = (args.num_fid_samples + args.batch_size - 1) // args.batch_size
    for _ in tqdm(range(num_batches), desc='Generating images in batches'):
        batch_size = min(args.batch_size, args.num_fid_samples - _ * args.batch_size)
        images = make_visualization_timestep(ema_diffusion_model, device, image_size, args.sampling_timesteps, batch_size=batch_size, need_tqdm=False, eta=0, clip_value=1.2)
        # Process generated batch images as needed

                    

    # Calculate FID score for each of the sampling_timesteps
    # and save to tensorboard
    for timestep, fid in fids.items():
        fid_score = fid.compute()
        print(f"FID score for {timestep} timesteps: {fid_score}")
        tensorboard.add_scalar(f"fid_{timestep}", fid_score, timestep)

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")
    make_dataset = getattr(M, "make_dataset")

    run_inference(args, make_model, make_dataset)
