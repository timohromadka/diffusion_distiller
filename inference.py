#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import importlib
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from train_utils import InfinityDataset, make_dataset

def convert_grayscale_to_rgb(image):
    """Converts a 1-channel grayscale image to a 3-channel RGB image."""
    return image.repeat(3, 1, 1) if image.shape[0] == 1 else image

def generate_images(model, device, num_samples, timestep, previously_generated_images=None):
    """
    Generates images using the model for the specified timestep.
    If applicable, reuses previously generated images.
    """
    # Check if we can reuse previously generated images
    if previously_generated_images is not None and previously_generated_images["timestep"] <= timestep:
        return previously_generated_images["images"]

    # Example placeholder function for generating images. Replace with your actual image generation logic.
    # Assuming the model generates images in the shape: [num_samples, 3, 256, 256]
    generated_images = torch.randn(num_samples, 3, 256, 256, device=device)  # Example tensor shape, adjust accordingly.
    
    # Store newly generated images
    return {"timestep": timestep, "images": generated_images}

def compute_fid_for_timesteps(model, device, test_loader, fid, args):
    """Computes FID for each timestep specified in args.sampling_timesteps, reusing generated images where possible."""
    previously_generated_images = None
    for timestep in sorted(args.sampling_timesteps):
        # Reset real features for each timestep
        fid.reset_real_features()
        with torch.no_grad():
            # Update FID with features from test dataset
            for batch in test_loader:
                images, _ = batch
                images = images.to(device)
                images = torch.stack([convert_grayscale_to_rgb(image) for image in images])
                fid.update(images, real=True)
            
            # Generate or reuse images and update FID
            previously_generated_images = generate_images(model, device, args.num_fid_samples, timestep, previously_generated_images)
            generated_images = previously_generated_images["images"]
            generated_images = torch.stack([convert_grayscale_to_rgb(image) for image in generated_images])
            fid.update(generated_images, real=False)
        
        # Compute FID score
        fid_score = fid.compute()
        print(f"FID score for timestep {timestep}: {fid_score}")

def main():
    parser = argparse.ArgumentParser(description='Generate images and compute FID against a test dataset.')
    parser.add_argument('--num_fid_samples', type=int, required=True, help='Number of images to generate for FID calculation.')
    parser.add_argument('--sampling_timesteps', type=int, nargs='+', required=True, help='List of timesteps for image generation.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Placeholder for loading your model
    model = torch.nn.Module()  # Replace with your model loading logic
    model.to(device)

    # Initialize FID
    fid = FrechetInceptionDistance(normalize=True, reset_real_features=False)
    fid.persistent(mode=True)
    fid.requires_grad_(False)

    # Load test dataset
    test_dataset = InfinityDataset(make_dataset(train=False), len(args.sampling_timesteps) * args.num_fid_samples)
    test_loader = DataLoader(test_dataset, batch_size=args.num_fid_samples, shuffle=False)

    compute_fid_for_timesteps(model, device, test_loader, fid, args)

if __name__ == "__main__":
    main()
