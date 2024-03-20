import torch

from diffusers import AutoencoderKL

class VAEHandler:
    def __init__(self, model_path, device='cuda'):
        self.model = self.load_model(model_path)
        self.model.to(device)
        self.device = device
        self.model.eval()

    def load_model(self, model_path): # can also be a local file
        model = AutoencoderKL.from_single_file(model_path)
        return model

    def decode(self, images):
        # Assuming images is a batch of latents ([4, 32, 32])
        with torch.no_grad():
            decoded_output = self.model.decode(images)
            decoded_output_samples = decoded_output.sample  # Get the decoded image tensor
        return decoded_output_samples    
    
    def encode(self, images):
        # Assuming images is a batch of latents ([4, 32, 32])
        with torch.no_grad():
            encoded_output = self.model.encode(images)
            latents = encoded_output.latent_dist.mean
        return latents
    
    def encode_item(self, img):
        img = img.to(self.device)
        encoded_output = self.model.encode(img.unsqueeze(0))  # Add batch dimension
        latent = encoded_output.latent_dist.mean.squeeze(0)  # Remove batch dimension
        return latent

