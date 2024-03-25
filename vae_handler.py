import torch

from diffusers import AutoencoderKL

class VAEHandler:
    def __init__(self, model_path, device='cuda'):
        self.model = self.load_model(model_path)
        self.model.to(device)
        self.device = device
        self.model.eval()
        self.scale_factor = 1

    def load_model(self, model_path): # can also be a local file
        if 'CompVis' in model_path:
            model = AutoencoderKL.from_pretrained(model_path, subfolder='vae')
            self.scale_factor = 0.18215
        else:
            model = AutoencoderKL.from_single_file(model_path)
        return model

    def decode(self, latents):
        if isinstance(latents, list):
            return [self.decode_item(latent) for latent in latents]
        else:
            return self.decode_item(latents)

    def decode_item(self, latent):
        latent = (1 / self.scale_factor) * latent.to(self.device).float()
        with torch.no_grad():
            decoded_output = self.model.decode(latent)
            decoded_image = decoded_output.sample  # Get the decoded image tensor
            decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)  # Normalize the images
            decoded_image = decoded_image.detach()
        return decoded_image

    def encode(self, images):
        if isinstance(images, list):
            return [self.encode_item(image) for image in images]
        else:
            return self.encode_item(images)

    def encode_item(self, image):
        image = image.to(self.device)
        if image.dim() == 3:  # If a single image without a batch dimension, add it
            image = image.unsqueeze(0)
        image = image * 2 - 1  # Adjusting for the image scale [-1, 1] before encoding
        with torch.no_grad():
            encoded_output = self.model.encode(image)
            latents = self.scale_factor * encoded_output.latent_dist.sample()  # Scale the latents
        return latents.squeeze(0)  # Remove batch dimension for single items

    @staticmethod
    def transform_image(image):
        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Assuming you want to resize the image
            transforms.ToTensor(),
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image

    @staticmethod
    def to_pil_image(tensor):
        transform = transforms.ToPILImage()
        return transform(tensor.squeeze(0))  # Remove batch dimension