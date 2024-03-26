# R255: Improving Progressive Distillation in Diffusion Models

This repository holds the code necessary to
- train teacher diffusion models (`train.py`)
- distillate knowledge into student models (`distillate.py`)
- run models for inference to calculate FID (`inference.py`)

The following is supported by the pipeline:
- FID calculation
- Tensorboard logging (loss, progression of samples, final sample results, fid)
- 3 dataset + accompanying models (Cifar10, MNIST, Celeba 256x256)
- Autoencoder integration (`CompVis/stable-diffusion-v1-4`)

- 
