# R255: Improving Progressive Distillation in Diffusion Models

This is the code repository for the ACS R255 Project. The initial code base was forked from [Hramchenko's repository]([url](https://github.com/Hramchenko/diffusion_distiller)). However, I needed to modify it to test and run my own experiments. My modifications to the code are:

- Integrated MNIST and Cifar10 support
- Fixed bugs within UNet class
- Integrated skip-factors for experimentation
- Added FID calculation and logging
- Incorporated improved tensorboard logging

This repository holds the code necessary to
- train teacher diffusion models (`train.py`)
- distillate knowledge into student models (`distillate.py`)
- run models for inference to calculate FID (`inference.py`)

The following is supported by the pipeline:
- FID calculation
- Tensorboard logging (loss, progression of samples, final sample results, fid)
- 3 datasets + accompanying models (Cifar10, MNIST, Celeba 256x256)
- Autoencoder integration (`CompVis/stable-diffusion-v1-4`)

## How to use
To train a teacher model, one may use the following example command:

```
python train.py --module mnist_model --name mnist_exp --dname training_teacher --num_timesteps 1024 --num_iters 30000 --batch_size 64 --lr 0.0002 --ckpt_step_interval 2000 --log_step_interval 500 --num_workers 2
```

To distillate the knowledge from a teacher model into a student model, one may use the following example command:
```
python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_512_sf2_ni3000 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 2 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 100
```

Finally, to run a model for inference to calculate FID score, one may use the following example command:
```
python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 32 --checkpoint_path ./checkpoints/mnist_exp/1024_512_sf2/checkpoint_1000.pt --dname 1024_512_sf2_inference --sampling_timesteps 4 64 128 512
```

