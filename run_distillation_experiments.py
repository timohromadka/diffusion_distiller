import subprocess

commands = [
    # MNIST DISTILLATION EXPERIMENTS
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_512_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 512_256_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/1024_512_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 256_128_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/512_256_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 128_64_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/256_128_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 64_32_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/128_64_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 32_16_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/64_32_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_8_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/32_16_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 8_4_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/16_8_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 4_2_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/8_4_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 2_1_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/4_2_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500", 

    # divide by 4
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_256_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 256_64_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/1024_256_sf4/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 64_16_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/256_64_sf4/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_4_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/16_4_sf4/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 4_1_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/4_1_sf4/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # divide by 8
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_128_sf8 --skip_factor 8 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 128_16_sf8 --skip_factor 8 --base_checkpoint ./checkpoints/mnist_exp/1024_128_sf8/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_2_sf8 --skip_factor 8 --base_checkpoint ./checkpoints/mnist_exp/128_16_sf8/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # divide by 16
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_64_sf16 --skip_factor 16 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 64_4_sf16 --skip_factor 16 --base_checkpoint ./checkpoints/mnist_exp/1024_64_sf16/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # divide by 32
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_32_sf32 --skip_factor 32 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 32_1_sf32 --skip_factor 32 --base_checkpoint ./checkpoints/mnist_exp/1024_32_sf32/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # divide by 256
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_4_sf256 --skip_factor 256 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",


    # ============================================================================================================
    
    # CIFAR10 EXPERIMENTS
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 1024_512_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 512_256_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/1024_512_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 256_128_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/512_256_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 128_64_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/256_128_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 64_32_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/128_64_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 32_16_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/64_32_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 16_8_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/32_16_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 8_4_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/16_8_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 4_2_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/8_4_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 2_1_sf2 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/4_2_sf2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500", 

    # divide by 4
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 1024_256_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 256_64_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/cifar10_exp/1024_256_sf4/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 64_16_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/cifar10_exp/256_64_sf4/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 16_4_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/cifar10_exp/16_4_sf4/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 4_1_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/cifar10_exp/4_1_sf4/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # divide by 8
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 1024_128_sf8 --skip_factor 8 --base_checkpoint ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 128_16_sf8 --skip_factor 8 --base_checkpoint ./checkpoints/cifar10_exp/1024_128_sf8/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 16_2_sf8 --skip_factor 8 --base_checkpoint ./checkpoints/cifar10_exp/128_16_sf8/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # divide by 16
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 1024_64_sf16 --skip_factor 16 --base_checkpoint ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 64_4_sf16 --skip_factor 16 --base_checkpoint ./checkpoints/cifar10_exp/1024_64_sf16/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # divide by 32
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 1024_32_sf32 --skip_factor 32 --base_checkpoint ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 32_1_sf32 --skip_factor 32 --base_checkpoint ./checkpoints/cifar10_exp/1024_32_sf32/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # divide by 256
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 1024_4_sf256 --skip_factor 256 --base_checkpoint ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500"

]

for cmd in commands:
    subprocess.call(cmd, shell=True)
