import subprocess

commands = [
    # # MNIST DISTILLATION EXPERIMENTS
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_512_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 512_256_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/1024_512_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 256_128_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/512_256_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 128_64_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/256_128_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 64_32_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/128_64_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 32_16_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/64_32_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_8_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/32_16_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 8_4_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/16_8_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 4_2_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/8_4_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 2_1_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/4_2_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32", 

    # # divide by 4
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_256_sf4_ni32 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 256_64_sf4_ni32 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/1024_256_sf4_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 64_16_sf4_ni32 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/256_64_sf4_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_4_sf4_ni32 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/64_16_sf4_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 4_1_sf4_ni32 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/16_4_sf4_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",

    # # divide by 8
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_128_sf8_ni32 --skip_factor 8 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 128_16_sf8_ni32 --skip_factor 8 --base_checkpoint ./checkpoints/mnist_exp/1024_128_sf8_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_2_sf8_ni32 --skip_factor 8 --base_checkpoint ./checkpoints/mnist_exp/128_16_sf8_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",

    # # divide by 16
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_64_sf16_ni32 --skip_factor 16 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 64_4_sf16_ni32 --skip_factor 16 --base_checkpoint ./checkpoints/mnist_exp/1024_64_sf16_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",

    # # divide by 32
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_32_sf32_ni32 --skip_factor 32 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 32_1_sf32_ni32 --skip_factor 32 --base_checkpoint ./checkpoints/mnist_exp/1024_32_sf32_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",

    # # divide by 256
    # "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_4_sf256_ni32 --skip_factor 256 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",


    # # ============================================================================================================
    
    # # CIFAR10 EXPERIMENTS
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 1024_512_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 512_256_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/1024_512_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 256_128_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/512_256_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 128_64_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/256_128_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 64_32_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/128_64_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 32_16_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/64_32_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 16_8_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/32_16_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 8_4_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/16_8_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 4_2_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/8_4_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 2_1_sf2_ni32 --skip_factor 2 --base_checkpoint ./checkpoints/cifar10_exp/4_2_sf2_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32", 

    # # divide by 4
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 1024_256_sf4_ni32 --skip_factor 4 --base_checkpoint ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 256_64_sf4_ni32 --skip_factor 4 --base_checkpoint ./checkpoints/cifar10_exp/1024_256_sf4_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 64_16_sf4_ni32 --skip_factor 4 --base_checkpoint ./checkpoints/cifar10_exp/256_64_sf4_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 16_4_sf4_ni32 --skip_factor 4 --base_checkpoint ./checkpoints/cifar10_exp/64_16_sf4_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 4_1_sf4_ni32 --skip_factor 4 --base_checkpoint ./checkpoints/cifar10_exp/16_4_sf4_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",

    # # divide by 8
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 1024_128_sf8_ni32 --skip_factor 8 --base_checkpoint ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 128_16_sf8_ni32 --skip_factor 8 --base_checkpoint ./checkpoints/cifar10_exp/1024_128_sf8_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 16_2_sf8_ni32 --skip_factor 8 --base_checkpoint ./checkpoints/cifar10_exp/128_16_sf8_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",

    # # divide by 16
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 1024_64_sf16_ni32 --skip_factor 16 --base_checkpoint ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 64_4_sf16_ni32 --skip_factor 16 --base_checkpoint ./checkpoints/cifar10_exp/1024_64_sf16_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",

    # # divide by 32
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 1024_32_sf32_ni32 --skip_factor 32 --base_checkpoint ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 32_1_sf32_ni32 --skip_factor 32 --base_checkpoint ./checkpoints/cifar10_exp/1024_32_sf32_ni32/checkpoint_32.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32",

    # # divide by 256
    # "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 1024_4_sf256_ni32 --skip_factor 256 --base_checkpoint ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 32 --ckpt_step_interval 32 --log_step_interval 32"

]

for cmd in commands:
    subprocess.call(cmd, shell=True)

# nohup python run_distillation_experiments_ni32_full.py > logs/distillation_experiments_ni32_full.log 2>&1 &
