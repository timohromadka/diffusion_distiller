import subprocess

commands = [
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_512 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 512_256 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/1024_512/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 256_128 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/512_256/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 128_64 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/256_128/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 64_32 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/128_64/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 32_16 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/64_32/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_8 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/32_16/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 8_4 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/16_8/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 4_2 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/8_4/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 2_1 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/4_2/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500", 

    # divide by 4
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_256 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 256_64 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/1024_256/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 64_16 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/256_64/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_4 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/64_16/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 4_1 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/16_4/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # divide by 8
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_128 --skip_factor 8 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 128_16 --skip_factor 8 --base_checkpoint ./checkpoints/mnist_exp/1024_128/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_2 --skip_factor 8 --base_checkpoint ./checkpoints/mnist_exp/128_16/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # divide by 16
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_64 --skip_factor 16 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 64_4 --skip_factor 16 --base_checkpoint ./checkpoints/mnist_exp/1024_64/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # divide by 32
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_32 --skip_factor 32 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 32_1 --skip_factor 32 --base_checkpoint ./checkpoints/mnist_exp/1024_32/checkpoint_1000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # divide by 256
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_4 --skip_factor 256 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500"

]

for cmd in commands:
    subprocess.call(cmd, shell=True)
