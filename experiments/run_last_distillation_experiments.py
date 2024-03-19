import subprocess

commands = [
    # MNIST DISTILLATION EXPERIMENTS

    # divide by 4
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_4_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/64_16_sf4/checkpoint_1000.pt --batch_size 64 --num_workers 1 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 4_1_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/16_4_sf4/checkpoint_1000.pt --batch_size 64 --num_workers 1 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",

    # CIFAR10 DISTILLATION EXPERIMENTS

    # divide by 4
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 16_4_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/cifar10_exp/64_16_sf4/checkpoint_1000.pt --batch_size 64 --num_workers 1 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",
    "python distillate.py --module cifar10_model --diffusion GaussianDiffusionDefault --name cifar10_exp --dname 4_1_sf4 --skip_factor 4 --base_checkpoint ./checkpoints/cifar10_exp/16_4_sf4/checkpoint_1000.pt --batch_size 64 --num_workers 1 --num_iters 1000 --ckpt_step_interval 500 --log_step_interval 500",


]

for cmd in commands:
    subprocess.call(cmd, shell=True)

# nohup python experiments/run_last_distillation_experiments.py > logs/last_distillation_experiments.log 2>&1 &
