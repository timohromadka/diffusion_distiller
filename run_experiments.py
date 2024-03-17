# This script is intended to run multiple experiments in succession
# Usage:
# nohup python run_experiments.py > experiments.log 2>&1 &

# to kill the nohup process
# ps -ef | grep run_experiments.py
# (the 2nd column contains PIDs)
# kill <PID>

# or directly as python command
# nohup python train.py --module mnist_model --name mnist_exp --dname teacher --num_timesteps 1024 --num_iters 20000 --batch_size 64 --lr 0.0002 --ckpt_step_interval 500 --log_step_interval 500 --num_workers 2 > logs/teacher_mnist.log 2>&1 &
# nohup python train.py --module cifar10_model --name cifar10_exp --dname teacher --num_timesteps 1024 --num_iters 30000 --batch_size 64 --lr 0.0002 --ckpt_step_interval 500 --log_step_interval 500 --num_workers 2 > logs/teacher_cifar10.log 2>&1 &

# then do
# tensorboard --logdir ./
# in the checkpoints dir
import subprocess

experiments = [
    {"module": "mnist_model", "name": "mnist_exp", "dname": "training_teacher"},
    {"module": "cifar10_model", "name": "cifar10_exp", "dname": "training_teacher"},
]

common_params = "--num_timesteps 1024 --num_iters 50000 --batch_size 64 --lr 0.0002 --ckpt_step_interval 500 --log_step_interval 500 --num_workers 8"

for exp in experiments:
    cmd = f"python train.py --module {exp['module']} --name {exp['name']} --dname {exp['dname']} {common_params}"
    print(f"Executing command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"Experiment with module {exp['module']} and name {exp['name']} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Experiment with module {exp['module']} and name {exp['name']} failed. Error: {e}")

print("All experiments completed.")


# distillation experiment:

# divide by 2
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_512 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_1024_512.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 512_256 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/1024_512/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_512_256.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 256_128 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/512_256/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_256_128.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 128_64 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/256_128/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_128_64.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 64_32 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/128_64/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_64_32.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 32_16 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/64_32/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_32_16.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_8 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/32_16/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_16_8.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 8_4 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/16_8/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_8_4.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 4_2 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/8_4/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_4_2.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 2_1 --skip_factor 2 --base_checkpoint ./checkpoints/mnist_exp/4_2/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_2_1.log 2>&1 &

# # divide by 4
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_256 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_1024_256.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 256_64 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/1024_256/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_256_64.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 64_16 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/256_64/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_64_16.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_4 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/64_16/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_16_4.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 4_1 --skip_factor 4 --base_checkpoint ./checkpoints/mnist_exp/16_4/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_4_1.log 2>&1 &

# # divide by 8
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_128 --skip_factor 8 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_1024_128.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 128_16 --skip_factor 8 --base_checkpoint ./checkpoints/mnist_exp/1024_128/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_128_16.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 16_2 --skip_factor 8 --base_checkpoint ./checkpoints/mnist_exp/128_16/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_16_2.log 2>&1 &

# # divide by 16
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_64 --skip_factor 16 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_1024_64.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 64_4 --skip_factor 16 --base_checkpoint ./checkpoints/mnist_exp/1024_64/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_64_4.log 2>&1 &

# # divide by 32
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_32 --skip_factor 32 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_1024_32.log 2>&1 &
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 32_1 --skip_factor 32 --base_checkpoint ./checkpoints/mnist_exp/1024_32/checkpoint_2000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_32_1.log 2>&1 &

# # divide by 256
# nohup python distillate.py --module mnist_model --diffusion GaussianDiffusionDefault --name mnist_exp --dname 1024_4 --skip_factor 256 --base_checkpoint ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --batch_size 64 --num_workers 4 --num_iters 2000 --ckpt_step_interval 500 --log_step_interval 500 > logs/mnist_1024_4.log 2>&1 &
