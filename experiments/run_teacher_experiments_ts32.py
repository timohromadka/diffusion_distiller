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
    {"module": "mnist_model", "name": "mnist_exp", "dname": "training_teacher_ts32"},
    {"module": "cifar10_model", "name": "cifar10_exp", "dname": "training_teacher_ts32"},
]

common_params = "--num_timesteps 32 --num_iters 30000 --batch_size 64 --lr 0.0002 --ckpt_step_interval 1000 --log_step_interval 500 --num_workers 2"

for exp in experiments:
    cmd = f"python train.py --module {exp['module']} --name {exp['name']} --dname {exp['dname']} {common_params}"
    print(f"Executing command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"Experiment with module {exp['module']} and name {exp['name']} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Experiment with module {exp['module']} and name {exp['name']} failed. Error: {e}")

print("All experiments completed.")

