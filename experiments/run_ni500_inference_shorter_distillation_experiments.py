import subprocess

commands = [
    # MNIST INFERENCE EXPERIMENTS

    # students sf2
    "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/1024_512_sf2/checkpoint_500.pt --dname 1024_512_sf2_ni500_inference --sampling_timesteps 4 64 128 512",
    "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/512_256_sf2/checkpoint_500.pt --dname 512_256_sf2_ni500_inference --sampling_timesteps 4 64 128 256",
    "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/256_128_sf2/checkpoint_500.pt --dname 256_128_sf2_ni500_inference --sampling_timesteps 4 32 64 128",
    "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/128_64_sf2/checkpoint_500.pt --dname 128_64_sf2_ni500_inference --sampling_timesteps 4 16 32 64",
    "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/64_32_sf2/checkpoint_500.pt --dname 64_32_sf2_ni500_inference --sampling_timesteps 4 8 16 32",
    "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/32_16_sf2/checkpoint_500.pt --dname 32_16_sf2_ni500_inference --sampling_timesteps 2 4 8 16",
    "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/16_8_sf2/checkpoint_500.pt --dname 16_8_sf2_ni500_inference --sampling_timesteps 1 2 4 8",
    "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/8_4_sf2/checkpoint_500.pt --dname 8_4_sf2_ni500_inference --sampling_timesteps 1 2 4",
    "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/4_2_sf2/checkpoint_500.pt --dname 4_2_sf2_ni500_inference --sampling_timesteps 1 2",
    "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/2_1_sf2/checkpoint_500.pt --dname 2_1_sf2_ni500_inference --sampling_timesteps 1",

    # CIFAR10 INFERENCE EXPERIMENTS
    
    # students sf2
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/1024_512_sf2/checkpoint_500.pt --dname 1024_512_sf2_ni500_inference --sampling_timesteps 4 64 128 512",
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/512_256_sf2/checkpoint_500.pt --dname 512_256_sf2_ni500_inference --sampling_timesteps 4 64 128 256",
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/256_128_sf2/checkpoint_500.pt --dname 256_128_sf2_ni500_inference --sampling_timesteps 4 32 64 128",
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/128_64_sf2/checkpoint_500.pt --dname 128_64_sf2_ni500_inference --sampling_timesteps 4 16 32 64",
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/64_32_sf2/checkpoint_500.pt --dname 64_32_sf2_ni500_inference --sampling_timesteps 4 8 16 32",
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/32_16_sf2/checkpoint_500.pt --dname 32_16_sf2_ni500_inference --sampling_timesteps 2 4 8 16",
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/16_8_sf2/checkpoint_500.pt --dname 16_8_sf2_ni500_inference --sampling_timesteps 1 2 4 8",
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/8_4_sf2/checkpoint_500.pt --dname 8_4_sf2_ni500_inference --sampling_timesteps 1 2 4",
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/4_2_sf2/checkpoint_500.pt --dname 4_2_sf2_ni500_inference --sampling_timesteps 1 2",
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 1 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/2_1_sf2/checkpoint_500.pt --dname 2_1_sf2_ni500_inference --sampling_timesteps 1",

]

for cmd in commands:
    subprocess.call(cmd, shell=True)

# nohup python experiments/run_ni500_inference_shorter_distillation_experiments.py > logs/ni500_shorter_distillation_inference_experiments.log 2>&1 &
