import subprocess

commands = [
    # MNIST INFERENCE EXPERIMENTS
    
    # teacher sf1
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/teacher/checkpoint_30000.pt --dname teacher_ni32_inference --sampling_timesteps 4 64 128 512 1024",
    
    # students sf2
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/1024_512_sf2_ni32/checkpoint_32.pt --dname 1024_512_sf2_ni32_inference --sampling_timesteps 512",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/512_256_sf2_ni32/checkpoint_32.pt --dname 512_256_sf2_ni32_inference --sampling_timesteps 256",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/256_128_sf2_ni32/checkpoint_32.pt --dname 256_128_sf2_ni32_inference --sampling_timesteps 128",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/128_64_sf2_ni32/checkpoint_32.pt --dname 128_64_sf2_ni32_inference --sampling_timesteps 64",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/64_32_sf2_ni32/checkpoint_32.pt --dname 64_32_sf2_ni32_inference --sampling_timesteps 32",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/32_16_sf2_ni32/checkpoint_32.pt --dname 32_16_sf2_ni32_inference --sampling_timesteps 16",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/16_8_sf2_ni32/checkpoint_32.pt --dname 16_8_sf2_ni32_inference --sampling_timesteps 8",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/8_4_sf2_ni32/checkpoint_32.pt --dname 8_4_sf2_ni32_inference --sampling_timesteps 4",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/4_2_sf2_ni32/checkpoint_32.pt --dname 4_2_sf2_ni32_inference --sampling_timesteps 2",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/2_1_sf2_ni32/checkpoint_32.pt --dname 2_1_sf2_ni32_inference --sampling_timesteps 1",

    # # students sf4
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/1024_256_sf4_ni32/checkpoint_32.pt --dname 1024_256_sf4_ni32_inference --sampling_timesteps 256",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/256_64_sf4_ni32/checkpoint_32.pt --dname 256_64_sf4_ni32_inference --sampling_timesteps 64",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/64_16_sf4_ni32/checkpoint_32.pt --dname 64_16_sf4_ni32_inference --sampling_timesteps 16",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/16_4_sf4_ni32/checkpoint_32.pt --dname 16_4_sf4_ni32_inference --sampling_timesteps 4",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/4_1_sf4_ni32/checkpoint_32.pt --dname 4_1_sf4_ni32_inference --sampling_timesteps 1",
   
    # # students sf8
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/1024_128_sf8_ni32/checkpoint_32.pt --dname 1024_128_sf8_ni32_inference --sampling_timesteps 128",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/128_16_sf8_ni32/checkpoint_32.pt --dname 128_16_sf8_ni32_inference --sampling_timesteps 16",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/16_2_sf8_ni32/checkpoint_32.pt --dname 16_2_sf8_ni32_inference --sampling_timesteps 2",

    # # students sf16
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/1024_64_sf16_ni32/checkpoint_32.pt --dname 1024_64_sf16_ni32_inference --sampling_timesteps 64",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/64_4_sf16_ni32/checkpoint_32.pt --dname 64_4_sf16_ni32_inference --sampling_timesteps 4",
    
    # # students sf32
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/1024_32_sf32_ni32/checkpoint_32.pt --dname 1024_32_sf32_ni32_inference --sampling_timesteps 32",
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/32_1_sf32_ni32/checkpoint_32.pt --dname 32_1_sf32_ni32_inference --sampling_timesteps 1",
    
    # # students sf256
    # "python inference.py --module mnist_model --name mnist_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/mnist_exp/1024_4_sf256_ni32/checkpoint_32.pt --dname 1024_4_sf256_ni32_inference --sampling_timesteps 4",

    # CIFAR10 INFERENCE EXPERIMENTS
    
    # teacher sf1
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/teacher/checkpoint_30000.pt --dname teacher_ni32_inference --sampling_timesteps 4 64 128 512 1024",
    
    # students sf2
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/1024_512_sf2_ni32/checkpoint_32.pt --dname 1024_512_sf2_ni32_inference --sampling_timesteps 512",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/512_256_sf2_ni32/checkpoint_32.pt --dname 512_256_sf2_ni32_inference --sampling_timesteps 256",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/256_128_sf2_ni32/checkpoint_32.pt --dname 256_128_sf2_ni32_inference --sampling_timesteps 128",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/128_64_sf2_ni32/checkpoint_32.pt --dname 128_64_sf2_ni32_inference --sampling_timesteps 64",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/64_32_sf2_ni32/checkpoint_32.pt --dname 64_32_sf2_ni32_inference --sampling_timesteps 32",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/32_16_sf2_ni32/checkpoint_32.pt --dname 32_16_sf2_ni32_inference --sampling_timesteps 16",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/16_8_sf2_ni32/checkpoint_32.pt --dname 16_8_sf2_ni32_inference --sampling_timesteps 8",
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/8_4_sf2_ni32/checkpoint_32.pt --dname 8_4_sf2_ni32_inference --sampling_timesteps 4",
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/4_2_sf2_ni32/checkpoint_32.pt --dname 4_2_sf2_ni32_inference --sampling_timesteps 2",
    "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/2_1_sf2_ni32/checkpoint_32.pt --dname 2_1_sf2_ni32_inference --sampling_timesteps 1",

    # # students sf4
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/1024_256_sf4_ni32/checkpoint_32.pt --dname 1024_256_sf4_ni32_inference --sampling_timesteps 256",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/256_64_sf4_ni32/checkpoint_32.pt --dname 256_64_sf4_ni32_inference --sampling_timesteps 64",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/64_16_sf4_ni32/checkpoint_32.pt --dname 64_16_sf4_ni32_inference --sampling_timesteps 16",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/16_4_sf4_ni32/checkpoint_32.pt --dname 16_4_sf4_ni32_inference --sampling_timesteps 4",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/4_1_sf4_ni32/checkpoint_32.pt --dname 4_1_sf4_ni32_inference --sampling_timesteps 1",
   
    # # students sf8
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/1024_128_sf8_ni32/checkpoint_32.pt --dname 1024_128_sf8_ni32_inference --sampling_timesteps 128",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/128_16_sf8_ni32/checkpoint_32.pt --dname 128_16_sf8_ni32_inference --sampling_timesteps 16",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/16_2_sf8_ni32/checkpoint_32.pt --dname 16_2_sf8_ni32_inference --sampling_timesteps 2",

    # # students sf16
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/1024_64_sf16_ni32/checkpoint_32.pt --dname 1024_64_sf16_ni32_inference --sampling_timesteps 64",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/64_4_sf16_ni32/checkpoint_32.pt --dname 64_4_sf16_ni32_inference --sampling_timesteps 4",
    
    # # students sf32
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/1024_32_sf32_ni32/checkpoint_32.pt --dname 1024_32_sf32_ni32_inference --sampling_timesteps 32",
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/32_1_sf32_ni32/checkpoint_32.pt --dname 32_1_sf32_ni32_inference --sampling_timesteps 1",
    
    # # students sf256
    # "python inference.py --module cifar10_model --name cifar10_exp --batch_size 64 --num_workers 4 --num_fid_samples 1024 --num_images_to_log 16 --checkpoint_path ./checkpoints/cifar10_exp/1024_4_sf256_ni32/checkpoint_32.pt --dname 1024_4_sf256_ni32_inference --sampling_timesteps 4",
    
]

for cmd in commands:
    subprocess.call(cmd, shell=True)

# nohup python experiments/run_inference_experiments_ni32_full.py > logs/inference_experiments_ni32_full.log 2>&1 &
