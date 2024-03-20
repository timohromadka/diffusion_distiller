import subprocess

commands = [
    # MNIST INFERENCE EXPERIMENTS
    
    # divide by 4 - linear schedule
    # "python inference.py --module mnist_model --name mnist_exp --dname 1024_256_sf4_ni1000_l_inference --checkpoint_path ./checkpoints/mnist_exp/1024_256_sf4_ni1000_l/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 32 128 256",
    # "python inference.py --module mnist_model --name mnist_exp --dname 256_64_sf4_ni1000_l_inference --checkpoint_path ./checkpoints/mnist_exp/256_64_sf4_ni1000_l/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 16 64",
    # "python inference.py --module mnist_model --name mnist_exp --dname 64_16_ni1000_l_inference --checkpoint_path ./checkpoints/mnist_exp/64_16_ni1000_l/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 2 4 16",
    # "python inference.py --module mnist_model --name mnist_exp --dname 16_4_sf4_ni1000_l_inference --checkpoint_path ./checkpoints/mnist_exp/16_4_sf4_ni1000_l/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 2 4",
    # "python inference.py --module mnist_model --name mnist_exp --dname 4_1_sf4_ni1000_l_inference --checkpoint_path ./checkpoints/mnist_exp/4_1_sf4_ni1000_l/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1",

    # # early focus
    # "python inference.py --module mnist_model --name mnist_exp --dname 1024_512_sf2_ni1000_ef_inference --checkpoint_path ./checkpoints/mnist_exp/1024_512_sf2_ni1000_ef/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 32 256 512",
    # "python inference.py --module mnist_model --name mnist_exp --dname 512_256_sf2_ni1000_ef_inference --checkpoint_path ./checkpoints/mnist_exp/512_256_sf2_ni1000_ef/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 32 128 256",
    # "python inference.py --module mnist_model --name mnist_exp --dname 256_128_sf2_ni1000_ef_inference --checkpoint_path ./checkpoints/mnist_exp/256_128_sf2_ni1000_ef/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 16 32 128",
    # "python inference.py --module mnist_model --name mnist_exp --dname 128_32_sf4_ni1000_ef_inference --checkpoint_path ./checkpoints/mnist_exp/128_32_sf4_ni1000_ef/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 2 16 32",
    # "python inference.py --module mnist_model --name mnist_exp --dname 32_8_sf4_ni1000_ef_inference --checkpoint_path ./checkpoints/mnist_exp/32_8_sf4_ni1000_ef/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 2 4 8",
    # "python inference.py --module mnist_model --name mnist_exp --dname 8_1_sf8_ni1000_ef_inference --checkpoint_path ./checkpoints/mnist_exp/8_1_sf8_ni1000_ef/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1",
    
    # # late focus
    # "python inference.py --module mnist_model --name mnist_exp --dname 1024_128_sf8_ni1000_lf_inference --checkpoint_path ./checkpoints/mnist_exp/1024_128_sf8_ni1000_lf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 16 32 128",
    # "python inference.py --module mnist_model --name mnist_exp --dname 128_32_sf4_ni1000_lf_inference --checkpoint_path ./checkpoints/mnist_exp/128_32_sf4_ni1000_lf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 4 8 32",
    # "python inference.py --module mnist_model --name mnist_exp --dname 32_8_sf4_ni1000_lf_inference --checkpoint_path ./checkpoints/mnist_exp/32_8_sf4_ni1000_lf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 4 8",
    # "python inference.py --module mnist_model --name mnist_exp --dname 8_4_sf2_ni1000_lf_inference --checkpoint_path ./checkpoints/mnist_exp/8_4_sf2_ni1000_lf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 2 4",
    # "python inference.py --module mnist_model --name mnist_exp --dname 4_2_sf2_ni1000_lf_inference --checkpoint_path ./checkpoints/mnist_exp/4_2_sf2_ni1000_lf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 2",
    # "python inference.py --module mnist_model --name mnist_exp --dname 2_1_sf2_ni1000_lf_inference --checkpoint_path ./checkpoints/mnist_exp/2_1_sf2_ni1000_lf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1",

    # # no early focus, very late focus
    # "python inference.py --module mnist_model --name mnist_exp --dname 1024_32_sf32_ni1000_nefvlf_inference --checkpoint_path ./checkpoints/mnist_exp/1024_32_sf32_ni1000_nefvlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 4 8 32",
    # "python inference.py --module mnist_model --name mnist_exp --dname 32_16_sf2_ni1000_nefvlf_inference --checkpoint_path ./checkpoints/mnist_exp/32_16_sf2_ni1000_nefvlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 4 8 16",
    # "python inference.py --module mnist_model --name mnist_exp --dname 16_8_sf2_ni1000_nefvlf_inference --checkpoint_path ./checkpoints/mnist_exp/16_8_sf2_ni1000_nefvlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 2 4 8",
    # "python inference.py --module mnist_model --name mnist_exp --dname 8_4_sf2_ni1000_nefvlf_inference --checkpoint_path ./checkpoints/mnist_exp/8_4_sf2_ni1000_nefvlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 2 4",
    # "python inference.py --module mnist_model --name mnist_exp --dname 4_2_sf2_ni1000_nefvlf_inference --checkpoint_path ./checkpoints/mnist_exp/4_2_sf2_ni1000_nefvlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 2",
    # "python inference.py --module mnist_model --name mnist_exp --dname 2_1_sf2_ni1000_nefvlf_inference --checkpoint_path ./checkpoints/mnist_exp/2_1_sf2_ni1000_nefvlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1",
    
    # # very early focus, no late focus
    "python inference.py --module mnist_model --name mnist_exp --dname 1024_512_sf2_ni1000_vefnlf_inference --checkpoint_path ./checkpoints/mnist_exp/1024_512_sf2_ni1000_vefnlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 64 128 512",
    "python inference.py --module mnist_model --name mnist_exp --dname 512_256_sf2_ni1000_vefnlf_inference --checkpoint_path ./checkpoints/mnist_exp/512_256_sf2_ni1000_vefnlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 32 128 256",
    "python inference.py --module mnist_model --name mnist_exp --dname 256_128_sf2_ni1000_vefnlf_inference --checkpoint_path ./checkpoints/mnist_exp/256_128_sf2_ni1000_vefnlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 16 32 128",
    "python inference.py --module mnist_model --name mnist_exp --dname 128_64_sf2_ni1000_vefnlf_inference --checkpoint_path ./checkpoints/mnist_exp/128_64_sf2_ni1000_vefnlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 8 16 64",
    "python inference.py --module mnist_model --name mnist_exp --dname 64_32_sf2_ni1000_vefnlf_inference --checkpoint_path ./checkpoints/mnist_exp/64_32_sf2_ni1000_vefnlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 4 8 32",
    "python inference.py --module mnist_model --name mnist_exp --dname 32_1_sf32_ni1000_vefnlf_inference --checkpoint_path ./checkpoints/mnist_exp/32_1_sf32_ni1000_vefnlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1",

    
    # CIFAR10 INFERENCE EXPERIMENTS  
    # divide by 4 - linear schedule
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 1024_256_sf4_ni1000_l_inference --checkpoint_path ./checkpoints/cifar10_exp/1024_256_sf4_ni1000_l/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 32 128 256",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 256_64_sf4_ni1000_l_inference --checkpoint_path ./checkpoints/cifar10_exp/256_64_sf4_ni1000_l/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 16 64",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 64_16_ni1000_l_inference --checkpoint_path ./checkpoints/cifar10_exp/64_16_ni1000_l/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 2 4 16",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 16_4_sf4_ni1000_l_inference --checkpoint_path ./checkpoints/cifar10_exp/16_4_sf4_ni1000_l/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 2 4",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 4_1_sf4_ni1000_l_inference --checkpoint_path ./checkpoints/cifar10_exp/4_1_sf4_ni1000_l/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1",

    # # early focus
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 1024_512_sf2_ni1000_ef_inference --checkpoint_path ./checkpoints/cifar10_exp/1024_512_sf2_ni1000_ef/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 32 256 512",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 512_256_sf2_ni1000_ef_inference --checkpoint_path ./checkpoints/cifar10_exp/512_256_sf2_ni1000_ef/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 32 128 256",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 256_128_sf2_ni1000_ef_inference --checkpoint_path ./checkpoints/cifar10_exp/256_128_sf2_ni1000_ef/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 16 32 128",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 128_32_sf4_ni1000_ef_inference --checkpoint_path ./checkpoints/cifar10_exp/128_32_sf4_ni1000_ef/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 2 16 32",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 32_8_sf4_ni1000_ef_inference --checkpoint_path ./checkpoints/cifar10_exp/32_8_sf4_ni1000_ef/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 2 4 8",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 8_1_sf8_ni1000_ef_inference --checkpoint_path ./checkpoints/cifar10_exp/8_1_sf8_ni1000_ef/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1",
    
    # # late focus
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 1024_128_sf8_ni1000_lf_inference --checkpoint_path ./checkpoints/cifar10_exp/1024_128_sf8_ni1000_lf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 16 32 128",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 128_32_sf4_ni1000_lf_inference --checkpoint_path ./checkpoints/cifar10_exp/128_32_sf4_ni1000_lf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 4 8 32",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 32_8_sf4_ni1000_lf_inference --checkpoint_path ./checkpoints/cifar10_exp/32_8_sf4_ni1000_lf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 4 8",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 8_4_sf2_ni1000_lf_inference --checkpoint_path ./checkpoints/cifar10_exp/8_4_sf2_ni1000_lf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 2 4",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 4_2_sf2_ni1000_lf_inference --checkpoint_path ./checkpoints/cifar10_exp/4_2_sf2_ni1000_lf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 2",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 2_1_sf2_ni1000_lf_inference --checkpoint_path ./checkpoints/cifar10_exp/2_1_sf2_ni1000_lf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1",

    # # no early focus, very late focus
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 1024_32_sf32_ni1000_nefvlf_inference --checkpoint_path ./checkpoints/cifar10_exp/1024_32_sf32_ni1000_nefvlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 4 8 32",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 32_16_sf2_ni1000_nefvlf_inference --checkpoint_path ./checkpoints/cifar10_exp/32_16_sf2_ni1000_nefvlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 4 8 16",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 16_8_sf2_ni1000_nefvlf_inference --checkpoint_path ./checkpoints/cifar10_exp/16_8_sf2_ni1000_nefvlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 2 4 8",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 8_4_sf2_ni1000_nefvlf_inference --checkpoint_path ./checkpoints/cifar10_exp/8_4_sf2_ni1000_nefvlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 2 4",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 4_2_sf2_ni1000_nefvlf_inference --checkpoint_path ./checkpoints/cifar10_exp/4_2_sf2_ni1000_nefvlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 2",
    # "python inference.py --module cifar10_model --name cifar10_exp --dname 2_1_sf2_ni1000_nefvlf_inference --checkpoint_path ./checkpoints/cifar10_exp/2_1_sf2_ni1000_nefvlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1",
    
    # # very early focus, no late focus
    "python inference.py --module cifar10_model --name cifar10_exp --dname 1024_512_sf2_ni1000_vefnlf_inference --checkpoint_path ./checkpoints/cifar10_exp/1024_512_sf2_ni1000_vefnlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 64 128 512",
    "python inference.py --module cifar10_model --name cifar10_exp --dname 512_256_sf2_ni1000_vefnlf_inference --checkpoint_path ./checkpoints/cifar10_exp/512_256_sf2_ni1000_vefnlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 32 128 256",
    "python inference.py --module cifar10_model --name cifar10_exp --dname 256_128_sf2_ni1000_vefnlf_inference --checkpoint_path ./checkpoints/cifar10_exp/256_128_sf2_ni1000_vefnlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 16 32 128",
    "python inference.py --module cifar10_model --name cifar10_exp --dname 128_64_sf2_ni1000_vefnlf_inference --checkpoint_path ./checkpoints/cifar10_exp/128_64_sf2_ni1000_vefnlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 4 8 16 64",
    "python inference.py --module cifar10_model --name cifar10_exp --dname 64_32_sf2_ni1000_vefnlf_inference --checkpoint_path ./checkpoints/cifar10_exp/64_32_sf2_ni1000_vefnlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1 4 8 32",
    "python inference.py --module cifar10_model --name cifar10_exp --dname 32_1_sf32_ni1000_vefnlf_inference --checkpoint_path ./checkpoints/cifar10_exp/32_1_sf32_ni1000_vefnlf/checkpoint_1000.pt --batch_size 64 --num_workers 2 --num_fid_samples 1024 --num_images_to_log 16 --sampling_timesteps 1",
    
]

for cmd in commands:
    subprocess.call(cmd, shell=True)

# nohup python experiments/run_scheduled_inference_experiments.py > logs/scheduled_inference_experiments.log 2>&1 &
