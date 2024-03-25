#!/usr/bin/env python
# coding: utf-8
import argparse
from datetime import datetime
import importlib
from v_diffusion import make_beta_schedule
from moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter

from train_utils import *
from vae_handler import VAEHandler

def make_argument_parser():
    parser = argparse.ArgumentParser()
    # training args
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--name", help="Experiment name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--dname", help="Distillation name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--checkpoint_to_continue", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--num_timesteps", help="Num diffusion steps.", type=int, default=1024)
    parser.add_argument("--num_iters", help="Num iterations.", type=int, default=100000)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=16)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=5e-5)
    parser.add_argument("--scheduler", help="Learning rate scheduler.", type=str, default="StrategyConstantLR")
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    # parser.add_argument("--log_interval", help="Log interval in minutes.", type=int, default=15)
    # parser.add_argument("--ckpt_interval", help="Checkpoints saving interval in minutes.", type=int, default=30)
    parser.add_argument("--ckpt_step_interval", help="Checkpoints saving interval in steps.", type=int, default=1000)
    parser.add_argument("--log_step_interval", help="Log interval in steps for image generation.", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=1)
    
    parser.add_argument("--use_vae", action='store_true')
    parser.add_argument("--vae_name", type=str, default=None, choices=['ft-mse', 'v1-4'])
    
    # wandb args
    parser.add_argument("--wandb_run_name", help="W&B run name for logging.", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--wandb_entity", help="W&B entity (user or team) under which the project is located.", type=str, default="timohrom")
    parser.add_argument("--project_name", help="W&B project name for logging.", type=str, default="diffusion_distiller_testing")
    parser.add_argument("--offline_mode", help="W&B project name for logging.", action='store_true')
    return parser


def train_model(args, make_model, make_dataset):
    if args.num_workers == -1:
        args.num_workers = args.batch_size * 2

    # print(args)
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    
    # if args.offline_mode:
    #     os.environ['WANDB_MODE'] = 'offline'
    # wandb.init(project=args.project_name, entity=args.wandb_entity, name=args.wandb_run_name, config=args)

    device = torch.device("cuda")

    if args.use_vae:
        if args.vae_name == 'ft-mse':
            model_path = "models/weights/vae-ft-mse-840000-ema-pruned.ckpt"
            vae_handler = VAEHandler(model_path=model_path, device=device)
        elif args.vae_name == 'v1-4':
            model_path = "CompVis/stable-diffusion-v1-4"
            vae_handler = VAEHandler(model_path=model_path, device=device)
        else:
            raise NotImplementedError(f"The VAE option <{args.vae_name}> is not yet implemented.")

    else:
        args.vae_name = None
        vae_handler = None

    train_dataset = InfinityDataset(make_dataset(vae_handler=vae_handler), args.num_iters*args.batch_size)
    test_dataset = InfinityDataset(make_dataset(train=False, vae_handler=vae_handler), args.num_iters*args.batch_size)

    len(train_dataset), len(test_dataset)

    img, anno = train_dataset[0]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    teacher_ema = make_model(vae_name=args.vae_name).to(device)

    checkpoints_dir = os.path.join("checkpoints", args.name, args.dname)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    def make_sheduler():
        M = importlib.import_module("train_utils")
        D = getattr(M, args.scheduler)
        return D()

    scheduler = make_sheduler()

    def make_diffusion(model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("v_diffusion")
        D = getattr(M, args.diffusion)
        return D(model, betas, time_scale=time_scale)

    teacher = make_model(vae_name=args.vae_name).to(device)
    teacher_ema = make_model(vae_name=args.vae_name).to(device)

    if args.checkpoint_to_continue != "":
        ckpt = torch.load(args.checkpoint_to_continue)
        teacher.load_state_dict(ckpt["G"])
        teacher_ema.load_state_dict(ckpt["G"])
        del ckpt
        print("Continue training...")
    else:
        print("Training new model...")
    init_ema_model(teacher, teacher_ema)

    tensorboard = SummaryWriter(os.path.join(checkpoints_dir, "tensorboard"))

    teacher_diffusion = make_diffusion(teacher, args.num_timesteps, 1, device)
    teacher_ema_diffusion = make_diffusion(teacher, args.num_timesteps, 1, device)

    image_size = teacher.image_size

    on_iter = make_iter_callback(teacher_ema_diffusion, device, checkpoints_dir, image_size, tensorboard, args.log_step_interval, args.ckpt_step_interval, False, vae_handler=vae_handler)
    diffusion_train = DiffusionTrain(scheduler)
    diffusion_train.train(train_loader, teacher_diffusion, teacher_ema, args.lr, device, make_extra_args=make_condition, on_iter=on_iter)
    print("Finished.")
    
    # At the end of train_model
    # wandb.finish()


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")
    make_dataset = getattr(M, "make_dataset")

    train_model(args, make_model, make_dataset)