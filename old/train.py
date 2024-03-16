#!/usr/bin/env python
# coding: utf-8
import argparse
from datetime import datetime
import importlib
import os
import wandb

from v_diffusion import make_beta_schedule
from moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter

from train_utils import *

def make_argument_parser():
    parser = argparse.ArgumentParser()
    # training args
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--name", help="Experiment name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--dname", help="Distillation name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--checkpoint_to_continue", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--num_timesteps", help="Num diffusion steps.", type=int, default=1024)
    parser.add_argument("--num_steps", help="Num iterations/steps.", type=int, default=100000)
    parser.add_argument("--num_val_steps", help="Num iterations/steps for the validation stage.", type=int, default=10)
    parser.add_argument("--use_on_iter", help="Whether or not to do tensorboard/model logging for each iteration.", action='store_true')
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=5e-5)
    parser.add_argument("--scheduler", help="Learning rate scheduler.", type=str, default="StrategyConstantLR")
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--log_interval", help="Log interval in minutes.", type=int, default=15)
    parser.add_argument("--ckpt_interval", help="Checkpoints saving interval in minutes.", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--generate_images_after_n_steps", help="Validate after every n steps.", type=int, default=2000)
    parser.add_argument("--validate_after_n_steps", help="Validate after every n steps.", type=int, default=9999999)
    parser.add_argument("--num_epochs", type=int, default=50, help='Specify the max number of epochs to train for.')
    parser.add_argument("--patience", type=int, default=1, help='Specify the patience for after how many unsucessfully raised validation losses to stop training. Minimum is 1.')
    parser.add_argument("--num_images", help="How many numbers to save after each validation check.", type=int, default=4)
    
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
    
    if args.offline_mode:
        os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project=args.project_name, entity=args.wandb_entity, name=args.wandb_run_name, config=args)

    device = torch.device("cuda")
    # train_dataset = test_dataset = InfinityDataset(make_dataset(), args.num_steps)
    train_dataset = InfinityDataset(make_dataset(train=True), args.num_steps)
    val_dataset = InfinityDataset(make_dataset(train=False), int(args.num_steps/5))

    len(train_dataset), len(val_dataset)

    img, anno = train_dataset[0]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    teacher_ema = make_model().to(device)

    checkpoints_dir = os.path.join("checkpoints", args.name, args.dname)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        
    image_checkpoints_dir = os.path.join(checkpoints_dir, 'validation_images')
    if not os.path.exists(image_checkpoints_dir):
        os.makedirs(image_checkpoints_dir)

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

    teacher = make_model().to(device)
    teacher_ema = make_model().to(device)

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

    if args.use_on_iter:
        on_iter = make_iter_callback(teacher_ema_diffusion, device, checkpoints_dir, image_size, tensorboard, args.log_interval, args.ckpt_interval, False)
    else:
        on_iter = None
    diffusion_train = DiffusionTrain(scheduler)
    # diffusion_train.train(train_loader, teacher_diffusion, teacher_ema, args.lr, device, make_extra_args=make_condition, on_iter=on_iter)
    diffusion_train.train(
        train_loader, 
        val_loader, 
        teacher_diffusion, 
        teacher_ema, 
        args.lr, 
        device, 
        args.num_steps,
        image_checkpoints_dir,
        args.num_images,
        make_extra_args=make_condition, 
        on_iter=on_iter, 
        validate_after_n_steps=args.validate_after_n_steps,
        num_val_steps=args.num_val_steps,
        generate_images_after_n_steps=args.generate_images_after_n_steps,
        patience=args.patience
    )
    print("Finished.")
    
    # At the end of train_model
    wandb.finish()


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")
    make_dataset = getattr(M, "make_dataset")

    train_model(args, make_model, make_dataset)