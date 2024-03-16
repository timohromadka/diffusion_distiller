import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm
from moving_average import moving_average
from strategies import *
import wandb


@torch.no_grad()
def p_sample_loop(diffusion, noise, extra_args, device, eta=0, samples_to_capture=-1, need_tqdm=True, clip_value=3):
    mode = diffusion.net_.training
    diffusion.net_.eval()
    img = noise
    imgs = []
    iter_ = reversed(range(diffusion.num_timesteps))
    c_step = diffusion.num_timesteps/samples_to_capture
    next_capture = c_step
    if need_tqdm:
        iter_ = tqdm(iter_)
    for i in iter_:
        img = diffusion.p_sample(
            img,
            torch.full((img.shape[0],), i, dtype=torch.int64).to(device),
            extra_args,
            eta=eta,
            clip_value=clip_value
        )
        if diffusion.num_timesteps - i > next_capture:
            imgs.append(img)
            next_capture += c_step
    imgs.append(img)
    diffusion.net_.train(mode)
    return imgs


def make_none_args(img, label, device):
    return {}


def default_iter_callback(N, loss, last=False):
    None


def make_visualization_(diffusion, device, image_size, need_tqdm=False, eta=0, clip_value=1.2):
    extra_args = {}
    noise = torch.randn(image_size, device=device)
    imgs = p_sample_loop(diffusion, noise, extra_args, "cuda", samples_to_capture=5, need_tqdm=need_tqdm, eta=eta, clip_value=clip_value)
    images_ = []
    for images in imgs:
        images = images.split(1, dim=0)
        images = torch.cat(images, -1)
        images_.append(images)
    images_ = torch.cat(images_, 2)
    return images_


def make_visualization(diffusion, device, image_size, need_tqdm=False, eta=0, clip_value=1.2):
    images_ = make_visualization_(diffusion, device, image_size, need_tqdm=need_tqdm, eta=eta, clip_value=clip_value)
    images_ = images_[0].permute(1, 2, 0).cpu().numpy()
    images_ = (255 * (images_ + 1) / 2).clip(0, 255).astype(np.uint8)
    return images_


def make_iter_callback(diffusion, device, checkpoint_path, image_size, tensorboard, log_interval, ckpt_interval, need_tqdm=False):
    state = {
        "initialized": False,
        "last_log": None,
        "last_ckpt": None
    }

    def iter_callback(N, loss, last=False):
        from datetime import datetime
        t = datetime.now()
        if True:
            tensorboard.add_scalar("loss", loss, N)
        if not state["initialized"]:
            state["initialized"] = True
            state["last_log"] = t
            state["last_ckpt"] = t
            return
        if ((t - state["last_ckpt"]).total_seconds() / 60 > ckpt_interval) or last:
            torch.save({"G": diffusion.net_.state_dict(), "n_timesteps": diffusion.num_timesteps, "time_scale": diffusion.time_scale}, os.path.join(checkpoint_path, f"checkpoint.pt"))
            print("Saved.")
            state["last_ckpt"] = t
        if ((t - state["last_log"]).total_seconds() / 60 > log_interval) or last:
            images_ = make_visualization(diffusion, device, image_size, need_tqdm)
            images_ = cv2.cvtColor(images_, cv2.COLOR_BGR2RGB)
            tensorboard.add_image("visualization", images_, global_step=N, dataformats='HWC')
            tensorboard.flush()
            state["last_log"] = t

    return iter_callback


class InfinityDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, L):
        self.dataset = dataset
        self.L = L

    def __getitem__(self, item):
        idx = random.randint(0, len(self.dataset) - 1)
        r = self.dataset[idx]
        return r

    def __len__(self):
        return self.L


def make_condition(img, label, device):
    return {}


class DiffusionTrain:

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def train(
        self, 
        train_loader, 
        val_loader, 
        diffusion, 
        model_ema, 
        model_lr, 
        device, 
        num_steps, 
        checkpoints_dir,
        num_images,
        make_extra_args=make_none_args, 
        on_iter=default_iter_callback, 
        validate_after_n_steps=2000,
        num_val_steps=100,
        generate_images_after_n_steps=2000,
        patience=10):  # Add a patience argument
        scheduler = self.scheduler
        scheduler.init(diffusion, model_lr, num_steps)
        diffusion.net_.train()
        print(f"Training for {num_steps} steps...")

        best_val_loss = float('inf')
        patience_counter = 0  # Counter for the patience mechanism

        pbar = tqdm(total=num_steps)
        N = 0  # Global step counter
        L_tot = 0
        while N < num_steps and patience_counter < patience:
            for img, label in train_loader:
                if N >= num_steps or patience_counter >= patience:
                    break  # Exit the loop if the number of steps exceeds the limit or patience ran out

                scheduler.zero_grad()
                img = img.to(device)
                time = torch.randint(0, diffusion.num_timesteps, (img.shape[0],), device=device)
                extra_args = make_extra_args(img, label, device)
                loss = diffusion.p_loss(img, time, extra_args)
                L_tot += loss.item()
                wandb.log({"train_loss": loss.item()}, step=N)
                pbar.set_description(f"Step {N}/{num_steps}, Loss: {L_tot / (N+1)}")

                loss.backward()
                nn.utils.clip_grad_norm_(diffusion.net_.parameters(), 1)
                scheduler.step()
                moving_average(diffusion.net_, model_ema)
                
                # model and tensorboard saving/logging
                if on_iter: on_iter(N, loss.item())

                if N % validate_after_n_steps == 0 and N > 0 or N == num_steps - 1:
                    print(f"Step {N}: Starting validation.")
                    avg_val_loss = self.validate(
                        diffusion, val_loader, device, make_extra_args, num_val_steps=num_val_steps
                    )
                    print(f"Validation Loss: {avg_val_loss}")
                    wandb.log({"val_loss": avg_val_loss}, step=N)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                # Independent image generation logic
                if N % generate_images_after_n_steps == 0 and N > 0 or N == num_steps - 1:
                    print(f"Step {N}: Generating images.")
                    self.generate_images(
                        diffusion, device, val_loader, checkpoints_dir, f'num_steps_{N}', num_images=num_images
                    )

                N += 1
                pbar.update(1)

        if on_iter: on_iter(N, loss.item(), last=True)
        if patience_counter >= patience:
            print(f"Early stopping triggered after {N} steps.")
        print("Finished Training.")


    @staticmethod
    def validate(diffusion, val_loader, device, make_extra_args=make_none_args, num_val_steps=None):
        diffusion.net_.eval()
        total_loss = 0.0
        total_count = 0
        step_count = 0
        with torch.no_grad():
            for img, label in val_loader:
                if num_val_steps is not None and step_count >= num_val_steps:
                    break  # Stop validation if we've reached the specified number of validation steps
                img = img.to(device)
                time = torch.randint(0, diffusion.num_timesteps, (img.shape[0],), device=device)
                extra_args = make_extra_args(img, label, device)
                loss = diffusion.p_loss(img, time, extra_args)
                total_loss += loss.item() * img.size(0)
                total_count += img.size(0)
                step_count += 1
        avg_loss = total_loss / total_count if total_count > 0 else float('inf')
        diffusion.net_.train()
        return avg_loss
    
    def generate_images(self, diffusion, device, val_loader, checkpoint_path, image_prefix, num_images=16, eta=0.0, clip_value=1.2):
        diffusion.net_.eval()
        with torch.no_grad():
            noise = torch.randn((num_images, *next(iter(val_loader))[0].shape[1:]), device=device)
            for i in range(num_images):
                generated_img = make_visualization(diffusion, device, noise[i].unsqueeze(0).shape, need_tqdm=False, eta=eta, clip_value=clip_value)
                generated_img = (generated_img + 1) / 2  # Rescale to [0, 1]
                save_path = os.path.join(checkpoint_path, f"{image_prefix}_generated_image_{i}.png")
                save_image(generated_img, save_path)
        diffusion.net_.train()



class DiffusionDistillation:

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def train_student_debug(self, distill_train_loader, teacher_diffusion, student_diffusion, student_ema, student_lr, device, make_extra_args=make_none_args, on_iter=default_iter_callback):
        total_steps = len(distill_train_loader)
        scheduler = self.scheduler
        scheduler.init(student_diffusion, student_lr, total_steps)
        teacher_diffusion.net_.eval()
        student_diffusion.net_.train()
        print(f"Distillation...")
        pbar = tqdm(distill_train_loader)
        N = 0
        L_tot = 0

        for img, label in pbar:
            scheduler.zero_grad()
            img = img.to(device)
            time = 2 * torch.randint(0, student_diffusion.num_timesteps, (img.shape[0],), device=device)
            extra_args = make_extra_args(img, label, device)
            loss = teacher_diffusion.distill_loss(student_diffusion, img, time, extra_args)
            L = loss.item()
            L_tot += L
            N += 1
            wandb.log({"train_loss": loss.item()}, step=N)
            pbar.set_description(f"Loss: {L_tot / N}")
            loss.backward()
            scheduler.step()
            moving_average(student_diffusion.net_, student_ema)
            if scheduler.stop(N, total_steps):
                break
            on_iter(N, loss.item())
        on_iter(N, loss.item(), last=True)

    def train_student(self, distill_train_loader, teacher_diffusion, student_diffusion, student_ema, student_lr, device, make_extra_args=make_none_args, on_iter=default_iter_callback):
        scheduler = self.scheduler
        total_steps = len(distill_train_loader)
        scheduler.init(student_diffusion, student_lr, total_steps)
        teacher_diffusion.net_.eval()
        student_diffusion.net_.train()
        print(f"Distillation...")
        pbar = tqdm(distill_train_loader)
        N = 0
        L_tot = 0
        for img, label in pbar:
            scheduler.zero_grad()
            img = img.to(device)
            time = 2 * torch.randint(0, student_diffusion.num_timesteps, (img.shape[0],), device=device)
            extra_args = make_extra_args(img, label, device)
            loss = teacher_diffusion.distill_loss(student_diffusion, img, time, extra_args)
            L = loss.item()
            L_tot += L
            N += 1
            wandb.log({"train_loss": loss.item()}, step=N)
            pbar.set_description(f"Loss: {L_tot / N}")
            loss.backward()
            scheduler.step()
            moving_average(student_diffusion.net_, student_ema)
            if scheduler.stop(N, total_steps):
                break
            on_iter(N, loss.item())
        on_iter(N, loss.item(), last=True)
