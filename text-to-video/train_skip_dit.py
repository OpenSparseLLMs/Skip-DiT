# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using PyTorch DDP.
"""

    
import torch
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import io
import os
import math
import argparse
import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from models import get_models
from datasets import get_dataset
from models.clip import TextEmbedder
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import (clip_grad_norm_, create_logger, update_ema, 
                requires_grad, cleanup, create_tensorboard, 
                write_tensorboard, setup_distributed,
                get_experiment_dir, text_preprocessing, get_grad_norm)
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
from torch.cuda.amp import autocast, GradScaler
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    setup_distributed()
    scaler = GradScaler()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    np.random.seed(seed)
    accumulation_steps = args.gradient_accumulation_steps
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}, accumulation_steps={accumulation_steps}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., Latte-XL/2 --> Latte-XL-2 (for naming folders)
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}"  # Create an experiment folder
        experiment_dir = get_experiment_dir(experiment_dir, args)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
    model = get_models(args)
    # Note that parameter initialization is done within the Latte constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    # use pretrained model?
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using model ckpt!')
            checkpoint = checkpoint["ema"]
        else:
            checkpoint = checkpoint['model']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                logger.info('Ignoring: {}'.format(k))
        logger.info('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(args.pretrained))
    linear_bias_l2 = torch.norm(model.skip_linears[13].linear.bias, p=2).item()
    linear_weight_l2 = torch.norm(model.skip_linears[13].linear.weight, p=2).item()
    
    norm_bias_l2 = torch.norm(model.skip_norms[13].bias, p=2).item()
    norm_weight_l2 = torch.norm(model.skip_norms[13].weight, p=2).item()
    logger.info(f'Linear: bias={linear_bias_l2}, weight={linear_weight_l2:.4f}; Norm: bias={norm_bias_l2:.4f}, weight={norm_weight_l2}')
        
    frozen_cont = 0
    free_cont = 0
    if args.use_compile:
        model = torch.compile(model)
    if args.freeze:
        logger.info(f'model is frozen except linear layer')
        for name, param in model.named_parameters():
            if "skip_linear" not in name and "skip_norm" not in name:
                param.requires_grad = False
                frozen_cont += 1
            else:
                free_cont += 1
        print(f'params: {free_cont=}, {frozen_cont=}')
    else:
        logger.info(f'model is not frozen')
    # set distributed training
    model = DDP(model.to(device), device_ids=[local_rank])
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()
    # Setup data:
    dataset = get_dataset(args)


    sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=rank,
    shuffle=True,
    seed=args.global_seed
    )
    
    def collate_t2v(batch):
        clean_example = [example for example in batch if example is not None]
        if len(clean_example) < len(batch):
            logger.error(f'read error data :{len(batch)-len(clean_example)}/{len(batch)}')
        batch = clean_example + [clean_example[-1]]*(len(batch)-len(clean_example))
        
        videos = torch.stack([example["video"] for example in batch])
        video_prompts = [item['video_prompt'] for item in batch]
        image_prompts = [item['image_prompts'] for item in batch]
        all_input_ids = []
        masks = []
        # input_ids = []
        for v,i in zip(video_prompts, image_prompts):
            all_prompts = [v]+i.split('====')
            tokening = tokenizer(all_prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            all_input_ids.append(tokening['input_ids'])
            masks.append(tokening['attention_mask'])
            
        all_input_ids = torch.stack(all_input_ids)
        masks = torch.stack(masks)
        return {'videos': videos, 'input_ids': all_input_ids, 'attention_mask': masks}
    
    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_t2v
    )
    
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")

    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # TODO, need to checkout
        # Get the most recent checkpoint
        dirs = os.listdir(os.path.join(experiment_dir, 'checkpoints'))
        dirs = [d for d in dirs if d.endswith("pt")]
        dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
        path = dirs[-1]
        logger.info(f"Resuming from checkpoint {path}")
        model.load_state(os.path.join(dirs, path))
        train_steps = int(path.split(".")[0])

        first_epoch = train_steps // num_update_steps_per_epoch
        resume_step = train_steps % num_update_steps_per_epoch

    if args.pretrained:
        train_steps = int(args.pretrained.split("/")[-1].split('.')[0])

    first_step = True
    
    with torch.autograd.detect_anomaly():
        for epoch in range(first_epoch, num_train_epochs):
            sampler.set_epoch(epoch)
            for step, video_data in enumerate(loader):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    continue
                x = video_data['videos'].to(device, non_blocking=True, dtype=torch.bfloat16)
                y = video_data["input_ids"].to(device, non_blocking=True)
                mask = video_data["attention_mask"].to(device, non_blocking=True)
                
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    encoder_hidden_states = []
                    for _y, _mask in zip(y, mask):
                        encoding = text_encoder(_y, _mask, return_dict=False)
                        encoder_hidden_states.append(encoding[0])
                    encoder_hidden_states = torch.stack(encoder_hidden_states)
                    model_kwargs = dict(encoder_hidden_states=encoder_hidden_states.to(dtype=torch.bfloat16), use_image_num=args.use_image_num)
                    b, _, _, _, _ = x.shape
                    x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    # x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()
                    x = rearrange(x, '(b f) c h w -> b c f h w', b=b).contiguous()
                    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                with autocast():
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                    loss = loss_dict["loss"].mean()/accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    try:
                        scaler.unscale_(opt) 
                        if train_steps < args.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
                            gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
                        else:
                            gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
                        if torch.isnan(gradient_norm) or gradient_norm > 0.02:
                            logger.info(f'skip because of nan')
                            scaler.update()
                            opt.zero_grad()
                        else:
                            scaler.step(opt)
                            scaler.update()
                            lr_scheduler.step()
                            update_ema(ema, model.module, decay=1-0.0001*(accumulation_steps+1)/2)
                            opt.zero_grad()
                    except Exception as e:
                        logger.info(f'fatal error: {e}')
                
                running_loss += loss.item() * accumulation_steps
                # Log loss values:
                lr = lr_scheduler.get_last_lr()[0]
                log_steps += 1
                train_steps += 1
                if (step + 1) % (accumulation_steps * 3)  == 0:
                    first_step = False
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    linear_bias_l2 = torch.norm(model.module.skip_linears[13].linear.bias, p=2).item()
                    linear_weight_l2 = torch.norm(model.module.skip_linears[13].linear.weight, p=2).item()
                    norm_bias_l2 = torch.norm(model.module.skip_norms[13].bias, p=2).item()
                    norm_weight_l2 = torch.norm(model.module.skip_norms[13].weight, p=2).item()
                    
                    logger.info(f"(step={train_steps:07d}/lr={lr:.7f}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Linear: bias={linear_bias_l2:.4f}, weight ={linear_weight_l2:.4f}, Norm: bias={norm_bias_l2}, weight={norm_weight_l2}, Train Steps/Sec: {steps_per_sec:.2f}")
                    
                    write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                    write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm, train_steps)
                    write_tensorboard(tb_writer, 'Learning Rate', lr, train_steps)
                    write_tensorboard(tb_writer, 'Linear Bias', linear_bias_l2, train_steps)
                    write_tensorboard(tb_writer, 'Linear Weight', linear_weight_l2, train_steps)
                    write_tensorboard(tb_writer, 'Norm Bias', norm_bias_l2, train_steps)
                    write_tensorboard(tb_writer, 'Norm Weight', norm_weight_l2, train_steps)
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save Latte checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            # "opt": opt.state_dict(),
                            # "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train Latte with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))
