import os
import torch
import argparse
import torchvision
import torch.distributed as dist

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler)
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from omegaconf import OmegaConf
from transformers import T5EncoderModel, T5Tokenizer

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from pipeline_latte import LattePipeline
from models import get_models
def main(args):
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)
    
    # setup ddp sample
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    if args.seed:
        seed = args.seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    # load model
    transformer_model = get_models(args).to(device, dtype=torch.float16)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        checkpoint = checkpoint['ema']
        model_dict = transformer_model.state_dict()
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                print('Ignoring: {}'.format(k))
        model_dict.update(pretrained_dict)
        transformer_model.load_state_dict(model_dict)
        if rank == 0:
            print('Successfully load model at {}!'.format(args.pretrained))
            print('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        
    if args.enable_vae_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == 'DDIM':
        scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, 
                                                subfolder="scheduler",
                                                beta_start=args.beta_start, 
                                                beta_end=args.beta_end, 
                                                beta_schedule=args.beta_schedule,
                                                variance_type=args.variance_type,
                                                clip_sample=False)
    elif args.sample_method == 'DDPM':
        scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, 
                                                subfolder="scheduler",
                                                beta_start=args.beta_start, 
                                                beta_end=args.beta_end, 
                                                beta_schedule=args.beta_schedule,
                                                variance_type=args.variance_type,
                                                clip_sample=False)


    videogen_pipeline = LattePipeline(vae=vae, 
                                text_encoder=text_encoder, 
                                tokenizer=tokenizer, 
                                scheduler=scheduler, 
                                transformer=transformer_model).to(device)

    if rank == 0:
        if not os.path.exists(args.save_img_path):
            os.makedirs(args.save_img_path)
        print(f'save results in {args.save_img_path}')

    import pandas as pd
    prompt_df = pd.read_csv(args.prompt_path)
    all_prompts = prompt_df['caption'].to_list()
    prompt_index = 1
    for _, prompt in enumerate(all_prompts):
        if prompt_index % dist.get_world_size() == int(rank):
            save_path = args.save_img_path + str(prompt_index).zfill(4) + '.mp4'
            if os.path.exists(save_path):
                prompt_index += 1
                continue
            print('{}: Processing the ({}) prompt'.format(prompt_index, prompt))
            videos = videogen_pipeline(prompt, 
                                    video_length=args.video_length, 
                                    height=args.image_size[0], 
                                    width=args.image_size[1], 
                                    num_inference_steps=args.num_sampling_steps,
                                    guidance_scale=args.guidance_scale,
                                    enable_temporal_attentions=args.enable_temporal_attentions,
                                    num_images_per_prompt=1,
                                    mask_feature=True,
                                    enable_vae_temporal_decoder=args.enable_vae_temporal_decoder
                                    ).video
            torchvision.io.write_video(save_path, videos[0], fps=8)
        prompt_index += 1
            
if __name__ == "__main__":
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--cache", type=str, default="")
    
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    if len(args.cache) > 0:
        omega_conf.model = omega_conf.model + '_' + args.cache
    elif omega_conf.cache is not None:
        omega_conf.model = omega_conf.model + '_' + omega_conf.cache
    main(omega_conf)
