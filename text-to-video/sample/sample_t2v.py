import os
import torch
import argparse
import torchvision


from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from omegaconf import OmegaConf
from transformers import T5EncoderModel, T5Tokenizer

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from pipeline_latte import LattePipeline
from models import get_models
from utils import save_video_grid
import imageio
from torchvision.utils import save_image

def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer_model = get_models(args).to(device, dtype=torch.float16)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        # breakpoint()
        if 'ema' in checkpoint.keys():
            print('using EMA checkpoint')
            checkpoint = checkpoint['ema']
        else:
            checkpoint = checkpoint['model']
        model_dict = transformer_model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                print('Ignoring: {}'.format(k))
        print('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        model_dict.update(pretrained_dict)
        transformer_model.load_state_dict(model_dict)
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
    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)
    print(f'save results in {args.save_img_path}')

    for num_prompt, prompt in enumerate(args.text_prompt):
        print('Processing the ({}) prompt'.format(prompt))
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
        if videos.shape[1] == 1:
            try:
                save_image(videos[0][0], args.save_img_path + prompt.replace(' ', '_') + '.png')
            except:
                save_image(videos[0][0], args.save_img_path + str(num_prompt)+ '.png')
                print('Error when saving {}'.format(prompt))
        else:
            try:
                imageio.mimwrite(args.save_img_path + prompt.replace(' ', '_') + '_%04d' % args.run_time + '.mp4', videos[0], fps=8, quality=9) # highest quality is 10, lowest is 0
            except:
                print('Error when saving {}'.format(prompt))
            
if __name__ == "__main__":
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--cache", type=str, default="")
    
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    if len(args.cache) > 0:
        omega_conf.model = omega_conf.model+'_'+args.cache
    main(omega_conf)


