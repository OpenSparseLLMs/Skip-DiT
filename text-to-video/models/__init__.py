import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from .latte_img import LatteIMG_models
from .latte_t2v import LatteT2V
from .latte_t2v_skip import LatteT2V_skip
from .latte_t2v_skip_cache import LatteT2V_skip_cache
from torch.optim.lr_scheduler import LambdaLR
def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(args):
    if 'LatteIMG' in args.model:
        return LatteIMG_models[args.model](
                input_size=args.latent_size,
                num_classes=args.num_classes,
                num_frames=args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras
            )
    elif 'LatteT2V' == args.model:
        print(f'loading model Latte,{args.model=}')
        return LatteT2V.from_pretrained(args.pretrained_model_path, subfolder="transformer", video_length=args.video_length)
    
    elif 'LatteT2V_skip' == args.model:
        print(f'loading model Latte_skip,{args.model=}')
        return LatteT2V_skip.from_pretrained(args.pretrained_model_path, subfolder="transformer", low_cpu_mem_usage=False, device_map=None, video_length=args.video_length,)
    elif 'LatteT2V_skip_cache' in args.model:
        cache_setting = args.model.replace('LatteT2V_skip_cache', '')
        if len(cache_setting)>1:
            # cache_setting = _Nxxx-xxx-xxx
            cache_gap = int(cache_setting.split('-')[0].replace('_N',''))
            cache_at_timesteps = [int(cache_setting.split('-')[1]), int(cache_setting.split('-')[2])]
        else:
            cache_gap = 2
            cache_at_timesteps = [700, 50]
        print(f'loading model: {args.model=} with cache setting gap: {cache_gap}, timestep: {cache_at_timesteps}')
        return LatteT2V_skip_cache.from_pretrained(args.pretrained_model_path, subfolder="transformer", low_cpu_mem_usage=False, device_map=None, video_length=args.video_length, cache_gap=cache_gap, cache_at_timesteps=cache_at_timesteps)
    else:
        raise f'{args.model} Model Not Supported!'
    