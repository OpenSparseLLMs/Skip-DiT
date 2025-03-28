import os
import json
import time
import tqdm
from pathlib import Path
from loguru import logger

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from hydit.config import get_args
from hydit.inference import End2End


def setup_ddp():
    """Initialize the distributed environment."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Cleanup the distributed environment."""
    dist.destroy_process_group()


def inferencer(local_rank):
    args = get_args()
    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load models
    gen = End2End(args, models_root_path)

    # Wrap in DDP
    gen = DDP(gen, device_ids=[local_rank], output_device=local_rank)

    # Try to enhance prompt (not used here)
    enhancer = None

    return args, gen, enhancer


def main():
    local_rank = setup_ddp()

    try:
        args, gen, enhancer = inferencer(local_rank)

        # Broadcast and load captions on all processes
        caption_path = args.caption_path
        if local_rank == 0:
            with open(caption_path, "r") as file:
                captions = json.load(file)
        else:
            captions = None
        captions = dist.broadcast_object_list([captions])[0]

        # Distributed inference
        logger.info(f"[Rank {local_rank}] Generating images...")
        height, width = args.image_size
        start_time = time.time()
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Each rank processes its subset of captions
        caption_id = 0
        total_captions = len(captions)
        pbar = tqdm.tqdm(total=total_captions, position=local_rank)

        for caption_id, caption in enumerate(captions):
            # Ensure only the process responsible for the index processes it
            if caption_id % dist.get_world_size() != local_rank:
                continue

            save_path = os.path.join(save_dir, f'{caption_id}.png')
            if os.path.exists(save_path):
                continue

            results = gen.predict(caption,
                                height=height,
                                width=width,
                                seed=args.seed,
                                enhanced_prompt=None,
                                negative_prompt=args.negative,
                                infer_steps=args.infer_steps,
                                guidance_scale=args.cfg_scale,
                                batch_size=args.batch_size,
                                src_size_cond=args.size_cond,
                                use_style_cond=args.use_style_cond,
                                )
            image = results['images'][0]

            # Save image
            image.save(save_path)
            logger.info(f"[Rank {local_rank}] Saved to {save_path}")

            pbar.update(1)
            pbar.set_description(f"[Rank {local_rank}] {caption_id}/{total_captions}")

        if local_rank == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"Total time: {elapsed_time:.2f}s")

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()