
torchrun --nnodes=1 --nproc_per_node=8 train.py \
--model DiT-skip --data-path /path/to/datasets/imagenet_video \
--results-dir ./results-skip
