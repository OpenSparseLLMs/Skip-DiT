gpu_num=4
torchrun --nnodes=1 --nproc_per_node=${gpu_num} --master_port=29520 sample_ddp.py