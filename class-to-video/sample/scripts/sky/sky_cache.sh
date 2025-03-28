
gpu_num=2
# to sample with ddp, relace `sample/sample.py` with `sample/sample_ddp.py`
torchrun --nnodes=1 --nproc_per_node=${gpu_num} --master_port=29525 sample/sample.py \
--config ./configs/sky/sky_sample_cache.yaml \
--ckpt path/to/ckpt \
--save_video_path "./videos" \
--sample_method "ddim" \
--num_sampling_steps 250 \
--deepcache 2
