gpu_num=2
srun --partition=MoE --mpi=pmi2 --gres=gpu:${gpu_num} -n1 --ntasks-per-node=1 --job-name=cgj --kill-on-bad-exit=1 --quotatype=auto \
torchrun --nnodes=1 --nproc_per_node=${gpu_num} --master_port=29520  sample/sample_t2v_vimeo.py --config ./configs/sample_vimeo/t2v-skip-cache.yaml