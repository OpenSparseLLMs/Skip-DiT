gpu_num=1
srun --partition=MoE --mpi=pmi2 --gres=gpu:${gpu_num} -c 12 -n1 --ntasks-per-node=1 --job-name=cgj --kill-on-bad-exit=1 --quotatype=auto \
python sample/sample_t2v.py --config ./configs/sample_vbench/t2v-origin.yaml