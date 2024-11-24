gpu_num=1
# export SLURM_JOB_ID=3729308
srun --partition=MoE --mpi=pmi2 --gres=gpu:${gpu_num} -c 12 -n1 --ntasks-per-node=1 --job-name=cgj --kill-on-bad-exit=1 --quotatype=auto \
python /mnt/petrelfs/chenguanjie/cgj/HunyuanDiT/hydit/modules/models.py