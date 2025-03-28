from huggingface_hub import snapshot_download

model_repo = "maxin-cn/Latte-0"
local_dir = "/mnt/petrelfs/share_data/cgj/Latte-0"
local_dir = snapshot_download(repo_id=model_repo, local_dir=local_dir)

print(f"Model downloaded to: {local_dir}")