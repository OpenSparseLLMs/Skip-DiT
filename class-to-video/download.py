from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
import os

# download skip-dit
for task in ['ucf101', 'taichi', 'skytimelapse', 'ffs']:
    hf_hub_download(repo_id="GuanjieChen/Skip-DiT", filename=f'{task}-skip.pt', local_dir=f"./pretrained/{task}-skip")

# download Latte
snapshot_download(repo_id="maxin-cn/Latte", local_dir="./pretrained/Latte", repo_type='model')
