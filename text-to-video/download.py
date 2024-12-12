from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
import os

# download skip-dit
hf_hub_download(repo_id="GuanjieChen/Skip-DiT", filename=f'Latte-skip.pt', local_dir=f"./pretrained/Latte-skip")

# # download Latte
snapshot_download(repo_id="maxin-cn/Latte-0", local_dir="./pretrained/Latte-0", repo_type='model')
token_file_ls = ['added_tokens.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json']
os.makedirs('./pretrained/Latte-0/tokenizer', exist_ok=True)
for token_file in token_file_ls:
    hf_hub_download(repo_id="maxin-cn/Latte-1", filename=f'tokenizer/{token_file}', local_dir=f"./pretrained/Latte-0/")

