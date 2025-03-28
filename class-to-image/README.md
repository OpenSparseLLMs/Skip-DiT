# Class-to-image instructions

![class-to-image visualizations](../visuals/case_c2i.jpg)

### About
This repository contains the official PyTorch implementation of the class-to-image task in the paper: **[Accelerating Vision Diffusion Transformers with Skip Branches]()**. 


### Pretrained Model
| Model | Task | Training Data | Backbone | Size(G) | Skip-Cache |
|:--:|:--:|:--:|:--:|:--:|:--:|
| [DiT-XL/2-skip](https://huggingface.co/GuanjieChen/Skip-DiT/blob/main/Latte-skip.pt) | class-to-image |ImageNet|DiT-XL/2|11.40|✅ |

`DiT-XL/2-skip` takes only 38% of the training cost of [DiT-XL/2](https://github.com/facebookresearch/DiT) and outperform it siginificantly.



### Quick Start
To generate videos by yourself, you just need 3 steps
```shell
# 1. Prepare your conda environments
cd class-to-image ; conda env create -f environment.yaml ; conda activate DiT
# 2. Download checkpoints
mkdir ckpts ; wget -P ckpts https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt ; huggingface-cli download GuanjieChen/Skip-DiT/DiT-XL-2-skip.pt -d ./ckpts
# 3. Generate images
python sample.py --ckpt path/to/DiT-skip --model DiT-skip
# 4. (Optional) To accelerate generation with skip-cache, run following command
python sample.py --ckpt ckpt --model DiT-cache-2 # or DiT-cache-3, DiT-cache-4, ...
```

### Training
To train the DiT-XL/2-skip:
1. Download the [ImageNet](https://www.image-net.org/) dataset.
2. Implement the TODO in the train.py
3. run the script `run_train.sh`

### Acknowledgement
Skip-DiT has been greatly inspired by the following amazing works and teams: [DeepCache](https://arxiv.org/abs/2312.00858), [Latte](https://github.com/Vchitect/Latte), [DiT](https://github.com/facebookresearch/DiT), and [HunYuan-DiT](https://github.com/Tencent/HunyuanDiT), we thank all the contributors for open-sourcing.

### License
The code and model weights are licensed under [LICENSE](./class-to-image/LICENSE).
