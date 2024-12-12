# Class-to-video Instructions

![class-to-video visualizations](../visuals/case_c2v.jpg)

### About
This repository contains the official PyTorch implementation of class-to-video tasks in the paper: **[Accelerating Vision Diffusion Transformers with Skip Branches]()**. 

### Pretrained Model
| Model | Task | Training Data | Backbone | Size(G) | Skip-Cache |
|:--:|:--:|:--:|:--:|:--:|:--:|
| [ucf101-skip](https://huggingface.co/GuanjieChen/Skip-DiT/blob/main/ucf101-skip.pt) | class-to-video|UCF101|Latte|2.77|✅ |
| [taichi-skip](https://huggingface.co/GuanjieChen/Skip-DiT/blob/main/taichi-skip.pt) | class-to-video|Taichi-HD|Latte|2.77|✅ |
| [skytimelapse-skip](https://huggingface.co/GuanjieChen/Skip-DiT/blob/main/skylapse-skip.pt) | class-to-video|SkyTimelapse|Latte|2.77|✅ |
| [ffs-skip](https://huggingface.co/GuanjieChen/Skip-DiT/blob/main/ffs-skip.pt) | class-to-video|FaceForensics|Latte|2.77|✅ |

### Quick Start
To generate videos by yourself, you just need 3 steps

1. Prepare the environmentss
```shell
cd class-to-video
conda env create -f environment.yaml
conda activate latte
```
2. Download the checkpoints.
```shell
python pretrained/download_ckpts.py
```

3. Run the inference scripts under `sample/scripts`


### Training
To train the class-to-video models:
1. Download the datasets offered by [Xin Ma](https://huggingface.co/maxin-cn) in huggingface: [skytimelapse](Skip-DiT-open/maxin-cn/SkyTimelapse), [taichi](Skip-DiT-open/maxin-cn/Taichi-HD), [ffs](Skip-DiT-open/maxin-cn/FaceForensics). And you have to download [ucf101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) from the website.
2. Implement the TODOs in the training configs under `class-to-video/configs`
3. Run the training scripts under `class-to-image/train_scripts`

### Acknowledgement
Skip-DiT has been greatly inspired by the following amazing works and teams: [DeepCache](https://arxiv.org/abs/2312.00858), [Latte](https://github.com/Vchitect/Latte), [DiT](https://github.com/facebookresearch/DiT), and [HunYuan-DiT](https://github.com/Tencent/HunyuanDiT), we thank all the contributors for open-sourcing.

### License
The code and model weights are licensed under [LICENSE](./class-to-image/LICENSE).
