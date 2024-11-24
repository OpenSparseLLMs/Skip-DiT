from torchvision import transforms
from datasets import video_transforms
from .t2v_joint_datasets import T2V_video_with_img
def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1     
    transform_t2v = transforms.Compose([
                video_transforms.ToTensorVideo(),
                video_transforms.CenterCropResizeVideo(args.image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    return T2V_video_with_img(args, transform=transform_t2v, temporal_sample=temporal_sample)
    