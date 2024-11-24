import torch
import torchvision
from torchvision import transforms
class t2v_dataset(torch.utils.data.Dataset):
    """Load the Vimeo video files
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                configs,
                transform=None,
                temporal_sample=None):
        self.configs = configs
        # image related
        self.use_image_num = configs.use_image_num
        self.image_tranform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        # TODO 0. init your dataset here
        self.all_video = None
        
    def __getitem__(self, index):
        # TODO 1.preprocess video with shape: T C H W
        video = None
        video_prompt = 'TODO'
        total_frames = len(video)
        assert total_frames == 16
        video = self.transform(video) # T C H W
        
        images = []
        image_prompts = ['TODO']
        for i in range(self.use_image_num):      
            # TODO 2. read other videos
            image = video[0]/255.
            image = self.image_tranform(image).unsqueeze(0)
            images.append(image)
        images =  torch.cat(images, dim=0)
        assert len(images) == self.use_image_num
        video_cat = torch.cat([video, images], dim=0)
        image_prompts = '===='.join(image_prompts)
        return {'video': video_cat, 'video_prompt': str(video_prompt), 'image_prompts': image_prompts}
            
    
    def __len__(self):
        # TODO 3. show the way to measure the length of your dataset
        return len(self.all_video)



class T2V_video_with_img(torch.utils.data.Dataset):
    """Load the UCF101 video files
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                configs,
                transform=None,
                temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.video = t2v_dataset(configs=configs, transform=transform, temporal_sample=temporal_sample)
    
    def __getitem__(self, index):
        try:
            return self.video[index]
        except Exception as e:
            print(f'read exception data with error {e}')
            return None
        
    def __len__(self):
        return len(self.video)
        