import random

import torch
import os
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPImageProcessor
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):

    def __init__(self,data_dir,sample_size=[512,512],sample_n_stride=4,sample_n_frames=16):

        self.data_dir=data_dir # 数据集的根目录

        self.ref_img_dir=os.path.join(data_dir,"ref_img") # ref_img的目录

        self.ref_img_file_list=os.listdir(self.ref_img_dir) # 这里拿到了ref_img目录下的所有文件夹

        self.num_ref_img_file=len(self.ref_img_file_list)

        self.video_dir=os.path.join(data_dir,"videos") # video的目录

        self.video_file_list=os.listdir(self.video_dir) # 获取video目录下的所有视频文件

        self.sample_n_stride=sample_n_stride

        self.sample_n_frames=sample_n_frames

        self.length=len(self.video_file_list)  # 这里其实是视频总数

        self.clip_image_processor=CLIPImageProcessor()

        self.pixel_transforms=transforms.Compose(
            [transforms.Resize([sample_size[1],sample_size[0]]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),]
        )

    def __len__(self):
        return len(self.video_file_list)

    # 对读取到的文件列表按序号进行排序
    def sort_file_list(self,file_list):

        def extract_file_num(file_name):
            return int(file_name.split(".")[0].split("_")[-1])

        file_list=sorted(file_list,key=extract_file_num)

        return file_list

    def contrast_normalization(self,img,lower_bound=0,upper_bound=255):
        img=img.astype(np.float32)

        normalized_img=img*(upper_bound-lower_bound)/255+lower_bound

        normalized_img=normalized_img.astype(np.uint8)

        return normalized_img

    # 这个方法供__getitem__方法调用
    def get_batch_wo_pose(self,index):
        """"""
        video_name=self.video_file_list[index] # 获取视频名称，在这里我们并没有给视频名称排序

        video_file_path=os.path.join(self.video_dir,video_name) # 视频文件夹的路径

        video_file=os.listdir(os.path.join(video_file_path))# 获取视频的所有文件

        frame_list=self.sort_file_list([file for file in video_file if "frame" in file])  # 帧文件名
        mesh_list=self.sort_file_list([file for file in video_file if "mesh" in file])  # mesh图文件名
        keypoint_list=self.sort_file_list([file for file in video_file if "keypoint" in file]) # 关键点图文件名

        video_length=len(frame_list)

        tmp_sample_stride=self.sample_n_stride

        # 这里也就是在算所需要的最少的视频帧数
        clip_length=min(video_length,(self.sample_n_frames-1)*tmp_sample_stride+1)

        start_idx=random.randint(0,video_length-clip_length)

        # 拿到我要采样的帧的序号
        batch_idx=np.linspace(start_idx,start_idx+clip_length-1,self.sample_n_frames,dtype=int)

        # 下面我们来处理ref_img相关

        ref_img_file_idx=random.randint(0,self.num_ref_img_file-1) # 选择一个文件夹序号

        ref_img_file_name=self.ref_img_file_list[ref_img_file_idx]  # 这里是拿到了ref_img中放帧。mesh和关键点的文件夹名

        ref_img_file_path=os.path.join(self.ref_img_dir,ref_img_file_name)  # 这里是拿到了ref_img中放帧。mesh和关键点的文件夹路径

        ref_file=os.listdir(ref_img_file_path)

        ref_frame_list=self.sort_file_list([file for file in ref_file if "frame" in file])  # ref_img帧文件名
        ref_mesh_list=self.sort_file_list([file for file in ref_file if "mesh" in file])  # ref_img mesh图文件名
        ref_keypoint_list=self.sort_file_list([file for file in ref_file if "keypoint" in file]) # ref_img 关键点图文件名

        ref_img_length=len(ref_frame_list)
        ref_img_idx=random.randint(0,ref_img_length-1)

        ref_frame_path=os.path.join(ref_img_file_path,ref_frame_list[ref_img_idx])
        ref_mesh_path=os.path.join(ref_img_file_path,ref_mesh_list[ref_img_idx])
        ref_keypoint_path=os.path.join(ref_img_file_path,ref_keypoint_list[ref_img_idx])


        # 读取ref_img的frame数据
        ref_img=cv2.imread(ref_frame_path)
        ref_img=cv2.cvtColor(ref_img,cv2.COLOR_BGR2RGB)
        ref_img=self.contrast_normalization(ref_img)
        ref_img_pil=Image.fromarray(ref_img)  # 将图像转化为PIL图像对象

        clip_ref_image=self.clip_image_processor(images=ref_img_pil,return_tensors="pt").pixel_values

        pixel_values_ref_img=torch.from_numpy(ref_img).permute(2,0,1).contiguous()
        pixel_values_ref_img=pixel_values_ref_img/255.

        # 在这里我们似乎还要读取ref_img的pose和mesh信息
        ref_mesh=cv2.imread(ref_mesh_path)
        ref_mesh=cv2.cvtColor(ref_mesh,cv2.COLOR_BGR2RGB)
        ref_mesh=self.contrast_normalization(ref_mesh)
        pixel_values_ref_mesh=torch.from_numpy(ref_mesh).permute(2,0,1).contiguous()
        pixel_values_ref_mesh=pixel_values_ref_mesh/255.

        ref_keypoint=cv2.imread(ref_keypoint_path)
        ref_keypoint=cv2.cvtColor(ref_keypoint,cv2.COLOR_BGR2RGB)
        ref_keypoint=self.contrast_normalization(ref_keypoint)
        pixel_values_ref_keypoint=torch.from_numpy(ref_keypoint).permute(2,0,1).contiguous()
        pixel_values_ref_keypoint=pixel_values_ref_keypoint/255.

        # 下面我们就要读取帧数据了，读视频帧的原始数据
        images=[cv2.imread(os.path.join(video_file_path,frame_list[idx])) for idx in batch_idx]
        images=[cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB) for bgr_image in images]
        image_np=np.array([self.contrast_normalization(img) for img in images])

        pixel_values=torch.from_numpy(image_np).permute(0,3,1,2).contiguous() # 这个0312和上面201的思想是一样的
        pixel_values=pixel_values/255.0

        # 读视频帧的mesh数据
        images_mesh=[cv2.imread(os.path.join(video_file_path,mesh_list[idx])) for idx in batch_idx]
        images_mesh=[cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB) for bgr_image in images_mesh]
        image_mesh_np=np.array([self.contrast_normalization(img) for img in images_mesh])

        pixel_values_mesh=torch.from_numpy(image_mesh_np).permute(0,3,1,2).contiguous()
        pixel_values_mesh=pixel_values_mesh/255.0

        # 读视频帧的关键点数据
        images_keypoint=[cv2.imread(os.path.join(video_file_path,keypoint_list[idx])) for idx in batch_idx]
        images_keypoint=[cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB) for bgr_image in images_keypoint]
        image_keypoint_np=np.array([self.contrast_normalization(img) for img in images_keypoint])

        pixel_values_keypoint=torch.from_numpy(image_keypoint_np).permute(0,3,1,2).contiguous()
        pixel_values_keypoint=pixel_values_keypoint/255.0

        return pixel_values,pixel_values_mesh,pixel_values_keypoint,clip_ref_image,pixel_values_ref_img,pixel_values_ref_mesh,pixel_values_ref_keypoint # 还有两个要补充的东西，现在先返回那么多




    # 必须有的方法，为了和dataloader配合起来使用
    def __getitem__(self, item):
        """"""
        pixel_values,pixel_values_mesh,pixel_values_keypoint,clip_ref_image,pixel_values_ref_img,pixel_values_ref_mesh,pixel_values_ref_keypoint=self.get_batch_wo_pose(item)

        pixel_values=self.pixel_transforms(pixel_values)
        pixel_values_mesh=self.pixel_transforms(pixel_values_mesh)
        pixel_values_keypoint=self.pixel_transforms(pixel_values_keypoint)

        pixel_values_ref_img=pixel_values_ref_img.unsqueeze(0)
        pixel_values_ref_img=self.pixel_transforms(pixel_values_ref_img)
        pixel_values_ref_img=pixel_values_ref_img.squeeze(0)

        pixel_values_ref_mesh=pixel_values_ref_mesh.unsqueeze(0)
        pixel_values_ref_mesh=self.pixel_transforms(pixel_values_ref_mesh)
        pixel_values_ref_mesh=pixel_values_ref_mesh.squeeze(0)

        pixel_values_ref_keypoint=pixel_values_ref_keypoint.unsqueeze(0)
        pixel_values_ref_keypoint=self.pixel_transforms(pixel_values_ref_keypoint)
        pixel_values_ref_keypoint=pixel_values_ref_keypoint.squeeze(0)

        # 这里还有一个drop_image_embeds 没写
        sample=dict(
            pixel_values=pixel_values,
            pixel_values_mesh=pixel_values_mesh,
            pixel_values_keypoint=pixel_values_keypoint,
            clip_ref_image=clip_ref_image,
            pixel_values_ref_img=pixel_values_ref_img,
            pixel_values_ref_mesh=pixel_values_ref_mesh,
            pixel_values_ref_keypoint=pixel_values_ref_keypoint
        )

        return sample






# mydataset=MyDataset("/remote-home/share/yfsong/shipu/dataset_video_true")
# mydataset.get_batch_wo_pose(0)


