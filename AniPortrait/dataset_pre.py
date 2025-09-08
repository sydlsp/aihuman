import random

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torch.distributed as dist
import json
import torchvision.transforms as transforms
from src.utils.draw_util import FaceMeshVisualizer
from transformers import CLIPImageProcessor

import cv2

# 只在rank为0的进程上打印信息，这段代码似乎有逻辑问题
def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0):
        print("### " + s)


class FaceDatasetValid(Dataset):
    def __init__(self,json_path,extra_json_path=None,sample_size=[512,512],sample_stride=4,
                 sample_n_frames=16,is_image=False,sample_stride_aug=False):
        """上面的参数是对json文件中的data解析得到的"""
        zero_rank_print(f"loading annotations from {json_path}")

        # 这里拿到了可用的视频名称列表和json字典
        self.data_dic_name_list,self.data_dic=self.get_data(json_path,extra_json_path)

        # 获取视频名称列表的长度
        self.length=len(self.data_dic_name_list)
        zero_rank_print(f"data scale: {self.length}")

        self.sample_stride=sample_stride
        self.sample_n_frames=sample_n_frames

        self.sample_stride_aug=sample_stride_aug

        self.sample_size=sample_size
        self.resize=transforms.Resize((sample_size[0],sample_size[1]))

        self.pixel_transforms=transforms.Compose([
            transforms.Resize([sample_size[1], sample_size[0]]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.visualizer=FaceMeshVisualizer(forehead_edge=False)
        self.clip_image_processor=CLIPImageProcessor()
        self.is_image=is_image

    # 重要方法一：__len__方法，返回数据集中样本的数量，帮助dataloader知道迭代的次数，参数列表固定
    def __len__(self):
        return len(self.data_dic_name_list)

    # 重要方法二：__getitem__方法，返回数据集中索引为index的样本，参数列表固定
    def __getitem__(self, idx):
        ref_img,pixel_values_pose,tar_gt,pixel_values_ref_pose=self.get_batch_wo_pose(idx)

        sample=dict(
            pixel_values_pose=pixel_values_pose,
            ref_img=ref_img,
            tar_gt=tar_gt,
            pixel_values_ref_pose=pixel_values_ref_pose,
        )
        return sample

    def contrast_normalization(self, image, lower_bound=0, upper_bound=255):
        # convert input image to float32
        image = image.astype(np.float32)

        # normalize the image
        normalized_image = image  * (upper_bound - lower_bound) / 255 + lower_bound

        # convert to uint8
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image

    # 那下面的重点就是看一下get_batch_wo_pose函数
    def get_batch_wo_pose(self,index):

        # 获取视频名称
        video_name=self.data_dic_name_list[index]

        # data_dic是一个字典，里面存储了视频名称对应的数据
        video_clip_num=len(self.data_dic[video_name]['clip_data_list'])

        # 这里随机选择一个片段索引，获取该片段的帧路径列表和网格路径列表
        # 那这里其实我也要编制一个字典，来存放帧数据和条件
        source_anchor=random.sample(range(video_clip_num),1)[0]
        source_image_path_list=self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        source_mesh2d_path_list=self.data_dic[video_name]['clip_data_list'][source_anchor]['mesh2d_path_list']

        video_length=len(source_image_path_list)

        if self.sample_stride_aug:
            tmp_sample_stride=self.sample_stride_aug if random.random() > 0.5 else 4

        else:
            tmp_sample_stride=self.sample_stride

        if not self.is_image:
            # sample_n_frames是采样帧数，tmp_sample_stride是帧间步长
            clip_length=min(video_length,(self.sample_n_frames-1)*tmp_sample_stride+1)
            start_idx=random.randint(0,video_length-clip_length)
            # 生成一个从start_idx开始，start_idx+clip_length-1结束，一共包含self.sample_n_frames个元素的等差数列
            batch_index=np.linspace(start_idx,start_idx+clip_length-1,self.sample_n_frames,dtype=int)
        else:
            batch_index=[random.randint(0,video_length-1)]

        # 看一下，这里ref_img还是要从ref_image列表获取
        ref_img_idx=random.randint(0,video_length-1)

        ref_img=cv2.imread(source_image_path_list[ref_img_idx])
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = self.contrast_normalization(ref_img)

        ref_mesh2d_clip=np.load(source_mesh2d_path_list[ref_img_idx]).astype(float)
        ref_pose_image=self.visualizer.draw_landmarks(self.sample_size,ref_mesh2d_clip,normed=True)

        # 这里是读帧数据
        images = [cv2.imread(source_image_path_list[idx]) for idx in batch_index]
        images = [cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) for bgr_image in images]
        image_np = np.array([self.contrast_normalization(img) for img in images])

        pixel_values=torch.from_numpy(image_np).permute(0,3,1,2).contiguous()
        pixel_values=pixel_values/255.

        mesh2d_clip=np.array([np.load(source_mesh2d_path_list[idx]).astype(float) for idx in batch_index])

        pixel_values_pose=[]
        for frame_id in range(mesh2d_clip.shape[0]):
            normed_mesh2d=mesh2d_clip[frame_id]

            pose_image=self.visualizer.draw_landmarks(self.sample_size,normed_mesh2d,normed=True)
            pixel_values_pose.append(pose_image)

        if self.is_image:
            pixel_values=pixel_values[0]
            pixel_values_pose=pixel_values_pose[0]
            image_np=image_np[0]

        return ref_img,pixel_values_pose













    def get_data(self,json_name,extra_json_name,augment_num=1):
        zero_rank_print(f"start loading data: {json_name}")
        with open(json_name, 'r') as f:
            data_dic=json.load(f)

        data_dic_name_list=[]
        # 将data_dic字典中的每个视频名称重复augment_num次，存入列表中，为了数据扩充
        for augment_index in range(augment_num):
            for video_name in data_dic.keys():
                data_dic_name_list.append(video_name)

        invalid_video_name_list=[]

        # 移除无效的视频名称
        for video_name in data_dic_name_list:
            video_clip_num=len(data_dic[video_name]['clip_data_list'])
            if video_clip_num<1:
                invalid_video_name_list.append(video_name)
        for name in invalid_video_name_list:
            data_dic_name_list.remove(name)

        # 对于extra_json 做基本类似的操作
        if extra_json_name is not None:
            zero_rank_print(f"start loading data: {extra_json_name}")
            with open(extra_json_name,'r') as f:
                extra_data_dic = json.load(f)
            data_dic.update(extra_data_dic)
            for augment_index in range(3*augment_num):
                for video_name in extra_data_dic.keys():
                    data_dic_name_list.append(video_name)

        random.shuffle(data_dic_name_list)
        zero_rank_print("finish loading")

        # 这里返回的是经过数据扩充和无效数据过滤后的视频名称，以及json字典
        return data_dic_name_list,data_dic

