import random

import torch
import os
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPImageProcessor
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

"""
和datase_write_1的区别就是加了个本来有的sample_stride_aug参数，这里只是怕出错所以复制了一个版本
这是专门用于train_stage_2的,train_stage_1也能用
"""
class MyDataset(Dataset):
    def __init__(self,data_dir,sample_size=[512,512],sample_stride=4,sample_n_frames=16,is_image=False,sample_stride_aug=4):

        self.data_dir=data_dir

        self.video_dir=os.path.join(data_dir,'videos') # 这里拿到训练文件夹的名称

        self.video_file=os.listdir(self.video_dir)  # 拿到训练文件夹内每个文件夹的名称

        self.sample_n_stride=sample_stride

        self.sample_n_frames=sample_n_frames

        self.sample_stride_aug=sample_stride_aug

        self.length=len(self.video_file)  # 拿到训练的视频总数

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transforms = transforms.Compose(
            [transforms.Resize([sample_size[1], sample_size[0]]),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # 这里改了一下，不要均值了
             ]
        )
        self.is_image = is_image

        self.ref_img_count=5  # 我们取前5帧作为ref

    def __len__(self):
        return len(self.video_file)

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

        video_name=self.video_file[index]  # 这里我们还是没有对文件夹进行排序，这里应该是不需要排序的
        video_file_path=os.path.join(self.video_dir,video_name)
        video_file=os.listdir(video_file_path)  # 拿到文件夹内的frames mesh以及kp文件的所有图片

        frame_list=self.sort_file_list([file for file in video_file if "frame" in file])  # 拿到所有的frame图片
        mesh_list=self.sort_file_list([file for file in video_file if "mesh" in file]) # 所有的mesh图片
        keypoint_list=self.sort_file_list([file for file in video_file if "keypoint" in file]) # 所有的keypoint图片

        # 在我们的数据中保证了frame都是大于5帧的
        video_total_length=len(frame_list)

        video_length=video_total_length-self.ref_img_count  # 我们取前5帧作为ref

        if self.sample_stride_aug:
            tmp_sample_stride = self.sample_n_stride if random.random() > 0.5 else 4
        else:
            tmp_sample_stride = self.sample_n_stride


        if not self.is_image:
            # 这里也就是在算所需要的最少的视频帧数，取一个视频中的帧数和所需要帧数的最小值
            clip_length = min(video_length, (self.sample_n_frames - 1) * tmp_sample_stride + 1)

            # 计算起始帧的序号
            start_idx=random.randint(self.ref_img_count,self.ref_img_count+video_length-clip_length)

            # 拿到我要采样帧的序号
            batch_idx=np.linspace(start_idx,start_idx+clip_length-1,self.sample_n_frames,dtype=int)
        else:
            batch_idx = [random.randint(self.ref_img_count, self.ref_img_count+video_length - 1)]

        # 下面我们开始处理ref_img相关
        ref_img_file_idx=random.randint(0,self.ref_img_count-1)

        ref_frame_path=os.path.join(video_file_path,frame_list[ref_img_file_idx])
        ref_mesh_path=os.path.join(video_file_path,mesh_list[ref_img_file_idx])
        ref_keypoint_path=os.path.join(video_file_path,keypoint_list[ref_img_file_idx])

        # 读取ref_img的frame数据
        ref_img = cv2.imread(ref_frame_path)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = self.contrast_normalization(ref_img)  # 为什么这里一开始没加呢，看一下加不加有什么作用吧
        ref_img_pil = Image.fromarray(ref_img)  # 将图像转化为PIL图像对象，这个PIL对象是下面clip要用的

        clip_ref_image = self.clip_image_processor(images=ref_img_pil, return_tensors="pt").pixel_values

        pixel_values_ref_img = torch.from_numpy(ref_img).permute(2, 0, 1).contiguous()
        pixel_values_ref_img = pixel_values_ref_img / 255.

        #ref_img可视化一下
        # pixel_values_ref_img_array=pixel_values_ref_img.permute(1,2,0).cpu().numpy()*255
        # pixel_values_ref_img_array = pixel_values_ref_img_array.astype(np.uint8)
        # pixel_values_ref_img_image=Image.fromarray(pixel_values_ref_img_array)
        # pixel_values_ref_img_image.save(os.path.join("/remote-home/yfsong/shipu/mycode/newdataset_see",'pixel_values_ref_img.png'))

        # 继续读取ref_img的mesh数据，只是这里不需要转化为PIL的那一步
        ref_mesh = cv2.imread(ref_mesh_path)
        ref_mesh = cv2.cvtColor(ref_mesh, cv2.COLOR_BGR2RGB)
        ref_mesh = self.contrast_normalization(ref_mesh)
        pixel_values_ref_mesh = torch.from_numpy(ref_mesh).permute(2, 0, 1).contiguous()
        pixel_values_ref_mesh = pixel_values_ref_mesh / 255.

        ref_keypoint = cv2.imread(ref_keypoint_path)
        ref_keypoint = cv2.cvtColor(ref_keypoint, cv2.COLOR_BGR2RGB)
        ref_keypoint = self.contrast_normalization(ref_keypoint)
        pixel_values_ref_keypoint = torch.from_numpy(ref_keypoint).permute(2, 0, 1).contiguous()
        pixel_values_ref_keypoint = pixel_values_ref_keypoint / 255.


        # 下面我们开始处理我们要用到的视频数据
        images=[cv2.imread(os.path.join(video_file_path,frame_list[idx])) for idx in batch_idx]
        images = [cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) for bgr_image in images]
        image_np = np.array([self.contrast_normalization(img) for img in images])

        pixel_values = torch.from_numpy(image_np).permute(0, 3, 1, 2).contiguous()  # 这个0312和上面201的思想是一样的
        pixel_values = pixel_values / 255.0

        # 视频的mesh数据和kp数据
        images_mesh=[cv2.imread(os.path.join(video_file_path,mesh_list[idx])) for idx in batch_idx]
        images_mesh = [cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) for bgr_image in images_mesh]
        image_mesh_np = np.array([self.contrast_normalization(img) for img in images_mesh])

        pixel_values_mesh = torch.from_numpy(image_mesh_np).permute(0, 3, 1, 2).contiguous()
        pixel_values_mesh = pixel_values_mesh / 255.0

        # 读视频帧的关键点数据
        images_keypoint = [cv2.imread(os.path.join(video_file_path, keypoint_list[idx])) for idx in batch_idx]
        images_keypoint = [cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) for bgr_image in images_keypoint]
        image_keypoint_np = np.array([self.contrast_normalization(img) for img in images_keypoint])

        pixel_values_keypoint = torch.from_numpy(image_keypoint_np).permute(0, 3, 1, 2).contiguous()
        pixel_values_keypoint = pixel_values_keypoint / 255.0

        if self.is_image:
            pixel_values=pixel_values[0]
            pixel_values_mesh=pixel_values_mesh[0]
            pixel_values_keypoint=pixel_values_keypoint[0]

        # 对图像可视化一番
        # pixel_values_array=pixel_values.permute(1,2,0).cpu().numpy()*255
        # pixel_values_array = pixel_values_array.astype(np.uint8)
        # pixel_values_image=Image.fromarray(pixel_values_array)
        # pixel_values_image.save(os.path.join("/remote-home/yfsong/shipu/mycode/newdataset_see",'pixel_values.png'))
        #
        # pixel_values_mesh_array=pixel_values_mesh.permute(1,2,0).cpu().numpy()*255
        # pixel_values_mesh_array = pixel_values_mesh_array.astype(np.uint8)
        # pixel_values_mesh_image=Image.fromarray(pixel_values_mesh_array)
        # pixel_values_mesh_image.save(os.path.join("/remote-home/yfsong/shipu/mycode/newdataset_see",'pixel_values_mesh.png'))
        #
        # pixel_values_keypoint_array=pixel_values_keypoint.permute(1,2,0).cpu().numpy()*255
        # pixel_values_keypoint_array = pixel_values_keypoint_array.astype(np.uint8)
        # pixel_values_keypoint_image=Image.fromarray(pixel_values_keypoint_array)
        # pixel_values_keypoint_image.save(os.path.join("/remote-home/yfsong/shipu/mycode/newdataset_see",'pixel_values_keypoint.png'))

        return pixel_values,pixel_values_mesh,pixel_values_keypoint,clip_ref_image,pixel_values_ref_img,pixel_values_ref_mesh,pixel_values_ref_keypoint


    # 必须有的方法，为了和dataloader配合起来使用

    def __getitem__(self, item):
        """"""
        pixel_values, pixel_values_mesh, pixel_values_keypoint, clip_ref_image, pixel_values_ref_img, pixel_values_ref_mesh, pixel_values_ref_keypoint = self.get_batch_wo_pose(
            item)

        pixel_values = self.pixel_transforms(pixel_values)
        pixel_values_mesh = self.pixel_transforms(pixel_values_mesh)
        pixel_values_keypoint = self.pixel_transforms(pixel_values_keypoint)

        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
        pixel_values_ref_img = self.pixel_transforms(pixel_values_ref_img)
        pixel_values_ref_img = pixel_values_ref_img.squeeze(0)

        pixel_values_ref_mesh = pixel_values_ref_mesh.unsqueeze(0)
        pixel_values_ref_mesh = self.pixel_transforms(pixel_values_ref_mesh)
        pixel_values_ref_mesh = pixel_values_ref_mesh.squeeze(0)

        pixel_values_ref_keypoint = pixel_values_ref_keypoint.unsqueeze(0)
        pixel_values_ref_keypoint = self.pixel_transforms(pixel_values_ref_keypoint)
        pixel_values_ref_keypoint = pixel_values_ref_keypoint.squeeze(0)

        # 这里还有一个drop_image_embeds 没写
        sample = dict(
            pixel_values=pixel_values,  # 视频帧
            pixel_values_mesh=pixel_values_mesh,  # 视频帧mesh
            pixel_values_keypoint=pixel_values_keypoint,  # 视频帧关键点
            clip_ref_image=clip_ref_image,
            pixel_values_ref_img=pixel_values_ref_img,  # ref图
            pixel_values_ref_mesh=pixel_values_ref_mesh,  # ref mesh
            pixel_values_ref_keypoint=pixel_values_ref_keypoint  # ref 关键点
        )

        return sample


def collate_fn(data):
    pixel_values=torch.stack([example["pixel_values"] for example in data])
    pixel_values_mesh=torch.stack([example["pixel_values_mesh"] for example in data])
    pixel_values_keypoint=torch.stack([example["pixel_values_keypoint"] for example in data])

    clip_ref_image=torch.cat([example["clip_ref_image"] for example in data])

    pixel_values_ref_img=torch.stack([example["pixel_values_ref_img"] for example in data])
    pixel_values_ref_mesh=torch.stack([example["pixel_values_ref_mesh"] for example in data])
    pixel_values_ref_keypoint=torch.stack([example["pixel_values_ref_keypoint"] for example in data])

    return {
         "pixel_values": pixel_values,
         "pixel_values_mesh": pixel_values_mesh,
         "pixel_values_keypoint": pixel_values_keypoint,
         "clip_ref_image": clip_ref_image,
         "pixel_values_ref_img": pixel_values_ref_img,
         "pixel_values_ref_mesh": pixel_values_ref_mesh,
         "pixel_values_ref_keypoint": pixel_values_ref_keypoint,
   }






# if __name__=='__main__':
#     data_dir='/remote-home/share/yfsong/shipu/dataset_video_true_2'
#     dataset=MyDataset(data_dir,is_image=True)
#     dataloader=  train_dataloader=torch.utils.data.DataLoader(dataset,
#                                                  batch_size=1,
#                                                  shuffle=True,
#                                                  num_workers=4,
#                                                  drop_last=True,
#                                                  collate_fn=collate_fn,
#                                           )
#
#     count=0
#     for i in dataloader:
#         if (count<1):
#             print(i)
#             count+=1
#         else:
#             break