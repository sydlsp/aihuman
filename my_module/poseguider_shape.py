"""
本代码用来探究poseguider的输入输出形状
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch

from AniPortrait.src.models.pose_guider import PoseGuider

pose_guider=PoseGuider()

ref_tensor=torch.randn(size=[1,3,256,256])
pose_tensor=torch.randn(size=[1,3, 1,256,256])


pose_fea=pose_guider(pose_tensor,ref_tensor)

print(type(pose_fea))

"""
pose_fea是一个list，这个list里面有五个元素，每个元素都是一个tensor
"""

print(len(pose_fea))
for i in pose_fea:
    print(type(i))
    print(i.shape)

"""
<class 'torch.Tensor'>
torch.Size([1, 320, 1, 32, 32])
<class 'torch.Tensor'>
torch.Size([1, 320, 1, 16, 16])
<class 'torch.Tensor'>
torch.Size([1, 640, 1, 8, 8])
<class 'torch.Tensor'>
torch.Size([1, 1280, 1, 4, 4])
<class 'torch.Tensor'>
torch.Size([1, 1280, 1, 4, 4])
"""


