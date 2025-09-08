# print(ref_mesh_tensor.shape)
# print(ref_keypoint_tensor.shape)
# print(pose_mesh_tensor.shape)
# print(pose_keypoint_tensor.shape)
# torch.Size([1, 3, 256, 256])
# torch.Size([1, 3, 256, 256])
# torch.Size([1, 3, 1, 256, 256])
# torch.Size([1, 3, 1, 256, 256])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
from einops import rearrange


gpu_id = torch.cuda.current_device()
print(f"当前使用的GPU编号: {gpu_id}")


batch_size,channels,frames,h,w=1,3,1,256,256

ref_mesh_tensor=torch.randn(size=[batch_size,channels,h,w]).to("cuda")
ref_keypoint_tensor=torch.randn(size=[batch_size,channels,h,w]).to("cuda")

# 先做卷积
conv_1=nn.Conv2d(in_channels=channels,out_channels=3,kernel_size=(3,3),stride=4).to("cuda")
ref_mesh_tensor=conv_1(ref_mesh_tensor)
ref_keypoint_tensor=conv_1(ref_keypoint_tensor)

# 然后把卷积完的结果给拉平
ref_mesh_tensor=rearrange(ref_mesh_tensor,"b c h w -> b  c (h w)")
print(ref_mesh_tensor.shape)

# 搞一个多头的注意力层
multihead_attn=nn.MultiheadAttention(embed_dim=ref_mesh_tensor.shape[2],num_heads=1,batch_first=True).to("cuda")
atten_output,atten_weight=multihead_attn(ref_mesh_tensor,ref_mesh_tensor,ref_mesh_tensor)
print(atten_output.shape)


