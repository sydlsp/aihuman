import torch
from AniPortrait.src.models.pose_guider import PoseGuider
from my_module.new_poseguider import new_poseguider_1
import torch.nn as nn


# model=PoseGuider()

# batch_size,channels,h,w
ref_img=torch.randn(size=[1,3,1,1]).to("cuda")
# batch_size,channels,frames,h,w
mesh_pic=torch.randn(size=[1,3,1,16,16]).to("cuda")
keypoint_pic=torch.randn(size=[1,3,1,16,16]).to("cuda")

fea_pic=torch.cat([mesh_pic,keypoint_pic],dim=1)

# 可以用3d卷积来调大小
conv=nn.Conv3d(in_channels=6,out_channels=3,kernel_size=(1,1,1),stride=(1,1,1),padding=(1,1,1)).to("cuda")
fea_pic=conv(fea_pic)
print(fea_pic.shape)
#out_put=model(fea_pic,ref_img)


#print(out_put.shape)