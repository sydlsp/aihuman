from AniPortrait.src.models.pose_guider import Transformer2DModel
import torch
import torch.nn as nn
import os
#
# input_tensor=torch.randn(64,320,32,32).to(device="cuda:3")
# print(input_tensor.shape)
# print(input_tensor.device)
# model=Transformer2DModel(in_channels=320,norm_num_groups=1).to("cuda:3")
# print(model(input_tensor,input_tensor).shape)
# print("ok")

# tf_model = Transformer2DModel(
#     in_channels=320
#     ).to('cuda:3')
#
# input_data = torch.randn(4,320,32,32).to(device="cuda:3")
# # input_emb = torch.randn(4,1,768).to(device="cuda")
# input_emb = torch.randn(4,320,32,32).to(device="cuda:3")
# o1 = tf_model(input_data, input_emb)
# print(o1.shape)

tensor=torch.randn(1,3,256,256).to("cuda:3")

conv_1=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=4,stride=8,padding=1).to("cuda:3")

output=conv_1(tensor)

re_conv=nn.ConvTranspose2d(in_channels=3,out_channels=3,kernel_size=4,stride=8,padding=1,output_padding=6).to("cuda:3")

output=re_conv(output)
print(output.shape)