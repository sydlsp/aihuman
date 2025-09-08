import sys
import os
import torch
import torch.nn as nn
from einops import rearrange



__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../smirk')))

from AniPortrait.src.models.pose_guider import PoseGuider,Transformer2DModel

class new_poseguider(nn.Module):
    def __init__(self,noise_latent_channels=320):
        super().__init__()
        self.conv_2d=nn.Conv2d(in_channels=6,out_channels=3,kernel_size=(1,1))
        self.conv_3d=nn.Conv3d(in_channels=6,out_channels=3,kernel_size=(1,1,1))
        self.poseguider=PoseGuider(noise_latent_channels=noise_latent_channels)

    def forward(self,ref_img_mesh,ref_img_keypoint,pose_img_mesh,pose_img_keypoint):
        # pose_mesh_pic/pose_keypoint_pic:[bs,channels,frames,h,w]
        # ref_img_mesh/ref_img_keypoint:[bs,channels,h,w]
        fea_pic=torch.cat([pose_img_mesh,pose_img_keypoint],dim=1)
        ref_img=torch.cat([ref_img_mesh,ref_img_keypoint],dim=1)

        fea_pic=self.conv_3d(fea_pic)
        ref_img=self.conv_2d(ref_img)
        out_put=self.poseguider(fea_pic,ref_img)
        return out_put

    # 这里补了一下dtype，也是不得已而为之的办法，用装饰器修饰一下，假装是一个属性
    @property
    def dtype(self):
        return self.poseguider.dtype


class new_poseguider_1(nn.Module):
    """
    新版本的pose_guider,在上个版本中我们仅仅是对mesh和kp进行了简单的拼接，用3d卷积调整了一下大小，然后送入了poseguider中
    在这个版本中，我们尝试使用crossatten来学习mesh和kp之间的对应关系
    这进来的都是256*256的图片，在这里面就写固定了，后续修改的话应该就是sample_size[0],sample_size[1]这两个参数
    """
    def __init__(self,noise_latent_channels=320):
        super().__init__()

        # 这里先做用同一个注意力网络来处理mesh和kp，看看能不能跑
        self.linear_in=nn.Linear(256*256,256)
        self.multihead_attn=nn.MultiheadAttention(embed_dim=256,num_heads=8)
        self.linear_out=nn.Linear(256,256*256)
        self.poseguider=PoseGuider(noise_latent_channels=noise_latent_channels)

        self.conv_2d=nn.Conv2d(in_channels=6,out_channels=3,kernel_size=(1,1))
        self.conv_3d=nn.Conv3d(in_channels=6,out_channels=3,kernel_size=(1,1,1))

    def forward(self,ref_img_mesh,ref_img_keypoint,pose_img_mesh,pose_img_keypoint):
        # pose_mesh_pic/pose_keypoint_pic:[bs,channels,frames,h,w]
        # ref_img_mesh/ref_img_keypoint:[bs,channels,h,w]
        fea_pic=torch.cat([pose_img_mesh,pose_img_keypoint],dim=1)
        ref_img=torch.cat([ref_img_mesh,ref_img_keypoint],dim=1)  # 这样拼完都是6通道

        fea_pic_tensor=rearrange(fea_pic,"b c f h w -> b (c f) (h w)")  # [bs,channels*frames,h*w]
        ref_img_tensor=rearrange(ref_img,"b c h w -> b c (h w)") # [bs,channels,(h*w)]


        fea_pic_tensor=self.linear_in(fea_pic_tensor)
        fea_pic_tensor=self.multihead_attn(fea_pic_tensor,fea_pic_tensor,fea_pic_tensor)[0]
        fea_pic_tensor=self.linear_out(fea_pic_tensor)

        ref_img_tensor=self.linear_in(ref_img_tensor)
        ref_img_tensor=self.multihead_attn(ref_img_tensor,ref_img_tensor,ref_img_tensor)[0]
        ref_img_tensor=self.linear_out(ref_img_tensor)

        # 把形状修改回来
        fea_pic_tensor=rearrange(fea_pic_tensor,"b (c f) (h w) -> b c f h w",c=6,f=fea_pic_tensor.shape[1]//6,h=256)
        ref_img_tensor=rearrange(ref_img_tensor,"b c (h w) -> b c h w",h=256)


        # 这里还是要过一遍卷积，来接水管
        fea_pic=self.conv_3d(fea_pic_tensor)
        ref_img=self.conv_2d(ref_img_tensor)
        out_put=self.poseguider(fea_pic,ref_img)
        return out_put

    # 这里补了一下dtype，也是不得已而为之的办法，用装饰器修饰一下，假装是一个属性
    @property
    def dtype(self):
        return self.poseguider.dtype


class new_poseguider_2(nn.Module):
    """
    这里我们改的是注意力的组合形式
    """
    def __init__(self,noise_latent_channels=320):
        super().__init__()

        # 这里先做用同一个注意力网络来处理mesh和kp，看看能不能跑
        self.linear_in=nn.Linear(256*256,256)
        self.multihead_attn=nn.MultiheadAttention(embed_dim=256,num_heads=8,batch_first=True)  # 这里我们加了个参数batch_first=True，表明我们在使用的时候第一维是batch_size
        self.linear_out=nn.Linear(256,256*256)
        self.poseguider=PoseGuider(noise_latent_channels=noise_latent_channels)

        self.conv_2d=nn.Conv2d(in_channels=6,out_channels=3,kernel_size=(1,1))
        self.conv_3d=nn.Conv3d(in_channels=6,out_channels=3,kernel_size=(1,1,1))

    def forward(self,ref_img_mesh,ref_img_keypoint,pose_img_mesh,pose_img_keypoint):
        # pose_mesh_pic/pose_keypoint_pic:[bs,channels,frames,h,w]
        # ref_img_mesh/ref_img_keypoint:[bs,channels,h,w]
        fea_pic=torch.cat([pose_img_mesh,pose_img_keypoint],dim=1)
        ref_img=torch.cat([ref_img_mesh,ref_img_keypoint],dim=1)  # 这样拼完都是6通道

        fea_pic_tensor=rearrange(fea_pic,"b c f h w -> (b f) c (h w)")  # [bs,channels*frames,h*w]
        ref_img_tensor=rearrange(ref_img,"b c h w -> b c (h w)") # [bs,channels,(h*w)]


        fea_pic_tensor=self.linear_in(fea_pic_tensor)
        fea_pic_tensor=self.multihead_attn(fea_pic_tensor,fea_pic_tensor,fea_pic_tensor)[0]
        fea_pic_tensor=self.linear_out(fea_pic_tensor)

        ref_img_tensor=self.linear_in(ref_img_tensor)
        ref_img_tensor=self.multihead_attn(ref_img_tensor,ref_img_tensor,ref_img_tensor)[0]
        ref_img_tensor=self.linear_out(ref_img_tensor)

        # 把形状修改回来
        fea_pic_tensor=rearrange(fea_pic_tensor,"(b f) c (h w) -> b c f h w",c=6,f=fea_pic_tensor.shape[1]//6,h=256)
        ref_img_tensor=rearrange(ref_img_tensor,"b c (h w) -> b c h w",h=256)


        # 这里还是要过一遍卷积，来接水管
        fea_pic=self.conv_3d(fea_pic_tensor)
        ref_img=self.conv_2d(ref_img_tensor)
        out_put=self.poseguider(fea_pic,ref_img)
        return out_put

    # 这里补了一下dtype，也是不得已而为之的办法，用装饰器修饰一下，假装是一个属性
    @property
    def dtype(self):
        return self.poseguider.dtype


class new_poseguider_3(nn.Module):
    """
    新版本的pose_guider,
    在这个版本中我们用Transformer块来进行信息的交互
    """
    def __init__(self,noise_latent_channels=320):
        super().__init__()

        # 这里我们还是用同一个网络
        # 这里是按照256*256来设置的这个卷积和反卷积，暂时还是用同一个网络
        self.in_conv = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=4,stride=8,padding=1)
        self.cross_attn = Transformer2DModel(in_channels=3, norm_num_groups=3)
        self.out_conv = nn.ConvTranspose2d(in_channels=3,out_channels=3,kernel_size=4,stride=8,padding=1,output_padding=6)

        self.poseguider=PoseGuider(noise_latent_channels=noise_latent_channels)


    def forward(self,ref_img_mesh,ref_img_keypoint,pose_img_mesh,pose_img_keypoint):
        # pose_mesh_pic/pose_keypoint_pic:[bs,channels,frames,h,w]
        # ref_img_mesh/ref_img_keypoint:[bs,channels,h,w]

        batch_size=pose_img_mesh.shape[0]
        # 这里是新补充的自己写的
        # 先把pose_mesh_pic和pose_keypoint_pic的形状调整一下
        pose_img_mesh=rearrange(pose_img_mesh,"b c f h w -> (b f) c h w")
        pose_img_keypoint=rearrange(pose_img_keypoint,"b c f h w -> (b f) c h w")

        # 下面开始做卷积

        pose_img_mesh=self.in_conv(pose_img_mesh)
        pose_img_keypoint=self.in_conv(pose_img_keypoint)

        ref_img_mesh=self.in_conv(ref_img_mesh)
        ref_img_keypoint=self.in_conv(ref_img_keypoint)

        # 开始用Transformer做信息交互
        pose_fea=self.cross_attn(pose_img_mesh,pose_img_keypoint)
        ref_fea=self.cross_attn(ref_img_mesh,ref_img_keypoint)

        # 用反卷积来对形状
        pose_fea=self.out_conv(pose_fea)
        ref_fea=self.out_conv(ref_fea)

        # 这里要把pose_fea的形状转化为5维的
        pose_fea=rearrange(pose_fea,"(b f) c h w -> b c f h w",b=batch_size)
        out_put=self.poseguider(pose_fea,ref_fea)
        return out_put

    # 这里补了一下dtype，也是不得已而为之的办法，用装饰器修饰一下，假装是一个属性
    @property
    def dtype(self):
        return self.poseguider.dtype

# model=new_poseguider()
# for name,param in model.named_parameters():
#     print(name,param.shape)
