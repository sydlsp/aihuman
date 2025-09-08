original="/remote-home/share/yfsong/shipu/smirk/assets/FLAME_masks/FLAME_masks.pkl"
destination="/remote-home/share/yfsong/shipu/smirk/assets/FLAME_masks/FLAME_masks_200.pkl"



content =''
outsize = 0


with open(original, 'rb') as infile:
  content = infile.read()
with open(destination, 'wb') as output:
  for line in content.splitlines():
      outsize += len(line) + 1
      output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))


class new_poseguider_2(nn.Module):
    """
    新版本的pose_guider,
    在这个版本中我们用Transformer块来进行信息的交互
    """
    def __init__(self,noise_latent_channels=320):
        super().__init__()

        # 这里我们还是用同一个网络
        # 这里是按照256*256来设置的这个卷积和反卷积，暂时还是用同一个网络
        self.in_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=4, padding=1)
        self.cross_attn = Transformer2DModel(in_channels=3, norm_num_groups=3)
        self.out_conv = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=4, padding=1,output_padding=3)

        self.pose_guider=PoseGuider(noise_latent_channels=noise_latent_channels)


    def forward(self,ref_img_mesh,ref_img_keypoint,pose_img_mesh,pose_img_keypoint):
        # pose_mesh_pic/pose_keypoint_pic:[bs,channels,frames,h,w]
        # ref_img_mesh/ref_img_keypoint:[bs,channels,h,w]


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
        out_put=self.poseguider(pose_fea,ref_fea)
        return out_put

    # 这里补了一下dtype，也是不得已而为之的办法，用装饰器修饰一下，假装是一个属性
    @property
    def dtype(self):
        return self.poseguider.dtype
