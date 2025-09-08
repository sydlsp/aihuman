from AniPortrait.src.models.pose_guider_1 import PoseGuider as pose_guider
import torch

pose_guider= pose_guider().to('cuda')

x=torch.randn(4,3,1,512,512).to('cuda')
ref_x=torch.randn(4,3,512,512).to('cuda')

result=pose_guider(x,ref_x)

for i in result:
    print(i.shape)
