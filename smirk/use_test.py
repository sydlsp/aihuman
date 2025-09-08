import os.path
import timm

import torch
import cv2
import numpy as np

from src.smirk_encoder import SmirkEncoder
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
from utils.mediapipe_utils import run_mediapipe
import src.utils.masking as masking_utils

checkpoint_path="/remote-home/share/yfsong/shipu/smirk/pretrained_models/SMIRK_em1.pt"

#os.environ["HF_HOME"]="/remote-home/share/yfsong/shipu/smirk/cache"


# 加载smirk的编码器
smirk_encoder = SmirkEncoder().to("cuda")
checkpoint = torch.load(checkpoint_path)
checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

smirk_encoder.load_state_dict(checkpoint_encoder)
smirk_encoder.eval()

# 加载smirk的生成器
from src.smirk_generator import SmirkGenerator
smirk_generator=SmirkGenerator(in_channels=6,out_channels=3,init_features=32,res_blocks=5).to("cuda")
checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if 'smirk_generator' in k} # checkpoint includes both smirk_encoder and smirk_generator
smirk_generator.load_state_dict(checkpoint_generator)
smirk_generator.eval()

from src.FLAME.FLAME import FLAME

# 加载flame模型，这里要手动改一下权重路径
# flame模型根据参数生成3D模型
flame_model_path="/remote-home/share/yfsong/shipu/smirk/assets/FLAME2020/generic_model.pkl"
flame_lmk_embedding_path="/remote-home/share/yfsong/shipu/smirk/assets/landmark_embedding.npy" # 这个就是在assert文件夹里，可以不写
flame=FLAME(flame_model_path=flame_model_path,flame_lmk_embedding_path=flame_lmk_embedding_path).to("cuda")

from src.renderer.renderer import Renderer
renderer=Renderer().to("cuda")

image=cv2.imread("use_demo.jpg")

orig_image_height, orig_image_width, _ = image.shape
print(orig_image_height, orig_image_width)

# 这里要点run_mediapipe,修改最上面base_options的路径
# 这里其实是检测关键点
kpt_mediapipe = run_mediapipe(image)

print(type(kpt_mediapipe))
print(kpt_mediapipe.shape)

cropped_image=image
cropped_kpt_mediapipe=kpt_mediapipe

# 对图像进行裁剪、转化为tensor等相关操作
cropped_image=cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB)
cropped_image=cv2.resize(cropped_image,(224,224)) # cv2.resize是插值或者抽值来缩放图片，不是直接裁剪
cropped_image=torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
cropped_image=cropped_image.to("cuda")

outputs=smirk_encoder(cropped_image)

# print(type(outputs))
# for key in outputs:
#     print(key,outputs[key].shape)

"""
encoder的输出包含如下方面：
pose_params torch.Size([1, 3])
cam torch.Size([1, 3])
shape_params torch.Size([1, 300])
expression_params torch.Size([1, 50])
eyelid_params torch.Size([1, 2])
jaw_params torch.Size([1, 3])
"""

flame_output=flame.forward(outputs)

# print(type(flame_output))
# for key in flame_output:
#     print(key,flame_output[key].shape)

"""
flame的输出包含如下方面:
vertices torch.Size([1, 5023, 3])
landmarks_fan torch.Size([1, 68, 3])
landmarks_fan_3d torch.Size([1, 68, 3])
landmarks_mp torch.Size([1, 105, 3])
"""

renderer_output=renderer.forward(flame_output['vertices'],outputs['cam'],landmarks_fan=flame_output['landmarks_fan'],landmarks_mp=flame_output['landmarks_mp'])
rendered_img=renderer_output['rendered_img']
#

# 我们这里不在原图上进行渲染
grid=torch.cat([cropped_image,rendered_img],dim=3)

# 这里就是把3DMesh给渲染成图片了
rendered_out=rendered_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0
cv2.imwrite("grid.jpg",rendered_out)

# 使用smirk的生成器
if kpt_mediapipe is None:
    print('Could not find landmarks for the image using mediapipe and cannot create the hull mask for the smirk generator. Exiting...')
    exit()

mask_ratio_mul=0.5
mask_ratio=0.01
mask_dilation_radius=10

hull_mask=create_mask(np.array(cropped_kpt_mediapipe.astype(np.int32)),(224,224))

face_probabilities=masking_utils.load_probabilities_per_FLAME_triangle()

rendered_mask=1-(rendered_img==0).all(dim=1,keepdim=True).float()
tmask_ratio=mask_ratio*mask_ratio_mul

# 基于FLAME模型的三角形面片生成一个均匀的面部遮罩，npoints是面部的点
npoints,_=masking_utils.mesh_based_mask_uniform_faces(renderer_output['transformed_vertices'],
                                                      flame_faces=flame.faces_tensor,
                                                      face_probabilities=face_probabilities,
                                                      mask_ratio=tmask_ratio)

pmask = torch.zeros_like(rendered_mask)
rsing = torch.randint(0, 2, (npoints.size(0),)).to(npoints.device) * 2 - 1
rscale = torch.rand((npoints.size(0),)).to(npoints.device) * (mask_ratio_mul - 1) + 1
rbound =(npoints.size(1) * (1/mask_ratio_mul) * (rscale ** rsing)).long()


for bi in range(npoints.size(0)):
    pmask[bi, :, npoints[bi, :rbound[bi], 1], npoints[bi, :rbound[bi], 0]] = 1

hull_mask = torch.from_numpy(hull_mask).type(dtype=torch.float32).unsqueeze(0).to("cuda")

extra_points = cropped_image * pmask
masked_img = masking_utils.masking(cropped_image, hull_mask, extra_points, mask_dilation_radius,
                                   rendered_mask=rendered_mask)

smirk_generator_input = torch.cat([rendered_img, masked_img], dim=1)

# reconstructed_img是还原的图像
reconstructed_img = smirk_generator(smirk_generator_input)

grid = torch.cat([grid, reconstructed_img], dim=3)

grid_numpy = grid.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0
grid_numpy = grid_numpy.astype(np.uint8)
grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)

cv2.imwrite("use_demo_result_1.jpg",grid_numpy)





