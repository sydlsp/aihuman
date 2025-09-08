import copy

import cv2
import os
import torch
from src.smirk_encoder import SmirkEncoder
from src.smirk_generator import SmirkGenerator
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
from datasets.base_dataset import create_mask
from utils.mediapipe_utils import run_mediapipe
import src.utils.masking as masking_utils
from skimage.transform import estimate_transform, warp

import mediapipe as mp
import numpy as np

video_path="/remote-home/share/yfsong/shipu/video_src/part_video/video_54/segments/video_54_high_quanlity_clip_0.mp4"

"""
给定视频，对于每一个视频帧，将关键点和mesh融合到一张图上
"""
def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform

# 先初始化smirk一系列东西
checkpoint_path="/remote-home/share/yfsong/shipu/smirk/pretrained_models/SMIRK_em1.pt"

# 加载smirk的编码器
smirk_encoder = SmirkEncoder().to("cuda")
checkpoint = torch.load(checkpoint_path)
checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

smirk_encoder.load_state_dict(checkpoint_encoder)
smirk_encoder.eval()

# 加载smirk的生成器，这个生成器其实就是重建图像的
smirk_generator=SmirkGenerator(in_channels=6,out_channels=3,init_features=32,res_blocks=5).to("cuda")
checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if 'smirk_generator' in k} # checkpoint includes both smirk_encoder and smirk_generator
smirk_generator.load_state_dict(checkpoint_generator)
smirk_generator.eval()

# 加载flame模型，flame是根据编码器的参数来生成mesh的
flame_model_path="/remote-home/share/yfsong/shipu/smirk/assets/FLAME2020/generic_model.pkl"
flame_lmk_embedding_path="/remote-home/share/yfsong/shipu/smirk/assets/landmark_embedding.npy" # 这个就是在assert文件夹里，可以不写
flame=FLAME(flame_model_path=flame_model_path,flame_lmk_embedding_path=flame_lmk_embedding_path).to("cuda")

# 加载Render，render是根据mesh来渲染图像的
renderer=Renderer().to("cuda")

# 初始化mediapipe姿态关键点检测模块
mp_pose=mp.solutions.pose
pose=mp_pose.Pose()

# 初始化mediapipe手部关键点检测模块
mp_hands=mp.solutions.hands
hands=mp_hands.Hands()

# mediapipe绘图模块
mp_drawing=mp.solutions.drawing_utils

# 这里我们手动设置一下绘图规格
landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2)



if os.path.exists(video_path):
    cap=cv2.VideoCapture(video_path)

    frame_count=0
    # 开始逐帧读取视频
    while True:
        ret,frame=cap.read()

        # 视频结束或出现错误ret是False
        if not ret:
            break

        frame_count+=1
        orig_image_height, orig_image_width, _ = frame.shape # 大图
        print(orig_image_height, orig_image_width)

        # 检测关键点
        kpt_mediapipe = run_mediapipe(frame)

        # 把人脸给裁出来
        if kpt_mediapipe is None:
            continue
        kpt_mediapipe=kpt_mediapipe[..., :2]
        tform=crop_face(frame,kpt_mediapipe,scale=1.4,image_size=224)
        cropped_image=warp(frame,tform.inverse,output_shape=(224,224),preserve_range=True).astype(np.uint8)

        cropped_kpt_mediapipe=np.dot(tform.params,np.hstack([kpt_mediapipe,np.ones([kpt_mediapipe.shape[0],1])]).T).T
        cropped_kept_mediapipe=cropped_kpt_mediapipe[:,:2]

        "_____________________________________________"

        # cropped_image=frame
        # cropped_kpt_mediapipe=kpt_mediapipe

        # 对图像进行裁剪
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224, 224))  # cv2.resize是插值或者抽值来缩放图片，不是直接裁剪

        pose_image=copy.deepcopy(frame)  # 这里我们深拷贝一下，pose_image用来检测姿态关键点


        # 这里先这样写
        smirk_image=copy.deepcopy(cropped_image)

        smirk_image = torch.tensor(smirk_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        smirk_image = smirk_image.to("cuda")

        # 在这里补一下cropped_image转化为tensor
        # cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # cropped_image=cropped_image.to("cuda")

        # 拿到smirk的编码器的输出
        outputs=smirk_encoder(smirk_image)

        # 生成mesh,flame模型的输出是有关键点的
        flame_mesh_output=flame.forward(outputs)

        print(type(flame_mesh_output))
        exit()
        # 把mesh变成图片
        renderer_output = renderer.forward(flame_mesh_output['vertices'], outputs['cam'],
                                           landmarks_fan=flame_mesh_output['landmarks_fan'],
                                           landmarks_mp=flame_mesh_output['landmarks_mp'])

        rendered_img = renderer_output['rendered_img']
        # rendered_img=rendered_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0
        # cv2.imwrite("./rendered_img.jpg",rendered_img)

        # 在原图片尺寸上进行渲染
        rendered_img_numpy= (rendered_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)
        rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(orig_image_height, orig_image_width),preserve_range=True).astype(np.uint8)
        rendered_img_orig = torch.Tensor(rendered_img_orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        full_image=torch.tensor(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float()/255.0
        grid=torch.cat([full_image,rendered_img_orig],dim=3)
        # 把rendered_img_orig转化为图片
        rendered_img_orig_pic=rendered_img_orig.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0


        # 来重建图像，这都可以不要
        mask_ratio_mul = 0.5
        mask_ratio = 0.01
        mask_dilation_radius = 10

        hull_mask = create_mask(np.array(cropped_kpt_mediapipe.astype(np.int32)), (224, 224))

        face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()

        rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
        tmask_ratio = mask_ratio * mask_ratio_mul

        # 基于FLAME模型的三角形面片生成一个均匀的面部遮罩，npoints是面部的点
        npoints, _ = masking_utils.mesh_based_mask_uniform_faces(renderer_output['transformed_vertices'],
                                                                 flame_faces=flame.faces_tensor,
                                                                 face_probabilities=face_probabilities,
                                                                 mask_ratio=tmask_ratio)

        pmask = torch.zeros_like(rendered_mask)
        rsing = torch.randint(0, 2, (npoints.size(0),)).to(npoints.device) * 2 - 1
        rscale = torch.rand((npoints.size(0),)).to(npoints.device) * (mask_ratio_mul - 1) + 1
        rbound = (npoints.size(1) * (1 / mask_ratio_mul) * (rscale ** rsing)).long()

        for bi in range(npoints.size(0)):
            pmask[bi, :, npoints[bi, :rbound[bi], 1], npoints[bi, :rbound[bi], 0]] = 1

        hull_mask = torch.from_numpy(hull_mask).type(dtype=torch.float32).unsqueeze(0).to("cuda")

        extra_points = smirk_image * pmask
        masked_img = masking_utils.masking(smirk_image, hull_mask, extra_points, mask_dilation_radius,
                                           rendered_mask=rendered_mask)

        smirk_generator_input = torch.cat([rendered_img, masked_img], dim=1)

        # reconstructed_img是还原的图像
        reconstructed_img = smirk_generator(smirk_generator_input)

        reconstructed_img_numpy = (reconstructed_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        reconstructed_img_orig = warp(reconstructed_img_numpy, tform,output_shape=(orig_image_height, orig_image_width), preserve_range=True).astype(np.uint8)
        reconstructed_img_orig = torch.Tensor(reconstructed_img_orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        grid = torch.cat([grid, reconstructed_img_orig], dim=3)

        grid_numpy = grid.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
        grid_numpy = grid_numpy.astype(np.uint8)
        grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)

        cv2.imwrite("./grid_1.jpg",grid_numpy)

        # 检测关键点
        pose_results=pose.process(pose_image)
        hands_results=hands.process(pose_image)

        # 绘制躯干和手部关键点
        if pose_results.pose_landmarks:
            print(type(pose_results.pose_landmarks.landmark))
            # 这里pose_image其实是np.ndarray
            mp_drawing.draw_landmarks(rendered_img_orig_pic,pose_results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec,landmark_drawing_spec)

        fuse_image = cv2.cvtColor(rendered_img_orig_pic, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./pose_image_result_111.jpg",fuse_image)

        # 那其实下面要做的事就是把关键点和之前的mesh映射出来的图片进行拼接，然后保存



        # if hands_results.multi_hand_landmarks:
        #     for hand_landmarks in hands_results.multi_hand_landmarks:
        #         mp_drawing.draw_landmarks(pose_image,hand_landmarks,mp_hands.HAND_CONNECTIONS,landmark_drawing_spec,landmark_drawing_spec)
        # pose_image=cv2.cvtColor(pose_image,cv2.COLOR_RGB2BGR)
        # cv2.imwrite("./pose_image.jpg",pose_image)

        if frame_count==1:
            exit()








