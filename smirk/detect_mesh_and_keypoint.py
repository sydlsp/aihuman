"""
在这里，我们将检测视频帧中的mesh和关键点写成一个类用来调用
"""
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

class DetectMeshandKeypoint():

    def __init__(self):

        # 初始化smirk的一系列东西
        checkpoint_path = "/remote-home/share/yfsong/shipu/smirk/pretrained_models/SMIRK_em1.pt"

        # 初始化smirk编码器
        self.smirk_encoder = SmirkEncoder().to("cuda")

        checkpoint=torch.load(checkpoint_path)

        checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k}

        self.smirk_encoder.load_state_dict(checkpoint_encoder)
        self.smirk_encoder.eval()

        # 加载smirk的生成器，这里生成器是重建图像的
        self.smirk_generator=SmirkGenerator(in_channels=6,out_channels=3,init_features=32,res_blocks=5).to("cuda")
        checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if'smirk_generator' in k}
        self.smirk_generator.load_state_dict(checkpoint_generator)
        self.smirk_generator.eval()

        # 加载flame模型
        flame_model_path = "/remote-home/share/yfsong/shipu/smirk/assets/FLAME2020/generic_model.pkl"
        flame_lmk_embedding_path = "/remote-home/yfsong/shipu/mycode/smirk/assets/landmark_embedding.npy" # 这个地址是跟着smirk项目走的，还是要改的
        self.flame=FLAME(flame_model_path=flame_model_path,flame_lmk_embedding_path=flame_lmk_embedding_path).to("cuda")

        # 加载Render
        self.renderer=Renderer().to("cuda")

        # 初始化mediapipe姿态关键点检测模块
        self.pose=mp.solutions.pose.Pose()

        # 初始化mediapipe手部关键点检测模块
        self.hands=mp.solutions.hands.Hands()

        # 加载mediapipe绘图相关
        self.mp_drawing = mp.solutions.drawing_utils

        # 设置mediapipe绘图规格
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2)

    def crop_face(self,frame,landmarks, scale=1.0, image_size=224):
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

    def get_mesh_and_keypoint_pic(self,video_path,out_dir):
        """

        """
        if os.path.exists(video_path):
            cap=cv2.VideoCapture(video_path)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            frame_count=0

            while True:
                ret,frame=cap.read()

                if not ret:
                    break


                orig_image_height, orig_image_width, _ = frame.shape  # 大图

                # 检测人脸关键点
                kpt_mediapipe = run_mediapipe(frame)

                # 下面我们所做的一切都是在检测到人脸关键点基础上的
                if kpt_mediapipe is None:
                    continue

                kpt_mediapipe = kpt_mediapipe[..., :2]

                tform=self.crop_face(frame,kpt_mediapipe,scale=1.4,image_size=224)
                cropped_image = warp(frame, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

                cropped_kpt_mediapipe = np.dot(tform.params,np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T).T
                cropped_kept_mediapipe = cropped_kpt_mediapipe[:, :2]

                # 对图像进行裁剪
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                cropped_image = cv2.resize(cropped_image, (224, 224))  # cv2.resize是插值或者抽值来缩放图片，不是直接裁剪

                pose_image = copy.deepcopy(frame)  # 这里我们深拷贝一下，pose_image用来检测姿态关键点

                # 将smirk_image转化为tensor
                smirk_image = copy.deepcopy(cropped_image)

                smirk_image = torch.tensor(smirk_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                smirk_image = smirk_image.to("cuda")

                # 拿到smirk的编码器的输出
                outputs=self.smirk_encoder(smirk_image)

                # 生成mesh
                flame_mesh_output=self.flame.forward(outputs)

                # 把mesh变成图片
                renderer_output=self.renderer.forward(flame_mesh_output['vertices'], outputs['cam'],
                                           landmarks_fan=flame_mesh_output['landmarks_fan'],
                                           landmarks_mp=flame_mesh_output['landmarks_mp'])

                rendered_img=renderer_output['rendered_img']

                # 在原图片尺寸上进行渲染
                rendered_img_numpy = (rendered_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(orig_image_height, orig_image_width),preserve_range=True).astype(np.uint8)
                rendered_img_orig = torch.Tensor(rendered_img_orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                full_image = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                grid = torch.cat([full_image, rendered_img_orig], dim=3)
                # 把rendered_img_orig转化为图片
                rendered_img_orig_pic = rendered_img_orig.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0

                # 重建图像的一系列东西，其实都可以不看

                # mask_ratio_mul = 0.5
                # mask_ratio = 0.01
                # mask_dilation_radius = 10
                #
                # hull_mask = create_mask(np.array(cropped_kpt_mediapipe.astype(np.int32)), (224, 224))
                #
                # face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()
                #
                # rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
                # tmask_ratio = mask_ratio * mask_ratio_mul
                #
                # npoints, _ = masking_utils.mesh_based_mask_uniform_faces(renderer_output['transformed_vertices'],
                #                                                          flame_faces=self.flame.faces_tensor,
                #                                                          face_probabilities=face_probabilities,
                #                                                          mask_ratio=tmask_ratio)
                #
                # pmask = torch.zeros_like(rendered_mask)
                # rsing = torch.randint(0, 2, (npoints.size(0),)).to(npoints.device) * 2 - 1
                # rscale = torch.rand((npoints.size(0),)).to(npoints.device) * (mask_ratio_mul - 1) + 1
                # rbound = (npoints.size(1) * (1 / mask_ratio_mul) * (rscale ** rsing)).long()
                #
                # for bi in range(npoints.size(0)):
                #     pmask[bi, :, npoints[bi, :rbound[bi], 1], npoints[bi, :rbound[bi], 0]] = 1
                #
                # hull_mask = torch.from_numpy(hull_mask).type(dtype=torch.float32).unsqueeze(0).to("cuda")
                #
                # extra_points = smirk_image * pmask
                # masked_img = masking_utils.masking(smirk_image, hull_mask, extra_points, mask_dilation_radius,
                #                                    rendered_mask=rendered_mask)
                #
                # smirk_generator_input = torch.cat([rendered_img, masked_img], dim=1)
                #
                # reconstructed_img = self.smirk_generator(smirk_generator_input)
                #
                # reconstructed_img_numpy = (reconstructed_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                # reconstructed_img_orig = warp(reconstructed_img_numpy, tform,output_shape=(orig_image_height, orig_image_width),preserve_range=True).astype(np.uint8)
                # reconstructed_img_orig = torch.Tensor(reconstructed_img_orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                #
                # grid = torch.cat([grid, reconstructed_img_orig], dim=3)
                #
                # grid_numpy = grid.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
                # grid_numpy = grid_numpy.astype(np.uint8)
                # grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)

                #cv2.imwrite("grid.jpg",grid_numpy)

                # 检测关键点
                pose_results=self.pose.process(pose_image)
                hands_results=self.hands.process(pose_image)

                # 在这里生成一张全黑的图像
                black_image=np.zeros((orig_image_height, orig_image_width,3),dtype=np.uint8)
                # 绘制关键点
                if pose_results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(black_image, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, self.landmark_drawing_spec,self.landmark_drawing_spec)

                # 将上面的mesh渲染图像和关键点图像保存
                mesh_name=os.path.basename(video_path).split(".")[0]+"_"+"mesh"+"_"+str(frame_count).zfill(4)+".jpg"
                keypoint_name=os.path.basename(video_path).split(".")[0]+"_"+"keypoint"+"_"+str(frame_count).zfill(4)+".jpg"
                frame_name=os.path.basename(video_path).split(".")[0]+"_"+"frame"+"_"+str(frame_count).zfill(4)+".jpg"

                rendered_img_orig_pic=cv2.cvtColor(rendered_img_orig_pic,cv2.COLOR_BGR2RGB)
                keypoint_pic=cv2.cvtColor(black_image,cv2.COLOR_BGR2RGB)


                cv2.imwrite(os.path.join(out_dir,mesh_name),rendered_img_orig_pic)
                cv2.imwrite(os.path.join(out_dir,keypoint_name),keypoint_pic)
                cv2.imwrite(os.path.join(out_dir,frame_name),frame)
                frame_count+=1







