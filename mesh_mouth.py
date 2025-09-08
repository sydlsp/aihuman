import torch
import mediapipe as mp
import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
"""
从mesh图中检测出嘴的位置
"""

fold_path="/remote-home/share/yfsong/shipu/dataset_video_true/shengyu/video_2_high_quanlity_clip_90"

mesh_file=[file for file in os.listdir(fold_path) if "mesh" in file]
frame_file=[file for file in os.listdir(fold_path) if "frame" in file]

# 初始化MediaPipe Face Mesh检测器
mp_face_mesh=mp.solutions.face_mesh



# MediaPipe Face Mesh嘴部区域的关键点索引
mouth_indices = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311,
    310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]

with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    count=0
    for frame in mesh_file:
        frame=cv2.imread(os.path.join(fold_path,frame))
        results=face_mesh.process(frame)
        if results.multi_face_landmarks:
            count+=1
            mouth_landmarks=[results.multi_face_landmarks[0].landmark[i] for i in mouth_indices]

        print(results.multi_face_landmarks)
        print(count)
