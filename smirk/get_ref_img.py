import os

import cv2

from utils.mediapipe_utils import run_mediapipe

dir_path="/remote-home/share/yfsong/shipu/dataset_video"

ref_dir_path=os.path.join(dir_path,"ref_img")

if not os.path.exists(ref_dir_path):
    os.makedirs(ref_dir_path)

"""
在这里我们暂定ref_img都是来源于同一个人的
"""

seg_dir_path=os.path.join(dir_path,"segments")

video_file_list=os.listdir(seg_dir_path)[0:5]

for video_file in video_file_list:
    pic_count=0

    video_file_path=os.path.join(seg_dir_path,video_file)

    video_file_name=video_file.split(".")[0]

    cap=cv2.VideoCapture(video_file_path)

    while True:
        ret,frame=cap.read()

        if not ret:
            break

        # 检测一下人脸关键点
        kpt_mediapipe=run_mediapipe(frame)

        # 还是基于人体关键点
        if kpt_mediapipe is None:
            continue

        frame_name=video_file_name+"_"+"ref_img"+"_"+str(pic_count).zfill(4)+".jpg"
        frame_path=os.path.join(ref_dir_path,frame_name)
        #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imwrite(frame_path,frame)
        pic_count+=1

        # 下面保存帧数据为图片就行了
