"""
专门用于检测手部关键点的代码
"""
import os

import cv2
import mediapipe as mp
import numpy as np

# 这个函数打个一张图使用的样
# def detect_hand_keypoint(image_path,save_path):
#
#     hands=mp.solutions.hands.Hands()
#     mp_drawing=mp.solutions.drawing_utils
#     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),thickness=1,circle_radius=2)
#
#     img=cv2.imread(image_path)
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#
#     height,width=img.shape[0],img.shape[1]
#
#     hand_results=hands.process(img)
#
#     black_image=np.zeros((height,width,3),dtype=np.uint8)
#
#     # 这还真是奇怪了，为什么这段代码摘出来就能检测到手，平时就不行
#     # 加了必须检测两只手这个条件又变严格了
#     if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks)==2:
#         for hand_landmarks in hand_results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(black_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
#                                            landmark_drawing_spec,landmark_drawing_spec)
#
#         save_hand_pic=cv2.cvtColor(black_image,cv2.COLOR_RGB2BGR)
#
#         cv2.imwrite(save_path,save_hand_pic)
#         print("ok")

if __name__=="__main__":

    hands = mp.solutions.hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)

    video_folder_path="/remote-home/share/yfsong/shipu/dataset_video_true_1/videos"
    video_folder=os.listdir(video_folder_path)

    save_floder = "/remote-home/share/yfsong/shipu/dataset_video_true_1/hands"

    count=0

    for folder in video_folder:
        frame_folder=os.path.join(video_folder_path,folder)
        frame_list=[file for file in os.listdir(frame_folder) if "frame" in file]
        print(frame_list)

        for file in frame_list:
            image_path=os.path.join(frame_folder,file)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            height, width = img.shape[0], img.shape[1]

            hand_results = hands.process(img)

            black_image = np.zeros((height, width, 3), dtype=np.uint8)

            # 这还真是奇怪了，为什么这段代码摘出来就能检测到手，平时就不行
            # 加了必须检测两只手这个条件又变严格了
            if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(black_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                              landmark_drawing_spec, landmark_drawing_spec)

                save_hand_pic = cv2.cvtColor(black_image, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(save_floder,f"{count}.jpg"), save_hand_pic)
                count+=1
                print("ok")
