# import cv2
# import mediapipe as mp
# import numpy as np
#
# # 初始化Mediapipe手部检测器
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.8)
#
# # 读取图像
# image_path = '/data/shipu/train_data/data/shipu/dataset_true/videos/video_3_high_quanlity_clip_84/video_3_high_quanlity_clip_84_frame_0000.jpg'
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # 获取图像尺寸
# height, width, _ = image.shape
#
# # 初始化注释通道
# finger_count_channel = np.zeros((height, width), dtype=np.uint8)
# finger_joint_channel = np.zeros((height, width), dtype=np.uint8)
# left_right_channel = np.zeros((height, width), dtype=np.uint8)
#
# # 处理图像
# results = hands.process(image_rgb)
#
# if results.multi_hand_landmarks:
#     for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#         # 获取手部是左手还是右手
#         hand_label = handedness.classification[0].label
#         if hand_label == 'Left':
#             hand_value = 1
#         else:
#             hand_value = 2
#
#         # 获取手指关键点
#         for i, landmark in enumerate(hand_landmarks.landmark):
#             x, y = int(landmark.x * width), int(landmark.y * height)
#             if i % 4 == 0:  # 指尖
#                 cv2.circle(finger_count_channel, (x, y), 5, 1, -1)
#             elif i % 4 == 1:  # 第一指节
#                 cv2.circle(finger_joint_channel, (x, y), 5, 1, -1)
#             elif i % 4 == 2:  # 第二指节
#                 cv2.circle(finger_joint_channel, (x, y), 5, 2, -1)
#             elif i % 4 == 3:  # 第三指节
#                 cv2.circle(finger_joint_channel, (x, y), 5, 3, -1)
#
#         # 标记左右手区域
#         for landmark in hand_landmarks.landmark:
#             x, y = int(landmark.x * width), int(landmark.y * height)
#             cv2.circle(left_right_channel, (x, y), 5, hand_value, -1)
#
# # 保存注释通道
# cv2.imwrite('/home/shipu/mycode/see_hand/finger_count_channel.png', finger_count_channel * 255)
# cv2.imwrite('/home/shipu/mycode/see_hand/finger_joint_channel.png', finger_joint_channel * 85)  # 85 = 255 / 3
# cv2.imwrite('/home/shipu/mycode/see_hand/left_right_channel.png', left_right_channel * 127)  # 127 = 255 / 2


"""
方案二：
"""
import os

import cv2
import mediapipe as mp
import numpy as np

# 初始化Mediapipe手部检测器
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

dateset_path="/data/shipu/train_data_hdkp/videos"

video_list=os.listdir(dateset_path)
success_count=0

for video in video_list:
    video_dir_path=os.path.join(dateset_path,video)
    image_names=os.listdir(video_dir_path)
    frame_names=[i for i in image_names if "frame" in i]

    for frame_name in frame_names:
        image_path =os.path.join(video_dir_path,frame_name)

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取图像尺寸
        height, width, _ = image.shape

        # 创建黑色背景图像
        black_background = np.zeros((height, width, 3), dtype=np.uint8)

        # 绘制手部骨架（不同灰度值表示不同手指）
        def draw_hand_skeleton(image, hand_landmarks):
            # 定义每根手指的关键点范围
            finger_ranges = {
                "Thumb": [1, 2, 3, 4],
                "Index": [5, 6, 7, 8],
                "Middle": [9, 10, 11, 12],
                "Ring": [13, 14, 15, 16],
                "Pinky": [17, 18, 19, 20]
            }
            # 定义每根手指的灰度值
            finger_colors = {
                "Thumb": 50,
                "Index": 100,
                "Middle":150,
                "Ring": 200,
                "Pinky": 250
            }
            # 遍历每根手指
            for finger_name, finger_indices in finger_ranges.items():
                color = (finger_colors[finger_name],) * 3  # 将灰度值转换为BGR颜色
                # 遍历当前手指的连接
                for i in range(len(finger_indices) - 1):
                    start_idx = finger_indices[i]
                    end_idx = finger_indices[i + 1]
                    start_point = (int(hand_landmarks.landmark[start_idx].x * width), int(hand_landmarks.landmark[start_idx].y * height))
                    end_point = (int(hand_landmarks.landmark[end_idx].x * width), int(hand_landmarks.landmark[end_idx].y * height))
                    # 绘制线条
                    cv2.line(image, start_point, end_point, color, 2)

        # 标记指节（不同颜色表示不同指节）
        def draw_finger_joints(image, hand_landmarks):
            # 定义指节颜色
            joint_colors = {
                0: (255, 0, 0),   # 红色：指尖
                1: (0, 255, 0),   # 绿色：第一指节
                2: (0, 0, 255),   # 蓝色：第二指节
                3: (255, 255, 0)  # 黄色：第三指节
            }
            finger_joints = {
                0: 0,  # 手腕（手掌）
                1: 3,  # 拇指第三指节
                2: 2,  # 拇指第二指节
                3: 1,  # 拇指第一指节
                4: 0,  # 拇指指尖
                5: 3,  # 食指第三指节
                6: 2,  # 食指第二指节
                7: 1,  # 食指第一指节
                8: 0,  # 食指指尖
                9: 3,  # 中指第三指节
                10: 2,  # 中指第二指节
                11: 1,  # 中指第一指节
                12: 0,  # 中指指尖
                13: 3,  # 无名指第三指节
                14: 2,  # 无名指第二指节
                15: 1,  # 无名指第一指节
                16: 0,  # 无名指指尖
                17: 3,  # 小指第三指节
                18: 2,  # 小指第二指节
                19: 1,  # 小指第一指节
                20: 0  # 小指指尖
            }
            # 遍历关键点
            for i, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * width), int(landmark.y * height)
                # 确定指节类型
                if i == 0:  # 手掌关键点
                    color = (128, 128, 128)  # 灰色
                else:
                    joint_type = finger_joints.get(i, 0)
                    color = joint_colors[joint_type]
                # 绘制指节点
                cv2.circle(image, (x, y), 5, color, -1)

        # 处理图像
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 绘制手部骨架
                draw_hand_skeleton(black_background, hand_landmarks)
                # 标记指节
                draw_finger_joints(black_background, hand_landmarks)

        new_name=frame_name.replace("frame","hdkp")
        # 保存结果
        cv2.imwrite(os.path.join(video_dir_path,new_name), black_background)

    new_image_names=os.listdir(video_dir_path)
    a=[i for i in new_image_names if "hdkp" in i]
    b=len(frame_names)
    if (len(a)==b):
        print(video_dir_path,"success!")
        success_count+=1
        print("success_count:",success_count)



# 释放资源
hands.close()
