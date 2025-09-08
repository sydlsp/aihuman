import cv2
import os
import mediapipe as mp
import shutil
# 初始化 MediaPipe 手部检测模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # 静态图片模式
    max_num_hands=2,         # 最多检测的手数
    min_detection_confidence=0.8  # 检测置信度
)
mp_draw = mp.solutions.drawing_utils  # 用于绘制关键点和连接线


videos_dir="/data/shipu/train_data/data/shipu/dataset_true/videos"

folders=os.listdir(videos_dir)
complete_count=0
correct_count=0
for fold in folders:
    video_path=os.path.join(videos_dir,fold)

    jpg_list=os.listdir(video_path)
    frame_list=[file for file in jpg_list if "frame" in file]

    for frame in frame_list:
        frame_path=os.path.join(video_path,frame)
        image = cv2.imread(frame_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 检测手部关键点
        results = hands.process(image_rgb)
        # 如果没有检测到手部
        if not results.multi_hand_landmarks:
            print("no hand detected")
            mesh_name=frame.replace("frame","mesh")
            keypoint_name=frame.replace("frame","keypoint")
            hands_name=frame.replace("frame","hands")
            os.remove(frame_path)
            os.remove(os.path.join(video_path,mesh_name))
            os.remove(os.path.join(video_path,keypoint_name))
            os.remove(os.path.join(video_path,hands_name))
        else:
            # 检测到手了
            flag=True
            for i,hand in enumerate(results.multi_hand_landmarks):
                hand_handedness = results.multi_handedness[i]  # 当前手的信息
                handedness_confidence = hand_handedness.classification[0].score  # 当前手的置信度
                if handedness_confidence<0.85:
                    flag=False
                    break
            if flag==False:
                print("hand detected but confidence is low")
                mesh_name = frame.replace("frame", "mesh")
                keypoint_name = frame.replace("frame", "keypoint")
                hands_name = frame.replace("frame", "hands")
                os.remove(frame_path)
                os.remove(os.path.join(video_path, mesh_name))
                os.remove(os.path.join(video_path, keypoint_name))
                os.remove(os.path.join(video_path, hands_name))


                # 判断移除完是不是对应的
    new_jpg_list = os.listdir(video_path)
    new_frame_list = [file for file in new_jpg_list if "frame" in file]
    new_mesh_list = [file for file in new_jpg_list if "mesh" in file]
    new_keypoint_list = [file for file in new_jpg_list if "keypoint" in file]
    new_hands_list = [file for file in new_jpg_list if "hands" in file]

    if (len(new_frame_list)==len(new_mesh_list)==len(new_keypoint_list)==len(new_hands_list)):
        print("OK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",fold,"is completed")
        correct_count+=1
    if (len(new_frame_list)<=5):
            shutil.rmtree(video_path)
    complete_count+=1
    print("complete_count:",complete_count)
    print("correct_count:", correct_count)


        # 加载图片
        # image_path = "/data/shipu/data/shipu/dataset_true/videos/video_103_high_quanlity_clip_42/video_103_high_quanlity_clip_42_frame_0013.jpg"  # 替换为你的图片路径
        # image = cv2.imread(image_path)
        #
        # # 检查图片是否加载成功
        # if image is None:
        #     print("无法加载图片，请检查路径是否正确！")
        #     exit()
        #
        # # 将图片从 BGR 转换为 RGB（MediaPipe 需要 RGB 格式）
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #
        # # 检测手部关键点
        # results = hands.process(image_rgb)
        #
        # # 如果检测到手部，绘制关键点和连接线
        # if results.multi_hand_landmarks:
        #     for hand_landmarks in results.multi_hand_landmarks:
        #         mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # else:
        #     print("未检测到手部！")
        # # 保存结果图片（可选）
        # output_path = "/home/shipu/hand_pic"  # 输出路径
        # cv2.imwrite(os.path.join(output_path,"10.jpg"), image)
        # print(f"检测结果已保存")