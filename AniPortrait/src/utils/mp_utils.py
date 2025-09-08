import os
import numpy as np
import cv2
import time
from tqdm import tqdm
import multiprocessing
import glob

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from . import face_landmark

# os.path.dirname(__file__)用来获取当前执行的脚本路径
CUR_DIR = os.path.dirname(__file__)


class LMKExtractor():
    def __init__(self, FPS=25):
        # Create an FaceLandmarker object.
        # 创建一个FaceLandmarker对象

        # mp.tasks.vision.FaceDetectorOptions.running_mode.IMAGE是mediapipe库中的枚举值
        # IMAGE表示处理单张图片，这意味着人脸检测器将在每个输入图像上独立运行，而不是在连续的视频帧上运行。
        self.mode = mp.tasks.vision.FaceDetectorOptions.running_mode.IMAGE

        # 创建 BaseOptions 类的实例
        # 用于设置模型的路径为：当前文件路径+mp_models/face_landmarker_v2_with_blendshapes.task，其实是用预训练的权重地址
        base_options = python.BaseOptions(model_asset_path=os.path.join(CUR_DIR, 'mp_models/face_landmarker_v2_with_blendshapes.task'))

        # 设置base_options的计算应该在cpu上执行

        base_options.delegate = mp.tasks.BaseOptions.Delegate.CPU

        # 创建 FaceLandmarkerOptions 类的实例，这个类用于进行面部标记，主要作用是在检测到的人脸上找到特定的面部标记点
        # 用于设置基本选项为base_options，运行模式为self.mode，输出面部混合形状为True，输出面部变换矩阵为True，人脸数为1
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            running_mode=self.mode,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)

        # 根据options创建FaceLandmarker对象
        self.detector = face_landmark.FaceLandmarker.create_from_options(options)

        # 将最后一次的时间戳初始值设置为0
        self.last_ts = 0

        # 将每帧的时间间隔设置为1000/FPS
        self.frame_ms = int(1000 / FPS)

       # 同样的，按照上面的流程：设置base_options、设置模型options、按照模型options创建FaceDetector对象
       # FaceDetector用来进行人脸检测
        det_base_options = python.BaseOptions(model_asset_path=os.path.join(CUR_DIR, 'mp_models/blaze_face_short_range.tflite'))
        det_options = vision.FaceDetectorOptions(base_options=det_base_options)
        self.det_detector = vision.FaceDetector.create_from_options(det_options)
                

    # __call__像调用函数一样调用类
    def __call__(self, img):

        # 将opencv读取出来的BGR格式转换为RGB格式
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 将转换完的RGB格式的图片封装为MediaPipe的Image对象，以便后续的处理
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        t0 = time.time()


        # 先理解一下处理模式为VIDEO和IMAGE的区别：
        # VIDEO：处理连续的视频帧，人脸检测器FaceDetector会在连续的视频帧上运行(也就是说要提供时间戳)
        # IMAGE：处理单张图片，人脸检测器FaceDetector会在每个输入图像上独立运行，不需要提供时间戳

        # 如果处理模式是VIDEO
        if self.mode == mp.tasks.vision.FaceDetectorOptions.running_mode.VIDEO:

            # 调用人脸检测器FaceDetector对图片进行人脸检测
            det_result = self.det_detector.detect(image)
            # 如果检测到的人脸数不等于1，返回None，直接结束
            if len(det_result.detections) != 1:
                return None
            # 如果检测到的人脸数等于1，调用面部标记FaceLandmarker对象的detect_for_video方法对图片进行处理
            self.last_ts += self.frame_ms

            # 得到面部标记结果和3D面格网格数据
            try:
                detection_result, mesh3d = self.detector.detect_for_video(image, timestamp_ms=self.last_ts)
            except:
                return None
        # 如果处理模式是IMAGE，和上面的操作类似
        elif self.mode == mp.tasks.vision.FaceDetectorOptions.running_mode.IMAGE:
            # det_result = self.det_detector.detect(image)

            # if len(det_result.detections) != 1:
            #     return None
            try:
                detection_result, mesh3d = self.detector.detect(image)
            except:
                return None
            
        # detection_result包含face_landmarks,face_blendshapes, facial_transformation_matrixes等信息
        # face_landmarks是一个列表，包含面部标记点的坐标，每个坐标是一个x,y,z对象
        # face_blendshapes是一个列表，包含面部混合形状的分数
        # facial_transformation_matrixes是一个列表，包含面部变换矩阵，面部变换矩阵是一个3x4的矩阵，用于描述面部在3D空间中的位置和方向
        bs_list = detection_result.face_blendshapes

        # 如果检测到的人脸数为1，这样写是因为以IMAGE方式处理没有强制检测不是一张人脸就返回None
        if len(bs_list) == 1:
            bs = bs_list[0]
            bs_values = []

            # 遍历一张脸上的所有面部混合形状的分数，将分数添加到bs_values列表中
            for index in range(len(bs)):
                bs_values.append(bs[index].score)
            # 在列表中去掉neutral表情得分
            bs_values = bs_values[1:] # remove neutral

            # 获取面部变换矩阵
            trans_mat = detection_result.facial_transformation_matrixes[0]

            # 获取面部标记点列表
            face_landmarks_list = detection_result.face_landmarks
            # 拿到第一个人脸的面部标记点
            face_landmarks = face_landmarks_list[0]

            # 将面部标记点[x,y,z]坐标添加到列表中
            lmks = []
            for index in range(len(face_landmarks)):
                x = face_landmarks[index].x
                y = face_landmarks[index].y
                z = face_landmarks[index].z
                lmks.append([x, y, z])
            lmks = np.array(lmks)

            # 提取mesh3d网格的顶点数据
            lmks3d = np.array(mesh3d.vertex_buffer)
            # 在这里可能一个顶点包含5个信息分别是x,y,z,u,v
            # 只需要x,y,z前三个信息，所以取前三个信息
            lmks3d = lmks3d.reshape(-1, 5)[:, :3]

            # 每个索引表示一个顶点在
            # mesh3d.vertex_buffer中的位置。在3D模型中，每个面（也称为三角形）通常由3个顶点组成
            # 所以mesh3d.index_buffer中的每三个索引表示一个面的3个顶点的索引。  例如，如果
            # 例如，如果mesh3d.index_buffer是[0, 1, 2, 2, 3, 0]，那么它表示的是两个面，第一个面的顶点索引是0、1、2，第二个面的顶点索引是2、3、0。
            # 这些索引对应mesh3d.vertex_buffer中的顶点数据。
            # +1是因为在一些图形库中索引是从1开始的
            mp_tris = np.array(mesh3d.index_buffer).reshape(-1, 3) + 1

            # 返回字典
            return {
                "lmks": lmks,
                'lmks3d': lmks3d,
                "trans_mat": trans_mat,
                'faces': mp_tris,
                "bs": bs_values
            }
        else:
            # print('multiple faces in the image: {}'.format(img_path))
            return None
        