# 使用ani运行
# 首先在系统中安装ffmpeg
# 安装PySceneDetect和moviepy    pip install scenedetect[opencv] moviepy
# 记住：python中的相对路径是根据文件运行时位置确定的，也就是基于运行python脚本时所在的当前工作目录确定的

"""
  使用PySceneDetect检测视频中的场景变化
"""


import math
import os.path
import shutil

import torch
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

import cv2

import threading

def detect_scenes(video_path,threshold=30.0,min_scene_len=15):

    """
    检测视频中的场景变化
    :param threshold:  场景变化的阈值
    :param min_scene_len: 最小场景长度(以帧为单位)
    :return: 场景列表，每个场景为(start_time,end_time)
    """

    # 创建对象，用于管理视频文件的读取和处理
    video_manager=VideoManager([video_path])

    scene_manager=SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold,min_scene_len=min_scene_len))

    video_manager.set_downscale_factor(1)  # 设置下采样因子为1，不进行下采样
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list=scene_manager.get_scene_list()
    video_manager.release()

    return scene_list


"""
移除转场区域并生成新视频
"""
import moviepy.editor as mp

def remove_transitions(video_path,output_path,transition_margin=1.0):
    """
    移除视频中的转场区域
    这里的video_path和上面函数需要的video_path是一样的，直接定位到视频的路径
    这里我们规定一下output_path是到文件夹的路径，我们在这个文件下存放无转场的视频的片段
    :param transition_margin: 每个场景的开头和结尾要移除的时间(秒)
    :return:
    """

    # 先处理一下路径
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 把无转场视频片段的文件夹名拿出来
    last_name=os.path.basename(output_path)

    # 再把上一级文件夹拿出来
    parent=os.path.dirname(output_path)
    second_last_name=os.path.basename(parent)

    # 检测场景
    scenes=detect_scenes(video_path)

    # 加载视频
    video=mp.VideoFileClip(video_path)

    # 定义要保留的片段
    clips=[]
    for i,scene in enumerate(scenes):

        start=scene[0].get_seconds()
        end=scene[1].get_seconds()

        # 移除转场边缘
        if i>0:
            start+=transition_margin
        if i<len(scenes)-1:
            end-=transition_margin

        # 这里start<end的才能保留
        if start<end:
            clips.append(video.subclip(start,end))

    for i,video in enumerate(clips):
        clip_output_path=os.path.join(output_path,f"{second_last_name}_{last_name}_video_{i}.mp4")
        video.write_videofile(clip_output_path,codec="libx264",fps=video.fps)

def split_video(video_path,output_path,segment_duration=5.0):
    """
    将切出来的转场视频再切分成一个一个小段
    :param video_path: 在这里针对video_path是非转场视频所在的文件夹
    :param output_path: 输出路径是在video_path下新建一个子文件夹用来存放切分好的一个一个片段
    :param segment_duration: 每小段视频的时间
    :return:
    """
    # 把文件过滤出来不要文件夹
    videos_file_name=[video_file for video_file in os.listdir(video_path) if os.path.isfile(os.path.join(video_path,video_file))]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 把非转场视频所在的文件夹名字拿出来
    last_name=os.path.basename(video_path)

    # 把非转场视频倒数第二级文件夹名字拿出来
    parent=os.path.dirname(video_path)
    second_last_name=os.path.basename(parent)

    i=0

    for video_file_name in videos_file_name:
        video_file_path=os.path.join(video_path,video_file_name)
        print("video_file_path",video_file_path)
        # 加载视频
        video_file=mp.VideoFileClip(video_file_path)
        # 获取视频的总时长
        video_file_duration=video_file.duration

        for start_time in range(0,int(video_file_duration),int(segment_duration)):
            end_time=start_time+segment_duration
            # 不满时长的片段不要
            if end_time>video_file_duration:
                break
            else:
                segment=video_file.subclip(start_time,end_time)
                segment_audio=segment.audio
                segment_path=os.path.join(output_path,f"{second_last_name}_{last_name}_clip_{i}.mp4")
                segment_audio_path=os.path.join(output_path,f"{second_last_name}_{last_name}_clip_{i}.m4a")
                i=i+1
                segment.write_videofile(segment_path,codec="libx264",fps=video_file.fps)
                segment_audio.write_audiofile(segment_audio_path,codec="aac") # 这里修改了下编码器看能不能用

def detect_person_in_video(model,video_path,output_path=None,interval=2,person_threshold=0.3,time_threshold=0.8):
    """

    :param model: yolov5s
    :param video_path: 要检测的视频本身路径
    :param output_path: 可视化结果的视频输出路径
    :param interval: 间隔多少帧对视频进行检测
    :param person_threshold: 人物在视频帧中的比重
    :param time_threshold: 符合条件的视频帧占视频时长的比重
    :return True or False
    """

    # 打开视频文件
    cap=cv2.VideoCapture(video_path)

    # 如果需要保存输出视频
    if output_path:
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        out=cv2.VideoWriter(output_path,fourcc,cap.get(cv2.CAP_PROP_FPS),
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count=0 # 总的帧数
    calculate_frame_count=0 # 参与计算的帧数
    correct_frame_count=0 # 符合条件的帧数

    while cap.isOpened():

        ret,frame=cap.read() # ret表示是否读取帧成功，frame表示读取到的帧
        if not ret:
            break
        if frame_count % interval ==0:

            calculate_frame_count+=1
            frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 使用模型进行检测
            results=model(frame_rgb)

            # results中包含了检测结果的边界框信息，置信度分数，对象的类别标签信息
            persons=[x for x in results.xyxy[0].cpu().numpy() if int(x[5]==0)]  # 类别 0 是person

            # 计算每帧中的人物面积
            frame_area=frame.shape[0]*frame.shape[1]
            person_area_sum=sum((x[2]-x[0])*(x[3]-x[1]) for x in persons)

            # 计算占比
            person_ratio=person_area_sum/frame_area
            if person_ratio > person_threshold :
                correct_frame_count+=1


            # 显示或保存处理后的帧
            if output_path:

                # 在帧上绘制边界框
                results.render()

                frame_bgr=cv2.cvtColor(results.ims[0],cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        frame_count+=1

    cap.release()
    # 如果满足条件的帧时间占比超过一定比例，则认为这个帧是符合条件的
    if  correct_frame_count/calculate_frame_count >= time_threshold:
        return True
    else:
        return False



import argparse
def parse_args():
    parse=argparse.ArgumentParser(description="视频转场处理")

    parse.add_argument("--in_dirs",type=str,help="输入视频文件夹路径",default="/data/shipu/video_src_new/full_video")
    parse.add_argument("--out_dirs",type=str,help="输出视频文件夹路径",default="/data/shipu/video_src_new/part_video")
    parse.add_argument("--interval",type=int,help="检测帧的间隔",default=4)
    parse.add_argument("--person_threshold",type=float,help="人物面积占视频帧的比例",default=0.25)
    parse.add_argument("--time_threshold",type=float,help="满足条件的视频帧所占的时长比例",default=0.8)
    parse.add_argument("--thread_num",type=int,help="线程数",default=2)

    args=parse.parse_args()
    return args

def processing(in_dirs,out_dirs,start,end,device):
    print(f"Loading model on {device}")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to(device)
    model.eval()  # 设置为评估模式

    # 拿到in_dirs下的所有文件夹，也就是video_rr2等文件夹
    in_dirs_subfolders = [folder for folder in os.listdir(in_dirs) if os.path.isdir(os.path.join(in_dirs, folder))][start:end]
    print(in_dirs_subfolders)

    for in_dirs_subfolder in in_dirs_subfolders:
        in_dirs_subfolder_filelist = os.listdir(os.path.join(in_dirs, in_dirs_subfolder))

        # 查找是否有 {in_dirs_subfolder}_combine.mp4文件
        if not f"{in_dirs_subfolder}_combine.mp4" in in_dirs_subfolder_filelist:
            continue

        video_path = os.path.join(in_dirs, in_dirs_subfolder, f"{in_dirs_subfolder}_combine.mp4")

        out_video_path = os.path.join(out_dirs, in_dirs_subfolder)  # /video_src/part_video/video_rr2/

        # 接下来创建文件夹，用来存放不包含转场的视频片段 /video_src/part_video/video_rr2/no_transitions
        no_transition_video_path = os.path.join(out_video_path, "no_transitions")

        if not os.path.exists(no_transition_video_path):
            os.makedirs(no_transition_video_path)

        scene_list = detect_scenes(video_path)
        remove_transitions(video_path, no_transition_video_path)

        # 针对无转场的视频，对其进行检测，如果满足条件就将其保存到高质量文件夹 /video_src/part_video/video_rr2/high_quanlity
        high_quanlity_video_folder = os.path.join(out_video_path, "high_quanlity")

        if not os.path.exists(high_quanlity_video_folder):
            os.makedirs(high_quanlity_video_folder)

        no_transition_video_list = os.listdir(no_transition_video_path)

        # 对无转场的视频进行遍历，如果满足条件就放到高质量的文件夹下
        for no_transition_video in no_transition_video_list:
            need_check_video_path = os.path.join(no_transition_video_path, no_transition_video)
            if detect_person_in_video(model, need_check_video_path, interval=args.interval,
                                      person_threshold=args.person_threshold, time_threshold=args.time_threshold):
                high_quanlity_video_path = os.path.join(high_quanlity_video_folder, no_transition_video)
                shutil.copy2(need_check_video_path, high_quanlity_video_path)
                print(f"已将 {need_check_video_path} 复制到 {high_quanlity_video_path}")

        # 接下来再创建文件夹，用来存放切分好的视频片段 /video_src/part_video/video_rr2/segments
        segments_path = os.path.join(out_video_path, "segments")
        if not os.path.exists(segments_path):
            os.makedirs(segments_path)


        split_video(high_quanlity_video_folder, segments_path)

        # 删除high_quanlity 的所有文件
        shutil.rmtree(high_quanlity_video_folder)
        shutil.rmtree(no_transition_video_path)


if __name__=='__main__':

    args=parse_args()

    in_dirs=args.in_dirs
    out_dirs=args.out_dirs
    if not os.path.exists(out_dirs):
        os.makedirs(out_dirs)


    thread_num=args.thread_num

    folder_num=len([folder for folder in os.listdir(in_dirs) if os.path.isdir(os.path.join(in_dirs,folder))])

    part_num=math.ceil(folder_num/thread_num)

    threads=[]
    device_list=["cuda:0","cuda:1","cuda:2","cuda:3"]
    for i in range(thread_num):
        start=i*part_num
        end=min((i+1)*part_num,folder_num)
        t=threading.Thread(target=processing,args=(in_dirs,out_dirs,start,end,device_list[i]))
        threads.append(t)
        t.start()

    for thread in threads:
        thread.join() # 等待所有线程执行完毕


    print("end")



    # # 拿到in_dirs下的所有文件夹，也就是video_rr2等文件夹
    # in_dirs_subfolders=[folder for folder in os.listdir(in_dirs) if os.path.isdir(os.path.join(in_dirs,folder))]
    # print(in_dirs_subfolders)
    #
    # for in_dirs_subfolder in in_dirs_subfolders:
    #     in_dirs_subfolder_filelist=os.listdir(os.path.join(in_dirs,in_dirs_subfolder))
    #
    #     # 查找是否有 {in_dirs_subfolder}.mp4文件
    #     if not f"{in_dirs_subfolder}.mp4" in  in_dirs_subfolder_filelist:
    #         continue
    #
    #     video_path=os.path.join(in_dirs,in_dirs_subfolder,f"{in_dirs_subfolder}.mp4")
    #
    #     out_video_path=os.path.join(out_dirs,in_dirs_subfolder) # /video_src/part_video/video_rr2/
    #
    #     # 接下来创建文件夹，用来存放不包含转场的视频片段 /video_src/part_video/video_rr2/no_transitions
    #     no_transition_video_path=os.path.join(out_video_path,"no_transitions")
    #
    #     if not os.path.exists(no_transition_video_path):
    #         os.makedirs(no_transition_video_path)
    #
    #     scene_list=detect_scenes(video_path)
    #     remove_transitions(video_path,no_transition_video_path)
    #
    #     # 针对无转场的视频，对其进行检测，如果满足条件就将其保存到高质量文件夹 /video_src/part_video/video_rr2/high_quanlity
    #     high_quanlity_video_folder=os.path.join(out_video_path,"high_quanlity")
    #
    #     if not os.path.exists(high_quanlity_video_folder):
    #         os.makedirs(high_quanlity_video_folder)
    #
    #     no_transition_video_list=os.listdir(no_transition_video_path)
    #
    #     # 对无转场的视频进行遍历，如果满足条件就放到高质量的文件夹下
    #     for no_transition_video in no_transition_video_list:
    #         need_check_video_path=os.path.join(no_transition_video_path,no_transition_video)
    #         if detect_person_in_video(model,need_check_video_path,interval=args.interval,person_threshold=args.person_threshold,time_threshold=args.time_threshold):
    #             high_quanlity_video_path=os.path.join(high_quanlity_video_folder,no_transition_video)
    #             shutil.copy2(need_check_video_path,high_quanlity_video_path)
    #             print(f"已将 {need_check_video_path} 复制到 {high_quanlity_video_path}")
    #
    #     # 接下来再创建文件夹，用来存放切分好的视频片段 /video_src/part_video/video_rr2/segments
    #     segments_path=os.path.join(out_video_path,"segments")
    #     if not os.path.exists(segments_path):
    #         os.makedirs(segments_path)
    #     split_video(high_quanlity_video_folder,segments_path)












