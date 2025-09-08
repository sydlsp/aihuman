import math
import os
import shutil
import torch
import cv2
import moviepy.editor as mp
import concurrent.futures
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


def detect_scenes(video_path, threshold=30.0, min_scene_len=15):
    """
    检测视频中的场景变化
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))

    video_manager.set_downscale_factor(1)
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()

    return scene_list


def remove_transitions(video_path, output_path, transition_margin=1.0):
    """
    移除视频中的转场区域并生成新的视频
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    last_name = os.path.basename(output_path)
    parent = os.path.dirname(output_path)
    second_last_name = os.path.basename(parent)

    scenes = detect_scenes(video_path)
    video = mp.VideoFileClip(video_path)
    clips = []

    for i, scene in enumerate(scenes):
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()

        if i > 0:
            start += transition_margin
        if i < len(scenes) - 1:
            end -= transition_margin

        if start < end:
            clips.append(video.subclip(start, end))

    for i, video_clip in enumerate(clips):
        clip_output_path = os.path.join(output_path, f"{second_last_name}_{last_name}_video_{i}.mp4")
        video_clip.write_videofile(clip_output_path, codec="libx264", fps=video_clip.fps)


def split_video(video_path, output_path, segment_duration=5.0):
    """
    将视频切割成更小的片段
    """
    videos_file_name = [video_file for video_file in os.listdir(video_path) if
                        os.path.isfile(os.path.join(video_path, video_file))]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    last_name = os.path.basename(video_path)
    parent = os.path.dirname(video_path)
    second_last_name = os.path.basename(parent)

    i = 0
    for video_file_name in videos_file_name:
        video_file_path = os.path.join(video_path, video_file_name)
        video_file = mp.VideoFileClip(video_file_path)
        video_file_duration = video_file.duration

        for start_time in range(0, int(video_file_duration), int(segment_duration)):
            end_time = start_time + segment_duration
            if end_time > video_file_duration:
                break
            else:
                segment = video_file.subclip(start_time, end_time)
                segment_audio = segment.audio
                segment_path = os.path.join(output_path, f"{second_last_name}_{last_name}_clip_{i}.mp4")
                segment_audio_path = os.path.join(output_path, f"{second_last_name}_{last_name}_clip_{i}.m4a")
                i += 1
                segment.write_videofile(segment_path, codec="libx264", fps=video_file.fps)
                segment_audio.write_audiofile(segment_audio_path, codec="aac")


def detect_person_in_video(model, video_path, interval=2, person_threshold=0.3, time_threshold=0.8):
    """
    检测视频中的人物并判断是否符合条件
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    calculate_frame_count = 0
    correct_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            calculate_frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            persons = [x for x in results.xyxy[0].cpu().numpy() if int(x[5] == 0)]
            frame_area = frame.shape[0] * frame.shape[1]
            person_area_sum = sum((x[2] - x[0]) * (x[3] - x[1]) for x in persons)
            person_ratio = person_area_sum / frame_area
            if person_ratio > person_threshold:
                correct_frame_count += 1
        frame_count += 1

    cap.release()
    if calculate_frame_count == 0:
        return False
    if correct_frame_count / calculate_frame_count >= time_threshold:
        return True
    else:
        return False


def process_video_folder(model, in_dirs, out_dirs, in_dirs_subfolder):
    """
    处理单个视频文件夹，去除转场并分割视频
    """
    subfolder_path = os.path.join(in_dirs, in_dirs_subfolder)
    in_dirs_subfolder_filelist = os.listdir(subfolder_path)

    if not f"{in_dirs_subfolder}_combine.mp4" in in_dirs_subfolder_filelist:
        return

    video_path = os.path.join(subfolder_path, f"{in_dirs_subfolder}_combine.mp4")
    out_video_path = os.path.join(out_dirs, in_dirs_subfolder)

    no_transition_video_path = os.path.join(out_video_path, "no_transitions")
    if not os.path.exists(no_transition_video_path):
        os.makedirs(no_transition_video_path)

    scene_list = detect_scenes(video_path)
    remove_transitions(video_path, no_transition_video_path)

    high_quality_video_folder = os.path.join(out_video_path, "high_quality")
    if not os.path.exists(high_quality_video_folder):
        os.makedirs(high_quality_video_folder)

    no_transition_video_list = os.listdir(no_transition_video_path)
    for no_transition_video in no_transition_video_list:
        need_check_video_path = os.path.join(no_transition_video_path, no_transition_video)
        if detect_person_in_video(model, need_check_video_path):
            high_quality_video_path = os.path.join(high_quality_video_folder, no_transition_video)
            shutil.copy2(need_check_video_path, high_quality_video_path)
            print(f"已将 {need_check_video_path} 复制到 {high_quality_video_path}")

    segments_path = os.path.join(out_video_path, "segments")
    if not os.path.exists(segments_path):
        os.makedirs(segments_path)
    split_video(high_quality_video_folder, segments_path)


def process_videos_in_parallel(model, in_dirs, out_dirs, thread_num=2):
    """
    使用多线程处理视频文件夹
    """
    in_dirs_subfolders = [folder for folder in os.listdir(in_dirs) if os.path.isdir(os.path.join(in_dirs, folder))]
    folder_num = len(in_dirs_subfolders)

    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = []
        for subfolder in in_dirs_subfolders:
            futures.append(
                executor.submit(process_video_folder, model, in_dirs, out_dirs, subfolder)
            )
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing folder: {e}")


if __name__ == '__main__':
    in_dirs = "/remote-home/share/yfsong/shipu/new_video_src/full_video"
    out_dirs = "/remote-home/share/yfsong/shipu/new_video_src/part_video"
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    process_videos_in_parallel(model, in_dirs, out_dirs, thread_num=2)

    print("end")
