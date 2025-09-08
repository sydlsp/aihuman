import os

path="/remote-home/share/yfsong/shipu/dataset_video/video_10_high_quanlity_clip_10"

file_list=os.listdir(path)

file_frame_list=[file for file in file_list if "frame" in file]

print(len(file_frame_list))

def extract_file_num(file_name):
    return int(file_name.split(".")[0].split("_")[-1])

file_frame_list=sorted(file_frame_list,key=extract_file_num)

print(file_frame_list)
