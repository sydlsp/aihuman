"""
本代码用来将文件夹中的视频文件放到full_video文件夹中并重命名
"""
import os
import shutil




root_dir="G:\\video_src_new"  # 这里用的时候要改

full_video_path=os.path.join(root_dir,"full_video")

os.makedirs(full_video_path,exist_ok=True)

video_file_list=[i for i in os.listdir(root_dir) if i.endswith(".mp4")]

print(video_file_list)
print(len(video_file_list))


start_idx=1 # 这里用的时候要改

for video_file in video_file_list:
    video_dir_path=os.path.join(full_video_path,"video_"+str(start_idx))
    os.makedirs(video_dir_path,exist_ok=True)

    print(os.path.join(root_dir,video_file))
    # 移动文件
    shutil.move(os.path.join(root_dir,video_file),os.path.join(video_dir_path,"video_"+str(start_idx)+"_combine"+".mp4"))
    start_idx+=1
    print("yes")

print("ok")