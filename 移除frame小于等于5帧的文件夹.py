import os
import shutil

"""
本代码是配合新建的Mydataset类写的
"""

data_dir="/data/shipu/dataset_true/videos"

video_dir=os.listdir(data_dir) # 这里拿到训练文件夹的名称

count=0
problem_count=0
for video in video_dir:
    count+=1
    print(count)

    video_file_path=os.path.join(data_dir,video)

    video_file=os.listdir(video_file_path)  # 拿到训练文件夹内每个文件夹的名称

    frame=[file for file in video_file if "frame" in file]  # 拿到所有的frame图片

    if (len(frame)<=5):
        shutil.rmtree(video_file_path)
        print("problem")
        problem_count+=1
        print(video)
        print("_______________________________________________________________________")


print("problem_count:",problem_count)

