import os

"""
这是修补工作，查清原因有些困难但是在脚本后补一块还是能解决掉的
"""
dir_path="/remote-home/share/yfsong/shipu/dataset_video_true/videos"

dir_list=os.listdir(dir_path)

for dir in dir_list:

    path=os.path.join(dir_path,dir)
    if len(os.listdir(path))<=0:
        print(path)
        # os.rmdir(path)
        # print("remove")