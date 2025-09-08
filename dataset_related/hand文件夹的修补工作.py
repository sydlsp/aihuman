"""
hand文件夹的修补工作，让hand文件夹内的每个文件按顺序命名
"""
import os

path="/remote-home/share/yfsong/shipu/dataset_video_true_1/hands"

file_list=os.listdir(path)



file_list=sorted(file_list,key=lambda x:int(x.split(".")[0]))


# 修改一下文件名，使其按顺序
#
# for index,file_name in enumerate(file_list):
#     new_name=f"{index}.jpg"
#     os.rename(os.path.join(path,file_name),os.path.join(path,new_name))
#
# print("Complete!")

for i in range (len(file_list)):

    if i != int(file_list[i].split(".")[0]):
        print(i,file_list[i])
        print("________________________")

print("All match!")