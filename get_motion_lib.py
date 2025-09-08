import os
import shutil
from tqdm import tqdm
dataset_path="/data/shipu/train_data_full/train_data_hdkp/videos"

motion_lib_path="/data/shipu/motion_lib"

motion_folder=os.listdir(dataset_path)

for folder in tqdm(motion_folder,desc="Processing folders"):

    print(folder)
    folder_path=os.path.join(dataset_path,folder)

    # 身体姿态关键点
    keypoint_list=[file for file in os.listdir(folder_path) if "keypoint" in file]
    # 手部的关键点
    hdkp_list=[file for file in os.listdir(folder_path) if "_hdkp_" in file]

    full_list=[file for file in os.listdir(folder_path) if "full" in file]

    # 新建文件夹
    new_folder=os.path.join(motion_lib_path,folder)
    os.makedirs(new_folder,exist_ok=True)

    # 复制文件
    for file in keypoint_list:
        file_path=os.path.join(folder_path,file)
        shutil.copy(file_path,new_folder)

    for file in hdkp_list:
        file_path=os.path.join(folder_path,file)
        shutil.copy(file_path,new_folder)

    for file in full_list:
        file_path=os.path.join(folder_path,file)
        shutil.copy(file_path,new_folder)
