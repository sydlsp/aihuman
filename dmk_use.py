from dataset_related.detect_mesh_and_keypoint import DetectMeshandKeypoint_1
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def sort_file_list(file_list):
    """
    排序函数，对文件夹进行排序
    """

    # 由于我们要排序的文件夹名是形如video_2的，所以我们要提取出2
    def extract_file_num(file_name):
        return int(file_name.split("_")[-1])
    file_list=sorted(file_list,key=extract_file_num)

    return file_list

data_dir_path="/remote-home/share/yfsong/shipu/video_src"  # 也就是原始视频的根目录

data_dir_part_path=os.path.join(data_dir_path,"part_video") # 也就是原始视频片段的文件夹
data_dir_part_list=os.listdir(data_dir_part_path) # 拿到了形如video_2等文件夹

data_dir_part_list=sort_file_list(data_dir_part_list)
print(data_dir_part_list)


dataset_dir_path="/remote-home/share/yfsong/shipu/dataset_video_true_1"  # 要创建文件夹的路径
dataset_dir_videos=os.path.join(dataset_dir_path,"videos") # 创建videos文件夹，用于存放每个视频片段的视频帧，mesh以及关键点的图片

os.makedirs(dataset_dir_videos,exist_ok=True)
os.makedirs(dataset_dir_videos,exist_ok=True)

dmk=DetectMeshandKeypoint_1()

print(data_dir_part_list[:5])

for dir in data_dir_part_list[:5]:

    print(dir)
    dir_path=os.path.join(data_dir_part_path,dir)

    seg_dir_path=os.path.join(dir_path,"segments") # 这里也就是拿到了形如/remote-home/share/yfsong/shipu/video_src/part_video/video_2/segments的文件路径
    if not os.path.exists(seg_dir_path):
        continue
    seg_dir_list=os.listdir(seg_dir_path)

    video_file=[]

    for file in seg_dir_list:
        if file.endswith(".mp4"):
            video_file.append(file) # video2 segments文件夹下的所有mp4文件

    for file in video_file:
        print(file)  #拿到了形如video_10_high_quanlity_clip_0.mp4的文件名
        file_name=file.split(".")[0]
        file_dir=os.path.join(dataset_dir_videos,file_name)
        os.makedirs(file_dir,exist_ok=True)
        # 这里要给视频路径以及输出文件夹路径
        dmk.get_mesh_and_keypoint_pic(os.path.join(seg_dir_path,file),file_dir)

        # 在这里我们补充以下，如果file_dir里面是空的，那么就移除这个文件夹
        if len(os.listdir(file_dir))==0:
            os.rmdir(file_dir)