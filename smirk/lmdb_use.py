import lmdb

# 创建或打开lmdb环境
env=lmdb.open("/remote-home/share/yfsong/shipu/dataset_video.lmdb",map_size=1e10) # 10G

print("ok")

# 将指定文件夹下的