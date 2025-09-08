import cv2
import os
def extract_number(filename):
    return int(filename.split('.')[0].split('_')[-1])
# 图片文件夹路径
image_folder = '/home/shipu/mycode/output/save_41'
# 输出视频文件名
video_name = 'output_video.mp4'

# 获取图片列表
images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
images=sorted(images,key=extract_number)
print(images)
# 读取第一张图片，获取尺寸
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 设置视频编码器和帧率
fps = 4  # 帧率
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# 将图片写入视频
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# 释放资源
video.release()
print(f"视频已生成: {video_name}")