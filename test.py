import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from PIL import Image
import torchvision.transforms as T
import os
import re

# 1. 定义图像预处理（转为Tensor + 归一化到[0,1]）
transform = T.Compose([
    T.Resize((512,512)),
    T.ToTensor(),  # 自动转换PIL.Image为[C, H, W]格式，范围[0,1]
])

# 2. 加载图像的函数
def load_image_as_tensor(image_path):
    img = Image.open(image_path).convert('RGB')  # 确保RGB格式
    return transform(img)  # 返回形状 [C, H, W], 范围 [0,1]

def natural_sort_key(s):
    # 提取文件名中的数字部分（适用于 "res_image_10.jpg" 格式）
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) if numbers else 0

def natural_sort_key_1(s):
    # 提取 "frame_0101" 中的数字部分（最后出现的连续数字）
    numbers = re.findall(r'frame_(\d+)', s)
    return int(numbers[-1]) if numbers else 0  # 转为整数排序

folder_path_1 = "/data/shipu/train_data_full/train_data_hdkp/videos/video_3_high_quanlity_clip_84"
folder_path_2="/home/shipu/mycode/output/save_126"

img_list_1=sorted([file for file in os.listdir(folder_path_1) if "frame" in file],key=natural_sort_key_1)
img_list_2=sorted(os.listdir(folder_path_2),key=natural_sort_key)

score=0
num=1
for i in range(0,num):

    # 3. 读取两张对比图像（替换为你的实际路径）
    img1_path = os.path.join(folder_path_1, img_list_1[i])
    img2_path = os.path.join(folder_path_2, img_list_2[i])

    # 加载图像
    img1 = load_image_as_tensor(img1_path).unsqueeze(0)  # 增加batch维度 -> [1, C, H, W]
    img2 = load_image_as_tensor(img2_path).unsqueeze(0)

    # 4. 检查图像尺寸是否一致
    assert img1.shape == img2.shape, f"图像尺寸不匹配: {img1.shape} vs {img2.shape}"

    # 5. 计算SSIM
    ssim = SSIM(data_range=1.0)  # 输入范围[0,1]
    score += ssim(img1, img2)
score /= num  # 平均SSIM分数
print(f"SSIM: {score.item():.4f}")  # .item()将张量转为Python float