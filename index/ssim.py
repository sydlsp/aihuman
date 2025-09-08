import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def extract_num(filename):
    """
    从文件名中提取数字部分
    :param filename: 文件名字符串
    :return: 提取的数字
    """

    num_str=filename.split('.')[0].split('_')[-1]
    return int(num_str)



def batch_ssim_folder(real_dir, gen_dir,num=30):
    """
    计算两文件夹中对应图像的SSIM（文件名需相同）
    :param real_dir: 真实图像文件夹路径
    :param gen_dir: 生成图像文件夹路径
    :return: 平均SSIM和每对图像的SSIM列表
    """
    real_files = sorted(os.listdir(real_dir),key=extract_num)[:num]
    gen_files = sorted(os.listdir(gen_dir),key=extract_num)[:num]
    assert len(real_files) == len(gen_files), "文件夹中图像数量不一致！"

    print(real_files)
    print(gen_files)

    ssim_scores = []
    for r_file, g_file in zip(real_files, gen_files):
        # 读取图像

        print(os.path.join(real_dir, r_file))
        img1 = cv2.imread(os.path.join(real_dir, r_file), cv2.IMREAD_COLOR)
        img2 = cv2.imread(os.path.join(gen_dir, g_file), cv2.IMREAD_COLOR)
        # 检查图像是否有效
        if img1 is None or img2 is None:
            print(f"警告：跳过无法读取的图像对 {r_file} 和 {g_file}")
            continue

        # 确保图像大小一致
        img1 = cv2.resize(img1, (512, 512), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (512, 512), interpolation=cv2.INTER_LINEAR)

        # 归一化并计算SSIM
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        score = ssim(img1, img2, channel_axis=2, data_range=1.0)
        ssim_scores.append(score)
        print(f"{r_file} vs {g_file}: SSIM = {score:.4f}")

    return np.mean(ssim_scores), ssim_scores


# 示例用法
if __name__ == "__main__":
    real_dir = "/data/shipu/video_153_high_quanlity_clip_1"
    gen_dir = "/home/shipu/mycode/output/save_153"
    mean_ssim, all_scores = batch_ssim_folder(real_dir, gen_dir)
    print(f"平均SSIM: {mean_ssim:.4f}")