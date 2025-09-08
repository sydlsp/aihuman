import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def extract_num(filename):
    """
    从文件名中提取数字部分
    :param filename: 文件名字符串
    :return: 提取的数字
    """
    num_str = filename.split('.')[0].split('_')[-1]
    return int(num_str)


def batch_psnr_folder(real_dir, gen_dir, num=30):
    """
    计算两文件夹中对应图像的PSNR（文件名需相同）
    :param real_dir: 真实图像文件夹路径
    :param gen_dir: 生成图像文件夹路径
    :param num: 要计算的图像数量
    :return: 平均PSNR和每对图像的PSNR列表
    """
    real_files = sorted(os.listdir(real_dir), key=extract_num)[:num]
    gen_files = sorted(os.listdir(gen_dir), key=extract_num)[:num]
    assert len(real_files) == len(gen_files), "文件夹中图像数量不一致！"

    print("真实图像文件:", real_files)
    print("生成图像文件:", gen_files)

    psnr_scores = []
    for r_file, g_file in zip(real_files, gen_files):
        # 读取图像
        img1_path = os.path.join(real_dir, r_file)
        img2_path = os.path.join(gen_dir, g_file)

        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

        # 检查图像是否有效
        if img1 is None or img2 is None:
            print(f"警告：跳过无法读取的图像对 {r_file} 和 {g_file}")
            continue

        # 确保图像大小一致
        img1 = cv2.resize(img1, (512, 512), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (512, 512), interpolation=cv2.INTER_LINEAR)

        # 计算PSNR
        # mse = np.mean((img1 - img2) ** 2)
        # if mse == 0:
        #     psnr = float('inf')  # 完全一致时PSNR为无穷大
        # else:
        #     psnr = 10 * np.log10((255.0 ** 2) / mse)
        #
        # psnr_scores.append(psnr)
        # print(f"{r_file} vs {g_file}: PSNR = {psnr:.2f} dB")
        # 计算 PSNR（data_range 根据图像类型设置）
        psnr_value = psnr(img1, img2, data_range=255)  # 8-bit 图像用 255
        print(f"PSNR (scikit-image): {psnr_value:.2f} dB")
        psnr_scores.append(psnr_value)

    return np.mean(psnr_scores), psnr_scores


# 示例用法
if __name__ == "__main__":
    real_dir = "/data/shipu/video_153_high_quanlity_clip_1"
    gen_dir = "/home/shipu/mycode/output/save_153"
    mean_psnr, all_scores = batch_psnr_folder(real_dir, gen_dir)
    print(f"平均PSNR: {mean_psnr:.3f} dB")