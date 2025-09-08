from pytorch_fid import fid_score
import torch

# 设置路径和参数
real_path = "/data/shipu/video_153_high_quanlity_clip_1"
gen_path = "/home/shipu/mycode/output/save_153"
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

# 计算FID
fid_value = fid_score.calculate_fid_given_paths(
    [real_path, gen_path],
    batch_size=batch_size,
    device=device,
    dims=2048,
    num_workers=0# Inception-v3特征维度
)
print(f"FID Score: {fid_value:.2f}")