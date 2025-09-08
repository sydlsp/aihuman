import torch
from pytorch_fvd import get_fvd_logits, frechet_distance

# 假设真实视频和生成视频均为 [N, T, C, H, W] 张量（范围[0,1]）
real_videos = torch.rand(10, 16, 3, 224, 224)  # 10个真实视频，每段16帧
fake_videos = torch.rand(10, 16, 3, 224, 224)  # 10个生成视频

# 提取I3D特征
real_logits = get_fvd_logits(real_videos)
fake_logits = get_fvd_logits(fake_videos)

# 计算FVD
fvd = frechet_distance(real_logits, fake_logits)
print(f"FVD: {fvd:.2f}")