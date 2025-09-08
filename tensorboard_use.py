import os.path
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
dir_path="/remote-home/share/yfsong/shipu/"
log_dir=os.path.join(dir_path,"logdir")


# 使用 torch.linspace 生成 0 到 2π 的值
x = torch.linspace(0, 2 * np.pi, steps=10000)
y = torch.sin(x)

print(x.shape)

print(len(x))
writer=SummaryWriter(log_dir=log_dir)

for i in range(len(x)):
    writer.add_scalar("y=sin(x)",y[i].item(),x[i].item())
    print(y[i].item(),x[i].item())

writer.close()