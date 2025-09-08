import argparse
from omegaconf import OmegaConf
from dataset_related.dataset_write_1 import MyDataset,collate_fn
import torch

# 加载config文件
parser=argparse.ArgumentParser()
parser.add_argument("--config",type=str,default="/remote-home/yfsong/shipu/mycode/train_config/train_1.yaml")
args=parser.parse_args()

if args.config[-5:] == ".yaml":
    config = OmegaConf.load(args.config)
else:
    raise ValueError("Do not support this format config file")


train_dataset=MyDataset(**config.data,is_image=True)

train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=config.train_bs,shuffle=True,num_workers=1,drop_last=True,collate_fn=collate_fn,)

for i,batch in enumerate(train_dataloader):
    if i==1:
        break