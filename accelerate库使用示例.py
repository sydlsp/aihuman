from accelerate import Accelerator
import torch
import argparse
# /root/.cache/huggingface/accelerate/default_config.yaml

accelerator=Accelerator()

# 获取可用的gpu数量
if torch.cuda.is_available():
    gpu_nums=torch.cuda.device_count()

print(gpu_nums)

devices=accelerator.state.num_processes
print(devices)

def parse_args():
    parse=argparse.ArgumentParser(description="This is a test")
    parse.add_argument("--num",type=int,default=1,help="This is a number")

    args=parse.parse_args()
    return args

args=parse_args()
print("___________________________________________")
print(args.num)
# accelerate launch --config_file /remote-home/yfsong/shipu/mycode/default_config.yaml /remote-home/yfsong/shipu/mycode/test.py --num 100
# accelerate launch