import torch

a=torch.randn(size=[1,2,3])

b=torch.randn(size=[2,2,4])

c=torch.cat([a,b],dim=0)

print(c.shape)