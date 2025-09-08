import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointTransformer(nn.Module):
    def __init__(self,d_model,nhead=4):
        super().__init__()
        self.embedding=nn.Linear(2,d_model)
        self.transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead),num_layers=3)
        self.fc=nn.Linear(d_model,1)

    def forward(self,points):

        embedded=self.embedding(points)
        transformer_output=self.transformer(embedded)

        attention_weights=transformer_output

        importance_scores=torch.mean(attention_weights,dim=-1)
        importance_scores_norm=F.softmax(importance_scores,dim=0)
        return importance_scores

batch_size,squence_length,d_model=64,32,2
points=torch.randn(size=[batch_size,squence_length,d_model],dtype=torch.float32)
model=PointTransformer(d_model=64)

importance_scores=model(points)
print(importance_scores)
print(importance_scores.shape)

# 下面选择每行最大的前5个点的序号进行

topk_values,topk_indices=torch.topk(importance_scores,k=5,dim=-1)