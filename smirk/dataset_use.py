import torch.utils.data

from dataset_write import MyDataset

train_dataset=MyDataset("/remote-home/share/yfsong/shipu/dataset_video")
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True,drop_last=True)

train_dataloader=iter(train_dataloader)

first_loader=next(train_dataloader)

for key in first_loader.keys():
    print(key,first_loader[key].shape)