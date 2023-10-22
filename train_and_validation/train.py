import torch
import os
from torch import nn
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
# 配置训练环境和超参数
# 配置GPU，这里有两种方式
## 方案一：使用os.environ
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
## 方案二：使用device，后续对要使用GPU的变量使用to(device)即可
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#MBSGD 小批量梯度下降 , 这个train_loader里面是有多个batch的
from train_and_validation.validation import validation
import torch.nn.functional as F


def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    train_acc= 0.0
    i=0

    for id, (data, target) in enumerate(train_loader):
        # if id==0:
        #     print("测试集：",data[0]) #这边同样DPSGD的验证集也是浮点型的
        data, target = data.to(device), target.to(device)
        output = model(data.to(torch.float32))  # 计算输出
        loss = F.cross_entropy(output, target.to(torch.long))  # 损失函数
        optimizer.zero_grad()  # 梯度清空
        loss.backward()  # 梯度求导
        optimizer.step()  # 参数优化更新



    return train_loss,train_acc
