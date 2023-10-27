import torch
from torchvision.models import alexnet, resnet18
from torch.nn.functional import relu, softmax, max_pool2d
from torch.nn.utils import spectral_norm
from torch import nn, tanh
import copy
from opacus.grad_sample import register_grad_sampler
from typing import Dict
import torchvision
from collections import OrderedDict
from numpy import median
import numpy as np
import torch.nn.functional as func
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# baseline ,无个性化转换
class mnist_fully_connected(nn.Module):
    def __init__(self, num_classes):
        super(mnist_fully_connected, self).__init__()
        self.hidden1 = 600
        self.hidden2 = 100
        self.fc1 = nn.Linear(28 * 28, self.hidden1, bias=False)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=False)
        self.fc3 = nn.Linear(self.hidden2, num_classes, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 线性转换，ax+b
class InputNorm(nn.Module):
    def __init__(self, num_channel, num_feature):
        super().__init__()
        self.num_channel = num_channel
        self.gamma = nn.Parameter(torch.ones(num_channel))
        self.beta = nn.Parameter(torch.zeros(num_channel, num_feature, num_feature))

    def forward(self, x):
        if self.num_channel == 1:
            x = self.gamma * x
            x = x + self.beta
            return x
        if self.num_channel == 3:
            return torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta


class mnist_fully_connected_IN(nn.Module):
    def __init__(self, num_classes):
        super(mnist_fully_connected_IN, self).__init__()
        self.hidden1 = 600
        self.hidden2 = 100
        self.fc1 = nn.Linear(28 * 28, self.hidden1, bias=False)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=False)
        self.fc3 = nn.Linear(self.hidden2, num_classes, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.norm = InputNorm(1, 28)

    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1, 28 * 28)  # 将输入变为28*28的一维向量
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 非线性转换，ax^3+b
class mnist_fully_connected_IN1(nn.Module):
    def __init__(self, num_classes):
        super(mnist_fully_connected_IN1, self).__init__()
        self.hidden1 = 600
        self.hidden2 = 100
        self.fc1 = nn.Linear(28 * 28, self.hidden1, bias=False)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=False)
        self.fc3 = nn.Linear(self.hidden2, num_classes, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.norm = InputNorm1(1, 28)

    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1, 28 * 28)  # 将输入变为28*28的一维向量
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 卷积转换
class InputNorm1(nn.Module):
    def __init__(self, num_channel, num_feature):
        super().__init__()
        self.num_channel = num_channel
        self.gamma = nn.Parameter(torch.ones(num_channel))
        self.beta = nn.Parameter(torch.zeros(num_channel, num_feature, num_feature))
        self.conv = nn.Sequential(nn.Conv2d(1, 1, 3, 1, padding=1))
        self.gamma1 = nn.Parameter(torch.ones(num_channel))
        self.plr = torch.ones(num_channel,device=device)
    def forward(self, x):
        if self.num_channel == 1:
            temp = self.conv(x)*self.gamma1*self.plr
            self.plr = 1/2*self.plr
            x = self.gamma * x
            x = x + temp
            x = x + self.beta

            return x
        if self.num_channel == 3:
            return torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta



#x = self.gamma * torch.log(1+x)
'''
class InputNorm1(nn.Module):
    def __init__(self, num_channel, num_feature):
        super().__init__()
        self.num_channel = num_channel
        self.gamma = nn.Parameter(torch.ones(num_channel))
        self.gamma1 = nn.Parameter(torch.ones(num_channel))
        self.eps = nn.Parameter(torch.ones(num_channel))
        self.beta = nn.Parameter(torch.zeros(num_channel, num_feature, num_feature))

    def forward(self, x):
        if self.num_channel == 1:
            x = self.gamma1 * torch.log(1+x)+self.gamma*x
            x = x + self.beta
            return x
        if self.num_channel == 3:
            return torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta
'''


'''方案1：效果不稳定，好的时候比baseline好一点，坏的时候差一点(原理是在线性变换前引入卷积变换，使的初始特征被表示的更全面了，但是毕竟定义的conv不可学习，或许多引入几个卷积层效果更好)
class InputNorm1(nn.Module):
    def __init__(self, num_channel, num_feature):
        super().__init__()
        self.num_channel = num_channel
        self.gamma = nn.Parameter(torch.ones(num_channel))
        self.beta = nn.Parameter(torch.zeros(num_channel, num_feature, num_feature))
        self.conv = nn.Sequential(nn.Conv2d(1, 1, 3, 1, padding=1))

    def forward(self, x):
        if self.num_channel == 1:
            temp = self.conv(x)

            x = self.gamma * x
            x = x + self.beta
            x = x + temp
            return x
        if self.num_channel == 3:
            return torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta


'''

'''方案2：效果稳定，比baseline好挺多的(原理是在线性变换前引入可学习的卷积变换，使的初始特征被表示的更全面了)
class InputNorm1(nn.Module):
    def __init__(self, num_channel, num_feature):
        super().__init__()
        self.num_channel = num_channel
        self.gamma = nn.Parameter(torch.ones(num_channel))
        self.beta = nn.Parameter(torch.zeros(num_channel, num_feature, num_feature))
        self.conv = nn.Sequential(nn.Conv2d(1, 1, 3, 1, padding=1))
        self.gamma1 = nn.Parameter(torch.ones(num_channel))
        self.beta1 = nn.Parameter(torch.zeros(num_channel, num_feature, num_feature))
    def forward(self, x):
        if self.num_channel == 1:
            temp = self.conv(x)*self.gamma1

            x = self.gamma * x
            x = x + self.beta
            x = x + temp+self.beta1
            return x
        if self.num_channel == 3:
            return torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta
'''

'''方案3，和方案2差不多，但是局部模型效果更好了
class InputNorm1(nn.Module):
    def __init__(self, num_channel, num_feature):
        super().__init__()
        self.num_channel = num_channel
        self.gamma = nn.Parameter(torch.ones(num_channel))
        self.beta = nn.Parameter(torch.zeros(num_channel, num_feature, num_feature))
        self.conv = nn.Sequential(nn.Conv2d(1, 1, 3, 1, padding=1))

    def forward(self, x):
        if self.num_channel == 1:
            temp = self.conv(x)*self.gamma

            x = self.gamma * x
            x = x + self.beta
            x = x + temp
            return x
        if self.num_channel == 3:
            return torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta
'''