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
import torch.nn.functional as F

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
        self.bn = OutputNorm1(1, 10)


    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1, 28 * 28)  # 将输入变为28*28的一维向量
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        x = self.bn(x)
        return x


# 卷积转换
class InputNorm1(nn.Module):
    def __init__(self, num_channel, num_feature):
        super().__init__()
        self.num_channel = num_channel
        self.gamma = nn.Parameter(torch.ones(num_channel))
        self.beta = nn.Parameter(torch.zeros(num_channel, num_feature,num_feature))
    def forward(self, x):
        if self.num_channel == 1:
            x = self.gamma * x
            x = x + self.beta
            return x
        if self.num_channel == 3:
            return torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta



class OutputNorm1(nn.Module):
    def __init__(self, num_channel, num_feature):
        super().__init__()
        self.num_channel = num_channel
        self.gamma = nn.Parameter(torch.ones(num_channel))
        self.beta = nn.Parameter(torch.zeros(num_channel, num_feature))

    def forward(self, x):
        if self.num_channel == 1:
            x = self.gamma * x
            x = x + self.beta
            return x
        if self.num_channel == 3:
            return torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta



class Cifar10CNN(nn.Module):
    def __init__(self):
        super(Cifar10CNN, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128*4*4, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 10, bias=True), )

    def forward(self,x):
        x=self.conv(x)
        return x

class Cifar10CNN_IN(nn.Module):
    def __init__(self):
        super(Cifar10CNN_IN, self).__init__()
        self.norm = InputNorm1(1, 32)
        self.conv= nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128*4*4, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 10, bias=True), )

    def forward(self,x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class Cifar10CNN_IN1(nn.Module):
    def __init__(self):
        super(Cifar10CNN_IN1, self).__init__()
        self.norm = InputNorm1(1, 32)
        self.bn = OutputNorm1(1, 10)
        self.conv= nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128*4*4, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 10, bias=True), )

    def forward(self,x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


