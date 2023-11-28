import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from model.vision import LeNet, ResNet18,weights_init
from model.modelUtil import mnist_fully_connected, mnist_fully_connected_IN, mnist_fully_connected_IN1

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="5",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

data_cifar = datasets.CIFAR10("../data", download=True)
To_tensor = transforms.ToTensor()
To_image = transforms.ToPILImage()

img_index = args.index

gt_data = To_tensor(data_cifar[img_index][0]).to(
    device)  # image_index[i][0]表示的是第I张图片的data，image_index[i][1]表示的是第i张图片的lable

if len(args.image) > 1:  # 得到预设参数的图片并将其转换为tensor对象
    gt_data = Image.open(args.image)
    gt_data = To_tensor(gt_data).to(device)

gt_data = gt_data.view(1, *gt_data.size())

gt_label = torch.Tensor([data_cifar[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label)

plt.imshow(To_image(gt_data[0].cpu()))
plt.axis('off')
plt.savefig("./attack_image/sample.png")
plt.clf()

net = LeNet().to(device)

torch.manual_seed(1234)

#net.apply(weights_init)
criterion = cross_entropy_for_onehot  # 调用损失函数

# compute original gradient 
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())  # 获取对参数W的梯度

original_dy_dx = list((_.detach().clone() for _ in dy_dx))  # 对原始梯度复制

# generate dummy data and label
#dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)

pat_1 = torch.rand([3,16,16])
pat_2 = torch.cat((pat_1,pat_1),dim=1)
pat_4 = torch.cat((pat_2,pat_2),dim=2)
dummy_data = torch.unsqueeze(pat_4,dim=0).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(To_image(dummy_data[0].cpu()))
#plt.imshow(tp(gt_data[imidx].cpu()), cmap='gray')   #灰度图像专用
optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

history = []
for iters in range(300):
    def closure():
        optimizer.zero_grad()  # 梯度清零

        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)  # faked数据得到的梯度

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()  # 计算fake梯度与真实梯度的均方损失
        grad_diff.backward()  # 对损失进行反向传播    优化器的目标是fake_data, fake_label

        return grad_diff


    optimizer.step(closure)
    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(To_image(dummy_data[0].cpu()))

plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    #plt.imshow(history[i], cmap='gray')#灰度图像
    plt.title("iter=%d" % (i * 5))
    plt.axis('off')
plt.savefig("./attack_image/attack_result.png")
plt.show()
