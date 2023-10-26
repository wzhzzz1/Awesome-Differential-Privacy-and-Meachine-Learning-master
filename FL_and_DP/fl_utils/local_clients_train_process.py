import sys

sys.path.append('/home/wangzihang/FL-DP/')

# 该函数在各个节点中训练本地模型,可打印部分客户端的训练信息,这个是无裁剪的梯度下降
import math
import os
import torch
from torch.utils.data import TensorDataset

from data.util.sampling import get_data_loaders_uniform_without_replace
from optimizer.clipping_and_adding_noise import PM_adding_noise
from train_and_validation.train import train
from train_and_validation.train_with_dp import train_dynamic_add_noise
import time
import pandas as pd


# 无加噪的联邦学习，本地每个numEpoch做满一个epoch，即本地客户端的所有本地数据集
def local_clients_train_process_without_dp_one_epoch(number_of_clients, clients_data_list, clients_model_list,
                                                     clients_criterion_list, clients_optimizer_list, numEpoch, q):
    # 循环客户端
    for i in range(number_of_clients):

        batch_size = math.floor(len(clients_data_list[i]) * q)
        batch_size = 256
        train_dl = torch.utils.data.DataLoader(
            clients_data_list[i], batch_size=batch_size, shuffle=False, drop_last=True)
        # 各客户端取到自己对应的模型,损失函数和优化器
        model = clients_model_list[i]
        model = model.cuda()
        criterion = clients_criterion_list[i]
        optimizer = clients_optimizer_list[i]

        for epoch in range(numEpoch):  # 每个客户端本地进行训练

            train_loss, train_accuracy = train(model, train_dl, optimizer)
            # test_loss, test_accuracy = validation(model, test_dl)  联邦下，这里本地没有合适的测试集了

            # if i < number_of_clients:
            #     print("epoch: {:3.0f}".format(epoch + 1) + " | train_loss: {:7.5f}".format(
            #         train_loss) + " | train_accuracy: {:7.5f}".format(train_accuracy))


# 无加噪的联邦学习，本地每个numEpoch做一个batch，即本地客户端的部分数据
def local_clients_train_process_without_dp_one_batch(number_of_clients, clients_data_list, clients_model_list,
                                                     clients_criterion_list, clients_optimizer_list, numEpoch, q):
    # 循环客户端
    for i in range(number_of_clients):
        print("第", i + 1, "个客服端进行训练")
        batch_size = math.floor(len(clients_data_list[i]) * q)
        batch_size = 64
        # 包装抽样函数
        minibatch_size = batch_size
        microbatch_size = 1  # 这里默认1就好
        iterations = 1  # n个batch，这边就定一个，每次训练采样一个Lot
        minibatch_loader, microbatch_loader = get_data_loaders_uniform_without_replace(minibatch_size, microbatch_size,
                                                                                       iterations)  # 无放回均匀采样
        # 各客户端取到自己对应的模型,损失函数和优化器
        model = clients_model_list[i]
        model = model.cuda()
        criterion = clients_criterion_list[i]
        optimizer = clients_optimizer_list[i]
        # print("clients_data_list:",clients_data_list[i][1] )
        # if i < number_of_clients:
        #     print("Client:", i)

        for epoch in range(numEpoch):  # 每个客户端本地进行训练
            print("第", epoch + 1, "轮")
            train_dl = minibatch_loader(clients_data_list[i])
            train_loss, train_accuracy = train(model, train_dl, optimizer)
            # test_loss, test_accuracy = validation(model, test_dl)  联邦下，这里本地没有合适的测试集了

            # if i < number_of_clients:
            #     print("epoch: {:3.0f}".format(epoch + 1) + " | train_loss: {:7.5f}".format(
            #         train_loss) + " | train_accuracy: {:7.5f}".format(train_accuracy))


# 加噪的联邦学习，本地每个numEpoch做一个epoch，即本地客户端的所有数据，每次batch训练完的梯度裁剪加噪
def local_clients_train_process_with_dp_one_epoch(number_of_clients, clients_data_list, clients_model_list,
                                                  clients_criterion_list, clients_optimizer_list, numEpoch, q):
    # 循环客户端
    for i in range(number_of_clients):

        batch_size = math.floor(len(clients_data_list[i]) * q)
        train_dl = torch.utils.data.DataLoader(
            clients_data_list[i], batch_size=batch_size, shuffle=False, drop_last=True)

        # 各客户端取到自己对应的模型,损失函数和优化器
        model = clients_model_list[i]
        optimizer = clients_optimizer_list[i]

        # if i < number_of_clients:
        #     print("Client:", i)

        for epoch in range(numEpoch):  # 每个客户端本地进行训练

            train_loss, train_accuracy = train_dynamic_add_noise(model, train_dl, optimizer)
            # test_loss, test_accuracy = validation(model, test_dl)  联邦下，这里本地没有合适的测试集了

            # if i < number_of_clients:
            #     print("epoch: {:3.0f}".format(epoch + 1) + " | train_loss: {:7.5f}".format(
            #         train_loss) + " | train_accuracy: {:7.5f}".format(train_accuracy))


# 加噪的联邦学习，本地每个numEpoch做一个batch，batch训练完的梯度裁剪加噪
def local_clients_train_process_with_dp_one_batch(number_of_clients, clients_data_list, clients_model_list,
                                                  clients_criterion_list, clients_optimizer_list, numEpoch, q):
    # 循环客户端
    for i in range(number_of_clients):

        batch_size = math.floor(len(clients_data_list[i]) * q)

        # 包装抽样函数
        minibatch_size = batch_size  # 这里比较好的取值是根号n，n为每个客户端的样本数
        microbatch_size = 1  # 这里默认1就好
        iterations = 1  # n个batch，这边就定一个，每次训练采样一个Lot
        minibatch_loader, microbatch_loader = get_data_loaders_uniform_without_replace(minibatch_size, microbatch_size,
                                                                                       iterations)  # 无放回均匀采样
        # 各客户端取到自己对应的模型,损失函数和优化器
        model = clients_model_list[i]
        optimizer = clients_optimizer_list[i]

        if i < number_of_clients:
            print("Client:", i)

        for epoch in range(numEpoch):  # 每个客户端本地进行训练

            train_dl = minibatch_loader(clients_data_list[i])
            train_loss, train_accuracy = train_dynamic_add_noise(model, train_dl, optimizer)
            # test_loss, test_accuracy = validation(model, test_dl)  联邦下，这里本地没有合适的测试集了

            if i < number_of_clients:
                print("epoch: {:3.0f}".format(epoch + 1) + " | train_loss: {:7.5f}".format(
                    train_loss) + " | train_accuracy: {:7.5f}".format(train_accuracy))


# 本地训练不进行裁剪加噪，最好上传到联邦中心方再进行裁剪加噪
def local_clients_train_process_one_epoch_with_ldp_PM(number_of_clients, clients_data_list, clients_model_list,
                                                      clients_criterion_list, clients_optimizer_list, numEpoch, q,
                                                      epsilon):
    # 循环客户端
    for i in range(number_of_clients):
        print('第', i + 1, '个客户端正在训练')
        batch_size = math.floor(len(clients_data_list[i]) * q)
        batch_size = 64
        train_dl = torch.utils.data.DataLoader(
            clients_data_list[i], batch_size=batch_size, shuffle=False, drop_last=True)

        # 各客户端取到自己对应的模型,损失函数和优化器
        model = clients_model_list[i]
        model = model.cuda()
        criterion = clients_criterion_list[i]
        optimizer = clients_optimizer_list[i]

        # if i < number_of_clients:
        #     print("Client:", i)

        for epoch in range(numEpoch):  # 每个客户端本地进行训练

            train_loss, train_accuracy = train(model, train_dl, optimizer)
            # test_loss, test_accuracy = validation(model, test_dl)  联邦下，这里本地没有合适的测试集了

            # if i < number_of_clients:
            #     print("epoch: {:3.0f}".format(epoch + 1) + " | train_loss: {:7.5f}".format(
            #         train_loss) + " | train_accuracy: {:7.5f}".format(train_accuracy))
        total_params_sum1 = 0
        params = model.state_dict()
        per_data_parameters_grad_dict = {}
        for key, var in params.items():
            per_data_parameters_grad_dict[key] = var.clone().detach()
            total_params_sum1 += var.sum().item()
        print('添加噪音前模型参数总和', total_params_sum1)

        print('第', i + 1, '个客户端正在对模型参数使用PM机制添加噪音')
        start = time.time()
        model = PM_adding_noise(model, epsilon)
        end = time.time()
        print("添加噪音耗时：", end - start)

        total_params_sum2 = 0
        params = model.state_dict()
        for key, var in params.items():
            total_params_sum2 += var.sum().item()
        print('添加噪音后模型参数总和', total_params_sum2)
        if abs(total_params_sum2 - total_params_sum1) > 100:

            for key, tensor in per_data_parameters_grad_dict.items():
                df = pd.DataFrame()
                # 将每个张量的值转换为NumPy数组，然后保存到DataFrame中
                df[key] = tensor.detach().cpu().numpy().flatten()
                df.to_csv('./result/error_' + key + '.csv', index=False)
            return
        print('----------------------------------------------------------------------------------------')
        # print("model:",model.state_dict())
