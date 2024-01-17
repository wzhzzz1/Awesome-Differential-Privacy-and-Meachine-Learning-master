import sys

sys.path.append('/home/wangzihang/FL-DP/')
from FL_and_DP.fl_utils.center_average_model_with_weights import set_averaged_weights_as_main_model_weights, \
    set_averaged_weights_as_main_model_weights_fully_averaged
from FL_and_DP.fl_utils.local_clients_train_process import local_clients_train_process_without_dp_one_epoch, \
    local_clients_train_process_without_dp_one_batch, local_clients_train_process_one_epoch_with_ldp_PM
from FL_and_DP.fl_utils.send_main_model_to_clients import send_main_model_to_clients
from data.fed_data_distribution.dirichlet_nonIID_data import fed_dataset_NonIID_Dirichlet
from FL_and_DP.fl_utils.optimizier_and_model_distribution import create_model_optimizer_criterion_dict
from data.fed_data_distribution.pathological_nonIID_data import pathological_split_noniid
from data.get_data import get_data
from model.modelUtil import mnist_fully_connected, mnist_fully_connected_IN, mnist_fully_connected_IN1, Cifar10CNN, \
    Cifar10CNN_IN, Cifar10CNN_IN1, ResNet18, ResNet18_IN, ResNet18_IN1
from train_and_validation.validation import validation
import torch
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import math


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'cifar100', 'fmnist', 'emnist', 'purchase', 'chmnist'])
    parser.add_argument('--client', type=int, default=10)
    parser.add_argument('--batchsize', type=int, help='the number of class for this dataset', default=64)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--model', type=str, default='DNN')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='狄立克雷的异质参数')
    parser.add_argument('--seed', type=int, default=1,
                        help='随机种子')
    parser.add_argument('--sr', type=float, default=0.1,
                        help='采样率')
    parser.add_argument('--eps', type=float, default=0,
                        help='隐私预算')
    parser.add_argument('--personal', type=int, default=0,
                        help='是否用个性化模型')
    parser.add_argument('--ptype', type=str, default='no',
                        help='是否用双个性化模型')
    parser.add_argument('--usedp', type=int, default=0,
                        help='是否用dp')
    parser.add_argument('--use_client_selection_by_similarity', type=int, default=0,
                        help='是否用基于余弦相似度的客户端选择')
    args = parser.parse_args()
    return args


def fed_avg(train_data, test_data, number_of_clients, learning_rate, model_kind, momentum, numEpoch, iters, alpha, seed,
            q, per, ptype, usedp, epsilon, use_cos_similarity):
    epoch_list = []
    acc_list = []
    # 客户端的样本分配
    clients_data_list, weight_of_each_clients, batch_size_of_each_clients = fed_dataset_NonIID_Dirichlet(train_data,
                                                                                                         number_of_clients,
                                                                                                         alpha, seed, q)
    # clients_data_list, weight_of_each_clients,batch_size_of_each_clients =pathological_split_noniid(train_data,number_of_clients,alpha,seed,q)

    # 初始化中心模型,本质上是用来接收客户端的模型并加权平均进行更新的一个变量
    if model_kind == 'DNN':
        center_model = mnist_fully_connected(10)
    elif model_kind == 'Resnet18':
        center_model = ResNet18()
    all_train_loss = []
    # 各个客户端的model,optimizer,criterion的分配

    if per == 0:
        clients_model_list, clients_optimizer_list, clients_criterion_list = create_model_optimizer_criterion_dict(
            number_of_clients, learning_rate, center_model)

    else:
        if ptype == 'single':
            if model_kind == 'DNN':
                clients_model_list, clients_optimizer_list, clients_criterion_list = create_model_optimizer_criterion_dict(
                    number_of_clients, learning_rate, mnist_fully_connected_IN(10))
            elif model_kind == 'Resnet18':
                clients_model_list, clients_optimizer_list, clients_criterion_list = create_model_optimizer_criterion_dict(
                    number_of_clients, learning_rate, ResNet18_IN())
        if ptype == 'double':
            if model_kind == 'DNN':
                clients_model_list, clients_optimizer_list, clients_criterion_list = create_model_optimizer_criterion_dict(
                    number_of_clients, learning_rate, mnist_fully_connected_IN1(10))
            elif model_kind == 'Resnet18':
                clients_model_list, clients_optimizer_list, clients_criterion_list = create_model_optimizer_criterion_dict(
                    number_of_clients, learning_rate, ResNet18_IN1())
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=256, shuffle=False)

    print("联邦学习整体流程开始-------------------")
    test_accuracy_record = []
    test_loss_record = []

    for i in range(iters):

        print("现在进行和中心方的第{:3.0f}轮联邦训练".format(i + 1))

        if usedp == 0:
            train_loss = local_clients_train_process_without_dp_one_epoch(number_of_clients, clients_data_list,
                                                                          clients_model_list,
                                                                          clients_criterion_list,
                                                                          clients_optimizer_list, numEpoch,
                                                                          q)
            all_train_loss.append(train_loss)
        else:
            train_loss = local_clients_train_process_one_epoch_with_ldp_PM(number_of_clients, clients_data_list,
                                                                           clients_model_list,
                                                                           clients_criterion_list,
                                                                           clients_optimizer_list, numEpoch,
                                                                           q, epsilon)
            all_train_loss.append(train_loss)
        if use_cos_similarity == 0:
            main_model = set_averaged_weights_as_main_model_weights(center_model, clients_model_list,
                                                                weight_of_each_clients)
        else:
            main_model = set_averaged_weights_as_main_model_weights_by_cos_similarity(center_model, clients_model_list,
                                                                    weight_of_each_clients)
            pass
        '''
        if i==iters-1:
            print('train finish')
            with torch.no_grad():
                for j in range(len(clients_model_list)):
                    if j ==9:
                        for key, value in clients_model_list[j].state_dict().items():
                            #if 'norm'  in key or 'bn' in key or 'downsample.1' in key:  # 这个downsample是resnet里特有的，norm就是个性化层
                            print(j)
                            print(key)
                            print(value)
                print('global model')
                for key, value in main_model.state_dict().items():
                    # if 'norm'  in key or 'bn' in key or 'downsample.1' in key:  # 这个downsample是resnet里特有的，norm就是个性化层
                    print(key)
                    print(value)
        '''
        clients_model_list = send_main_model_to_clients(center_model, clients_model_list)
        '''
        if i==iters-1:
            print('fuhe')
            with torch.no_grad():
                for j in range(len(clients_model_list)):
                    if j ==9:
                        print('local model')
                        for key, value in clients_model_list[j].state_dict().items():
                            #if 'norm'  in key or 'bn' in key or 'downsample.1' in key:  # 这个downsample是resnet里特有的，norm就是个性化层
                            print(j)
                            print(key)
                            print(value)
        '''
        '''
        if per == 1:
            for j in range(len(clients_model_list)):
                p_test_loss, p_test_accuracy = validation(clients_model_list[j], test_dl)
                print(
                    f'第{j + 1}个客户端模型' f'Test set: Average loss: {p_test_loss:.4f}, 'f'Accuracy: ({p_test_accuracy:.2f}%)')
        '''
        # 查看效果中心方模型效果
        test_loss, test_accuracy = validation(main_model, test_dl)
        print(f'服务器模型:')
        print(f'Test set: Average loss: {test_loss:.4f}, 'f'Accuracy: ({test_accuracy:.2f}%)')

        test_loss_record.append(test_loss)
        test_accuracy_record.append(test_accuracy)

        epoch_list.append(i + 1)
        acc_list.append(test_accuracy)

    plt.figure(figsize=(24, 16))
    plt.plot(epoch_list, acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xticks(range(0, 101, 10), rotation=45)
    plt.yticks(range(0, 101, 5), rotation=45)
    plt.savefig('./result/fedavg_result_' + 'iters' + str(iters) + '_appha' + str(alpha) + '_clients' + str(
        number_of_clients) + '_lr' + str(learning_rate) + '_personal' + str(per) + '_ptype_' + str(
        ptype) + '_usedp' + str(usedp) + '_eps' + str(epsilon) + '.png')
    data = {'Epoch': epoch_list, 'Accuracy': acc_list, 'test_loss': test_loss_record}
    df = pd.DataFrame(data)
    df.to_csv('./result/fedavg_result_' + 'iters' + str(iters) + '_appha' + str(alpha) + '_clients' + str(
        number_of_clients) + '_lr' + str(learning_rate) + '_personal' + str(per) + '_ptype_' + str(
        ptype) + '_usedp' + str(usedp) + '_eps' + str(epsilon) + '.csv', index=False)
    print(all_train_loss)
    # record=[iters,numEpoch,test_loss_record,test_accuracy_record]

    # torch.save(record, "../record/{}.pth".format(int(numEpoch)))


if __name__ == "__main__":
    args = parse_arguments()
    train_data, test_data = get_data(args.data, augment=False)
    # print(train_data.data)
    #
    # print(train_data.__dict__)
    batch_size = args.batchsize  # 小批量
    learning_rate = args.lr  # 学习率
    numEpoch = args.epoch  # 客户端本地下降次数
    number_of_clients = args.client  # 客户端数量
    momentum = 0.9  # 动量
    iters = args.iters  # 联邦学习中的全局迭代次数
    alpha = args.alpha  # 狄立克雷的异质参数
    seed = args.seed  # 随机种子
    q_for_batch_size = args.sr  # 基于该数据采样率组建每个客户端的batchsize
    epsilon = args.eps
    model_kind = args.model
    per = args.personal
    ptype = args.ptype
    usedp = args.usedp
    use_cos_similarity = args.use_client_selection_by_similarity
    fed_avg(train_data, test_data, number_of_clients, learning_rate, model_kind, momentum, numEpoch, iters, alpha, seed,
            q_for_batch_size, per, ptype, usedp, epsilon, use_cos_similarity)
