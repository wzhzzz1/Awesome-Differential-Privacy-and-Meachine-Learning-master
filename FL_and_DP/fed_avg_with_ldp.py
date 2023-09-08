import math
import os
from FL_and_DP import set_averaged_weights_as_main_model_weights, \
    set_averaged_weights_as_main_model_weights_fully_averaged
from FL_and_DP import local_clients_train_process_with_dp_one_epoch, \
    local_clients_train_process_with_dp_one_batch, local_clients_train_process_without_dp_one_batch, \
    local_clients_train_process_one_epoch_with_ldp_gaussian
from FL_and_DP import send_main_model_to_clients
from data.fed_data_distribution.dirichlet_nonIID_data import fed_dataset_NonIID_Dirichlet
from FL_and_DP import create_model_optimizer_criterion_dict, \
    create_model_optimizer_criterion_dict_with_dp_optimizer
from data.fed_data_distribution.pathological_nonIID_data import pathological_split_noniid
from data.get_data import get_data
from model.CNN import CNN
from optimizer.clipping_and_adding_noise import clipping_and_adding_noise
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from train_and_validation.validation import validation
import torch
import matplotlib.pyplot as plt
import math

# 配置训练环境和超参数
# 配置GPU，这里有两种方式
## 方案一：使用os.environ
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
## 方案二：使用device，后续对要使用GPU的变量使用to(device)即可
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#fl+基于高斯噪声的ldp，模型上传到中心方之前做一次加噪，也是需要裁剪获得本地局部敏感度的
def fed_avg_with_ldp_gaussian(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed,q,max_norm,sigma,delta):
    epoch_list = []
    acc_list = []

    #客户端的样本分配
    clients_data_list, weight_of_each_clients,batch_size_of_each_clients = fed_dataset_NonIID_Dirichlet(train_data,number_of_clients,alpha,seed,q)

    #clients_data_list, weight_of_each_clients,batch_size_of_each_clients=pathological_split_noniid(train_data, number_of_clients, alpha, seed,q)



    # 初始化中心模型,本质上是用来接收客户端的模型并加权平均进行更新的一个变量
    center_model = CNN()
    # # 获取各个客户端的model,optimizer,criterion的组合字典
    clients_model_list, clients_optimizer_list, clients_criterion_list = create_model_optimizer_criterion_dict(number_of_clients, learning_rate,
                                                                                       center_model)
    # #创建一个中心方有噪声和无噪声的model，optimizer和criterion
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=256, shuffle=False)


    orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
    rdp=0.
    print("联邦学习整体流程开始-------------------")

    for i in range(iters):

        print("现在进行和中心方的第{:3.0f}轮联邦训练".format(i+1))
        print("1 中心方广播参数给各个客户端")
        # 1 中心方广播参数给各个客户端
        model_dict = send_main_model_to_clients(center_model, clients_model_list)
        print("2 本地梯度下降最后裁剪加噪一次")
        # 2本地梯度下降最后裁剪加噪一次
        local_clients_train_process_one_epoch_with_ldp_gaussian(number_of_clients,clients_data_list,clients_model_list,clients_criterion_list,clients_optimizer_list,numEpoch,q,max_norm,sigma)

        #上传前裁剪加噪,这边计算DP是一次性计算，没有下采样。
        #这边计算rdp的次数，一个是看多少个epoch,一个是看一个epoch里面做了多少次batch的迭代，这边应该是1/q向下取整，因为我们在组装datasize的时候会trop_out
        # rdp += compute_rdp(q, sigma, numEpoch*math.floor(1/q), orders)
        # epsilon, best_alpha = compute_eps(orders, rdp, delta)
        # print("Iteration: {:3.0f}".format(i + 1) + " | epsilon: {:7.4f}".format(
        #     epsilon) + " | best_alpha: {:7.4f}".format(best_alpha))
        print("3 客户端上传参数到中心方进行加权平均并更新中心方参数(根据客户端数量加权平均)")
        # 3 客户端上传参数到中心方进行加权平均并更新中心方参数(根据客户端数量加权平均)
        main_model = set_averaged_weights_as_main_model_weights(center_model,clients_model_list,weight_of_each_clients)
        #main_model = set_averaged_weights_as_main_model_weights_fully_averaged(center_model, clients_model_list,weight_of_each_clients)

        print("4 查看效果中心方模型效果")
        # 查看效果中心方模型效果
        test_loss, test_accuracy = validation(main_model, test_dl)
        print("Iteration", str(i + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))
        epoch_list.append(i)
        acc_list.append(test_accuracy)
        print("test_loss",test_loss)
    print(acc_list)
    print(epoch_list)
    plt.plot(epoch_list, acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()


if __name__=="__main__":
    train_data, test_data = get_data('mnist', augment=False)
    model = CNN()    #创建一个CNN类的实例对象
    print(model)
    batch_size=64     #批量大小
    learning_rate = 0.02   #学习率
    numEpoch = 1      #客户端本地下降次数
    number_of_clients=10  #客户端的数量
    momentum=0.9  #动量
    iters=20
    alpha=0.05 #狄立克雷的异质参数
    seed=2  #狄立克雷的随机种子
    q_for_batch_size=0.1     #基于该数据采样率组建每个客户端的batchsize
    max_norm=5  #这里需要大一点的裁剪范数对于目前这个情况
    eps=1000
    delta = 1e-2  # 不满足差分隐私的概率
    sigma=(math.sqrt(2*math.log(1.25/delta,math.e)))/eps
    print(sigma)
    fed_avg_with_ldp_gaussian(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed,q_for_batch_size,max_norm,sigma,delta)