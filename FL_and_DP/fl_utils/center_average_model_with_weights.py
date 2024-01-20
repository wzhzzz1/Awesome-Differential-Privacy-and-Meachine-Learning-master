import torch
import torch
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def set_averaged_weights_as_main_model_weights(center_model,clients_model_list,weight_of_each_clients):
    sum_parameters = None  # 用来接所有边缘节点的模型的参数
    global_parameters = {}
    for key, var in center_model.state_dict().items():
        global_parameters[key] = var.clone()

    with torch.no_grad():

        for i in range(len(clients_model_list)):

            local_parameters = clients_model_list[i].state_dict()  # 先把第i个客户端的model取出来

            if sum_parameters is None:  # 先初始化模型字典，主要是初始化key
                sum_parameters = {}
                for key, var in local_parameters.items():
                    if 'norm' not in key and 'bn' not in key and 'downsample.1' not in key:
                        sum_parameters[key] = weight_of_each_clients[i] * var.clone()

            else:  # 然后做值的累加,这边直接加权了
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + weight_of_each_clients[i] * local_parameters[var].to(device)

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var])

    center_model.load_state_dict(global_parameters, strict=True)
    return center_model

def set_averaged_weights_as_main_model_weights_by_cos_similarity(center_model,clients_model_list,weight_of_each_clients):
    
      # 用来接所有边缘节点的模型的参数
    global_parameters = {}
    client_paramaters_list=[]
    cos_similarity_value=[]

    for key, var in center_model.state_dict().items():
        print(key)
        print(var.size())
        global_parameters[key] = var.clone()

    with torch.no_grad():

        for i in range(len(clients_model_list)):
            temp_parameters = {}
            local_parameters = clients_model_list[i].state_dict()  # 先把第i个客户端的model取出来
            for key, var in local_parameters.items():
                if 'norm' not in key and 'bn' not in key and 'downsample.1' not in key:
                    temp_parameters[key] = var.clone()
            print(temp_parameters.keys())
            client_paramaters_list.append(temp_parameters)

        for i in range(len(clients_model_list)):
            vector_A = torch.cat([tensor.view(-1) for tensor in client_paramaters_list[i].values()])
            vector_B = torch.cat([tensor.view(-1) for tensor in global_parameters.values()])
            cos_similarity_value.append(F.cosine_similarity(vector_A.unsqueeze(0), vector_B.unsqueeze(0)))
        print(cos_similarity_value)
    center_model.load_state_dict(global_parameters, strict=True)
    return center_model




#简单的平均，不做加权
def set_averaged_weights_as_main_model_weights_fully_averaged(center_model,clients_model_list,weight_of_each_clients):
    sum_parameters = None  # 用来接所有边缘节点的模型的参数
    global_parameters = {}
    for key, var in center_model.state_dict().items():
        global_parameters[key] = var.clone()

    with torch.no_grad():

        for i in range(len(clients_model_list)):

            local_parameters = clients_model_list[i].state_dict()  # 先把第i个客户端的model取出来

            if sum_parameters is None:  # 先初始化模型字典，主要是初始化key
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()

            else:  # 然后做值的累加,这边直接加权了
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] +  local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / len(clients_model_list))

    center_model.load_state_dict(global_parameters, strict=True)
    return center_model