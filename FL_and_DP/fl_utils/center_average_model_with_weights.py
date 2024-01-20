import torch
import torch
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cosine_similarity(dict1, dict2):
    # 获取两个字典的键的并集
    all_keys = set(dict1.keys()) | set(dict2.keys())

    # 计算两个字典的向量表示
    vector1 = [dict1.get(key, 0) for key in all_keys]
    vector2 = [dict2.get(key, 0) for key in all_keys]

    # 计算余弦相似度的分子
    dot_product = sum(x * y for x, y in zip(vector1, vector2))

    # 计算余弦相似度的分母
    magnitude1 = sqrt(sum(x ** 2 for x in vector1))
    magnitude2 = sqrt(sum(y ** 2 for y in vector2))

    # 计算余弦相似度
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity



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
        global_parameters[key] = var.clone().to(device)
    print(global_parameters.keys())
    with torch.no_grad():

        for i in range(len(clients_model_list)):
            temp_parameters = {}
            local_parameters = clients_model_list[i].state_dict()  # 先把第i个客户端的model取出来
            for key, var in local_parameters.items():
                if 'norm' not in key and 'bn' not in key and 'downsample.1' not in key:
                    temp_parameters[key] = var.clone().to(device)
            print(temp_parameters.keys())
            client_paramaters_list.append(temp_parameters)

    for i in range(len(clients_model_list)):
        cos_similarity_value.append(cosine_similarity(global_parameters, client_paramaters_list[i]))

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