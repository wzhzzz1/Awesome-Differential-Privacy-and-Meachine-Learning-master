import torch


# 该函数将主模型的参数发送到节点。
def send_main_model_to_clients(center_model, clients_model_list):
    with torch.no_grad():
        weights=center_model.state_dict()
        for i in range(len(clients_model_list)):
            for key, value in clients_model_list[i].state_dict().items():
                if 'norm' not in key and 'bn' not in key and 'weight' not in key and 'downsample.1' not in key:  # 这个downsample是resnet里特有的，norm就是个性化层
                    clients_model_list[i].state_dict()[key].data.copy_(weights[key])
    return clients_model_list



