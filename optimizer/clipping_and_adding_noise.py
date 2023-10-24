import torch


def PM_adding_noise(model,epsilon):

    per_data_parameters_grad_dict={}
    params=model.state_dict()
    with torch.no_grad():
        #计算范数
        for key,var in params.items():

            per_data_parameters_grad_dict[key]=var.clone().detach()

        for key in per_data_parameters_grad_dict:

            if 'norm' not in key and 'bn' not in key and 'downsample.1' not in key:  # 这个downsample是resnet里特有的，norm就是个性化层
                print(per_data_parameters_grad_dict[key])


        #问题出现在这个model.load_state_dict,我们看一下具体是什么问题
        model.load_state_dict(per_data_parameters_grad_dict, strict=True)
    return model



def PM(eps, num):
    c = (math.e ** (eps / 2) + 1) / (math.e ** (eps / 2) - 1)

    r = random.random()
    if r < (math.e ** (eps / 2)) / (math.e ** (eps / 2) + 1):
        lt = ((c + 1) / 2) * num - ((c - 1) / 2)
        rt = lt + c - 1
        perturbed_num = random.uniform(lt, rt)
    else:
        lt = (c + 1) / 2 * num - (c - 1) / 2
        rt = lt + c - 1
        if random.random()<(lt+c)/(c+1):  # 随机选择一个范围
            perturbed_num = random.uniform(-c, lt-0.00000001)
        else:
            perturbed_num = random.uniform(rt+0.000000001, c)

    return perturbed_num