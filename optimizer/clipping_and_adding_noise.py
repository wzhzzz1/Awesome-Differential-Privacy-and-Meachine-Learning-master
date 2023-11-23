import torch
import math
import random
import numpy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def PM_adding_noise(model,epsilon): #这个地方可能最好调用以下ray来计算，不然感觉太慢了

    per_data_parameters_grad_dict={}
    params=model.state_dict()
    with torch.no_grad():
        #计算范数
        for key,var in params.items():

            per_data_parameters_grad_dict[key]=var.clone().detach()

        for key in per_data_parameters_grad_dict:

            if 'norm' not in key and 'bn' not in key and 'downsample.1' not in key:  # 这个downsample是resnet里特有的，norm就是个性化层

                max_value = torch.max(per_data_parameters_grad_dict[key])
                min_value = torch.min(per_data_parameters_grad_dict[key])
                bound = max(abs(max_value),abs(min_value))
                per_data_parameters_grad_dict[key] = per_data_parameters_grad_dict[key]/bound
                temp = per_data_parameters_grad_dict[key].cpu().numpy()
                if len(temp.shape) == 2:
                    num_rows, num_cols = temp.shape
                    for i in range(num_rows):  # 遍历行
                        for j in range(num_cols):  # 遍历列
                            temp[i][j] = PM(epsilon, temp[i][j])
                elif len(temp.shape) == 1:
                    num_rows = len(temp)
                    for i in range(num_rows):  # 遍历行
                        temp[i] = PM(epsilon, temp[i])
                elif len(temp.shape) == 4:
                    num_rows, num_cols ,num_x,num_y= temp.shape
                    for i in range(num_rows):  # 遍历行
                        for j in range(num_cols):  # 遍历列
                            for m in range(num_x):
                                for n in range(num_y):
                                    temp[i][j] = PM(epsilon, temp[i][j][m][n])
                per_data_parameters_grad_dict[key] = torch.tensor(temp).to(device) * bound

        #问题出现在这个model.load_state_dict,我们看一下具体是什么问题
        #model.load_state_dict(per_data_parameters_grad_dict, strict=True)
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
            perturbed_num = random.uniform(-c, lt)
        else:
            perturbed_num = random.uniform(rt, c)

    return perturbed_num
