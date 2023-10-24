import torch


def PM_adding_noise(model,epsilon):

    per_data_parameters_grad_dict={}
    params=model.state_dict()
    with torch.no_grad():
        #计算范数
        for key,var in params.items():

            per_data_parameters_grad_dict[key]=var.clone().detach()

        for key in per_data_parameters_grad_dict:


            per_data_parameters_grad_dict[key] += max_norm * noise_scale * torch.randn_like(per_data_parameters_grad_dict[key])


        #问题出现在这个model.load_state_dict,我们看一下具体是什么问题
        model.load_state_dict(per_data_parameters_grad_dict, strict=True)
    return model