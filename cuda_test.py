import torch


def gpu_is_available():
    print('\nGPU details:')
    print(f'    gpu_is_available      : ', torch.cuda.is_available())
    print(f'    cuda_device_count     : ', torch.cuda.device_count())
    print(f'    cuda_device_name      : ', torch.cuda.get_device_name())
    print(f'    cuda_device_capability: ', torch.cuda.get_device_capability(0))


gpu_is_available()
#
