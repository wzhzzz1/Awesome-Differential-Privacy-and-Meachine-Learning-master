import time
import numpy as np
import math
#import tensorflow as tf
import copy

# PM机制
# 输入一个value_t∈[-1,1]，隐私参数ε，输出无偏的扰动值noised_t∈[-C,C]
def PMmechanism(value_t, epsilon):

    # 定义算法参数
    C = (pow(math.e,epsilon/2)+1) / (pow(math.e,epsilon/2)-1)
    p = ( pow(math.e,epsilon) - pow(math.e,epsilon/2) ) / ( 2*pow(math.e,epsilon/2)+2 )
    lt = value_t * (C+1)/2 - (C-1)/2
    rt = lt + C - 1
    #print('分段区间：',-C,lt,rt,C)
    # 区间[lt,rt]内一点的输出概率为p，[-C,C]-[lt,rt]内一点的输出概率为p/e^ε
    p_near = p * (C-1) # p_near表示高概率区间的采样概率
    
    # 生成伯努利变量
    pr = np.random.uniform(low=0, high=1)
    noised_t = .0
    if pr<p_near:
        # 在高概率区间均匀采样
        noised_t = np.random.uniform(low=lt, high=rt)
    else:
        # 在低概率区间均匀采样
        noised_t = np.random.uniform(low=-C, high=C)
        while (noised_t>lt) & (noised_t<rt):
            noised_t = np.random.uniform(low=-C, high=C)
  
    return noised_t

# 辅助功能函数
# 输入指定夹角和v1、v2，判断向量v1，v2的夹角是否在指定夹角内
def InAngle(cos,v1,v2):
    cosin = np.dot(v1,v2)/(np.linalg.norm(x=v1, ord=2)*np.linalg.norm(x=v2, ord=2))
    # 如果实际的cosin比目标的cos大，则说明在夹角范围内，返回True
    return cosin>cos

# 辅助功能函数
# 生成单位球壳上的标准向量
def make_random_vec():
    # 单位立方体中的向量vec
    vec = np.random.uniform(low=-1,high=1,size=3)
    # 仅保留范围球内的向量vec
    while np.linalg.norm(vec, ord=2)>1:
        vec = np.random.uniform(low=-1,high=1,size=3)
    # 标准化，定位到球壳上
    vec = vec / np.linalg.norm(vec, ord=2)
    return vec

# RD机制
# 输入一个三维向量input_t,隐私参数ε，输出无偏的扰动向量nV
def RDmechanism(input_t,epsilon):
    # 定义算法参数
    cos_alpha = (pow(math.e,epsilon)-1) / (pow(math.e,epsilon)+1)
    print(cos_alpha)
    # 高概率区间的采样概率
    p_H = pow(math.e,epsilon)*(1-cos_alpha) / ( pow(math.e,epsilon)*(1-cos_alpha)+(1+cos_alpha) )
    print(p_H)
    # 生成伯努利变量
    r = np.random.uniform(low=0, high=1)
    
    # 随机标准向量
    nV = make_random_vec()
    if r<p_H :
        # 在近方向采样，要求夹角小于alpha
        while not InAngle(cos_alpha, nV, input_t):
            nV = make_random_vec()
    else:
        # 在远方向采样，要求夹角大于alpha
        while InAngle(cos_alpha, nV, input_t):
            nV = make_random_vec()
    # 进行范数修正，保证nV的期望范数为1
    nV = nV*2 / cos_alpha
    # 原向量范数
    l = np.linalg.norm(x=input_t, ord=2) 
    
    return nV*l

# Laplace机制
# 输入一个value_t∈[-1,1]，隐私参数ε，输出noised_t
def LapMechanism(value_t,epsilon):
    # 定义算法参数，
    sensit = 2 
    sigma = sensit/epsilon
    # 噪音为Lap(0,Δf/epsilon)
    return value_t + np.random.laplace(loc=0.0, scale=sigma)

if __name__ == "__main__":
    arr = np.empty((10000, 3))

    # 循环添加1000个三维向量
    for i in range(10000):
        # 生成一个三维向量
        vector = make_random_vec
        # 将向量添加到ndarray中
        arr[i] = np.random.uniform(low=0,high=3,size=3)
    arr_perturbtion=copy.deepcopy(arr)
    for i in range(len(arr_perturbtion)):
        arr_perturbtion[i]=RDmechanism(arr_perturbtion[i], 0.5)
    sum_vector = np.sum(arr, axis=0)
    sum_vector_perturbation=np.sum(arr_perturbtion, axis=0)
    print(sum_vector)
    print(sum_vector_perturbation)