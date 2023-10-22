# FL-DP
environment python3.8

判断是否支持GPU加速可以运行 python3 cuda_test.py

<<<<<<< HEAD
### DP and cytro
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Efficient Differentially Private Secure Aggregation for Federated Learning via Hardness of Learning with Errors | University of Vermont | CCS/2021 | 该文章在原来的用同态加密进行联邦聚合的基础上，刻画了噪声的DP衡量，因为之前用的LWE天然存在噪声，所以该文章把这个噪声用DP量化出来。同时，也可以理解成其用同态加密的方法将原来的LDP加噪变成CDP的加噪方式，类似shuffle的理念。[【vedio】](https://www.bilibili.com/video/BV1fR4y1D7dU/?spm_id_from=333.999.list.card_archive.click&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)| 
| Private, Efficient, and Accurate: Protecting Models Trained by Multi-party Learning with Differential Privacy | Fudan University | SP/2022 | 核心在于利用了多方安全的秘密分享构建出一个虚拟的联邦中心方，使的不同客户端的样本数据可以进行秘密分享后“集中”式训练，然后在再训练的梯度上加DP，以此满足模型的差分隐私。相较于之前的模型，该模型不用LDP,也不用shuffle，将所以客户端数据整合成集中式变成CDP的形式（这样可以用更小的eps，即不造成更大的精度损失），并且没有可信第三方。| 
# Code
# FL-DP
# FL-DP
# Awesome-Differential-Privacy-and-Meachine-Learning-master
=======
python添加__init__.py还是找不到包可以参考https://blog.csdn.net/weixin_39663060/article/details/128015541
>>>>>>> 72416828af8ba2edb632c2a8f08b898f05b14c73
