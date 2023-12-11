#https://www.secretflow.org.cn/docs/heu/latest/zh-Hans/getting_started/installation#id3
from heu import numpy as hnp
from heu import phe

kit = hnp.setup(phe.SchemaType.ZPaillier, 2048)
encryptor = kit.encryptor()
decryptor = kit.decryptor()
evaluator = kit.evaluator()

encoder = kit.float_encoder(1000000000)

harr = kit.array([[1, 2, 3], [4, 5, 6]], encoder)
print('原文:', [[1, 2, 3], [4, 5, 6]])
#支持List，嵌套的 Tuple 以及 numpy.ndarray
print('编码后:', harr)
ct_arr = encryptor.encrypt(harr)
pt_arr = decryptor.decrypt(ct_arr)
nparr = harr.to_numpy(encoder)

c2 = evaluator.add(ct_arr, pt_arr)
print('明文加密文:', decryptor.decrypt(c2).to_numpy(encoder))

c2 = evaluator.sub(ct_arr, pt_arr)
print('明文减密文:',decryptor.decrypt(c2).to_numpy(encoder))

"""
#这个地方有点奇怪只支持明文乘密文，所以我们可以用以下两个方法解决
1)保证乘法的其中一个乘数的 scale 为 1
arr1 = kit.array([1.4], phe.FloatEncoderParams())
arr2 = kit.array([2])
print(evaluator.mul(arr1, arr2).to_numpy(edr))  # [2.8]

2)在结果转为明文或原文后手动除以 scale
scale = 100
edr = phe.FloatEncoder(phe.SchemaType.ZPaillier, scale)
res = evaluator.mul(hnp.array([1.4], edr), hnp.array([2.5], edr))
print(res.to_numpy(edr) / scale)  # [3.5]
# or
print(res.to_numpy(kit.float_encoder(scale**2)))  # [3.5]
"""
c2 = evaluator.mul(ct_arr, kit.array([2]))
print('明文乘密文', decryptor.decrypt(c2).to_numpy(encoder))