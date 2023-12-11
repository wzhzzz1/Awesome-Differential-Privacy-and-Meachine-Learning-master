import pickle
from PIL import Image
import matplotlib.pyplot as plt
# 假设你想加载索引为0的文件

# 加载.pkl文件
history=[]
for i in range(9):
    with open(f"./attack_image/e2/attack_result_eps8_cifar10_{i}.pkl", 'rb') as file:
        loaded_image = pickle.load(file)
        history.append(loaded_image)
with open(f"./attack_image/e2/attack_result_cifar10_lu.pkl", 'rb') as file:
    loaded_image = pickle.load(file)
    history.append(loaded_image)
for i in range(9):
    with open(f"./attack_image/e2/attack_result_eps8_cifar10_PDPFL_{i}.pkl", 'rb') as file:
        loaded_image = pickle.load(file)
        history.append(loaded_image)
with open(f"./attack_image/e2/attack_result_cifar10_lu.pkl", 'rb') as file:
    loaded_image = pickle.load(file)
    history.append(loaded_image)
for i in range(9):
    with open(f"./attack_image/e2/attack_result_eps20_cifar10_{i}.pkl", 'rb') as file:
        loaded_image = pickle.load(file)
        history.append(loaded_image)
with open(f"./attack_image/e2/attack_result_cifar10_lu.pkl", 'rb') as file:
    loaded_image = pickle.load(file)
    history.append(loaded_image)
for i in range(9):
    with open(f"./attack_image/e2/attack_result_eps20_cifar10_PDPFL_{i}.pkl", 'rb') as file:
        loaded_image = pickle.load(file)
        history.append(loaded_image)
with open(f"./attack_image/e2/attack_result_cifar10_lu.pkl", 'rb') as file:
    loaded_image = pickle.load(file)
    history.append(loaded_image)
print(len(history))
plt.figure(figsize=(12, 8),dpi=300)
for i in range(40):
    plt.subplot(4, 10, i + 1)
    #plt.imshow(history[i])
    plt.imshow(history[i], cmap='gray')#灰度图像
    if i==9:
        plt.title("private data")
    elif i<9:
        plt.title("iter=%d" % (i * 10))
    plt.axis('off')
plt.subplots_adjust(wspace=0.1, hspace=-0.65)
plt.figtext(0.06,  0.762, r"attack iteration", ha='center', va='center')
plt.figtext(0.06,  0.69, r"Fedavg($\epsilon$=8)", ha='center', va='center')
plt.figtext(0.06,  0.56, r"PD-LDPFL($\epsilon$=8)", ha='center', va='center')
plt.figtext(0.06,  0.425, r"Fedavg($\epsilon$=20)", ha='center', va='center')
plt.figtext(0.06,  0.298, r"PD-LDPFL($\epsilon$=20)", ha='center', va='center')
#plt.tight_layout()
plt.savefig("./attack_image/attack_result.png")
plt.show()
