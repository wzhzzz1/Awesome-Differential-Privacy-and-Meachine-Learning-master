'''
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(1, 11)  # 从1到10
y = np.arange(1, 11)  # 从1到10
z = np.zeros(10)

dx = np.ones(10)
dy = np.ones(10)
dz = np.arange(1, 11)  # 从1到10，作为柱子的高度

ax.bar3d(x, y, z, dx, dy, dz, color='skyblue')

ax.set_xlabel('client')
ax.set_ylabel('label')
ax.set_zlabel('numbers')

# 设置X和Y轴的刻度
ax.set_xticks(x)
ax.set_yticks(y)

plt.show()
'''

'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data = [
    [691, 675, 541, 568, 591, 549, 666, 518, 595, 590],
    [554, 628, 620, 615, 634, 606, 562, 709, 589, 526],
    [560, 614, 595, 597, 601, 522, 537, 556, 576, 560],
    [528, 769, 555, 634, 471, 508, 639, 684, 554, 533],
    [700, 751, 609, 611, 596, 524, 641, 554, 559, 619],
    [546, 641, 569, 563, 627, 545, 542, 663, 585, 580],
    [680, 668, 613, 649, 569, 518, 616, 680, 605, 729],
    [477, 724, 645, 673, 570, 612, 525, 589, 675, 484],
    [659, 641, 625, 717, 594, 485, 617, 680, 571, 752],
    [528, 631, 586, 504, 589, 552, 573, 632, 542, 576]
]

fig = plt.figure(dpi=400)
ax = fig.add_subplot(111, projection='3d')

x_data, y_data = np.meshgrid(np.arange(1, 11), np.arange(1, 11))
x = x_data.flatten()
y = y_data.flatten()
z = np.zeros_like(x)

dx = dy = 0.24  # 调整柱子宽度
dz = np.array(data).flatten()

# 定义颜色映射
colors = plt.cm.get_cmap('tab10')  # 使用tab10颜色映射，其中每个标签对应不同颜色

# 为每个柱子分配对应的颜色
colors_array = colors(y - 1)

ax.bar3d(x, y, z, dx, dy, dz, color=colors_array)

ax.set_xlabel('Client')
ax.set_ylabel('Label')
ax.set_zlabel('Number')
ax.set_xticks(x)
ax.set_yticks(y)
plt.show()
'''
import matplotlib.pyplot as plt
import numpy as np

data1 = [
    [691, 675, 541, 568, 591, 549, 666, 518, 595, 590],
    [554, 628, 620, 615, 634, 606, 562, 709, 589, 526],
    [560, 614, 595, 597, 601, 522, 537, 556, 576, 560],
    [528, 769, 555, 634, 471, 508, 639, 684, 554, 533],
    [700, 751, 609, 611, 596, 524, 641, 554, 559, 619],
    [546, 641, 569, 563, 627, 545, 542, 663, 585, 580],
    [680, 668, 613, 649, 569, 518, 616, 680, 605, 729],
    [477, 724, 645, 673, 570, 612, 525, 589, 675, 484],
    [659, 641, 625, 717, 594, 485, 617, 680, 571, 752],
    [528, 631, 586, 504, 589, 552, 573, 632, 542, 576]
]

data2 = [
    [1288, 940, 157, 294, 170, 0, 464, 383, 372, 0],
    [0, 100, 22, 1548, 455, 544, 23, 75, 606, 301],
    [159, 1466, 3658, 1057, 12, 29, 531, 44, 5, 2051],
    [257, 7, 166, 12, 189, 2239, 1386, 4247, 514, 1450],
    [1166, 30, 1, 95, 155, 5, 1434, 68, 881, 114],
    [1302, 9, 90, 1299, 1631, 724, 1987, 474, 83, 286],
    [309, 3079, 719, 435, 0, 596, 5, 5, 45, 1593],
    [6, 561, 21, 911, 99, 1, 3, 449, 335, 117],
    [1290, 549, 1118, 435, 1432, 0, 77, 117, 812, 23],
    [146, 1, 6, 45, 1699, 1283, 8, 403, 2198, 14]
]

fig = plt.figure(figsize=(20, 9))
fig = plt.figure(dpi=300)
ax1 = fig.add_subplot(121, projection='3d')
x_data1, y_data1 = np.meshgrid(np.arange(1, 11), np.arange(1, 11))
x1 = x_data1.flatten()
y1 = y_data1.flatten()
z1 = np.zeros_like(x1)
dx1 = dy1 = 0.25  # 柱子宽度调整
dz1 = np.array(data1).flatten()
colors1 = plt.cm.get_cmap('tab10')
colors_array1 = colors1(y1 - 1)
ax1.bar3d(x1, y1, z1, dx1, dy1, dz1, color=colors_array1)
ax1.set_xlabel('Client ID', fontsize=6, labelpad=-5)
ax1.set_ylabel('Label ID', fontsize=6, labelpad=-5)
ax1.set_zlabel('Number', fontsize=6, labelpad=-5)
ax1.set_xticks(x1)
ax1.set_yticks(y1)
ax1.tick_params(axis='x', pad=-1,labelsize=5)
ax1.tick_params(axis='y', pad=-1,labelsize=5)
ax1.tick_params(axis='z', pad=-1,labelsize=5)

itle_text = ax1.text2D(0.5, -0.1, '(a) IID', transform=ax1.transAxes, ha='center', fontsize=6, va='top')
ax2 = fig.add_subplot(122, projection='3d')
x_data2, y_data2 = np.meshgrid(np.arange(1, 11), np.arange(1, 11))
x2 = x_data2.flatten()
y2 = y_data2.flatten()
z2 = np.zeros_like(x2)
dx2 = dy2 = 0.25
dz2 = np.array(data2).flatten()
colors2 = plt.cm.get_cmap('tab10')
colors_array2 = colors2(y2 - 1)
ax2.bar3d(x2, y2, z2, dx2, dy2, dz2, color=colors_array2)
ax2.set_xlabel('Client ID', fontsize=6, labelpad=-5)
ax2.set_ylabel('Label ID', fontsize=6, labelpad=-5)
ax2.set_zlabel('Number', fontsize=6, labelpad=-5)
ax2.set_xticks(x2)
ax2.set_yticks(y2)
ax2.tick_params(axis='x', pad=-1,labelsize=5)
ax2.tick_params(axis='y', pad=-1,labelsize=5)
ax2.tick_params(axis='z', pad=-1,labelsize=5)
itle_text = ax2.text2D(0.5, -0.1, '(b) NO-IID', transform=ax2.transAxes, ha='center', fontsize=6, va='top')
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, right=0.93,top=1.25)
plt.show()
plt.show()



