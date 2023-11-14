'''
import matplotlib.pyplot as plt

# 第一部分数据
x1 = [4, 6, 8, 10]
y1 = [0.054, 0.028, 0.0165, 0.0162]

# 第二部分数据
x2 = [4, 6, 8, 10]
y2 = [ 0.0155, 0.0155, 0.0155, 0.0155]

# 绘制曲线图
plt.figure(figsize=(8, 5))  # 可选：设置图形大小
plt.plot(x1, y1, label='LDP')
plt.plot(x2, y2, label='noise-free')

plt.xlabel('epsilon')
plt.ylabel('Heterogeneity(sum of loss)')
plt.suptitle('(a)IID', x=0.5, y=0.08, ha='center', fontsize=14)
plt.legend()  # 添加图例
plt.grid(True)  # 添加网格线
plt.subplots_adjust(bottom=0.2)
plt.show()
'''
import matplotlib.pyplot as plt

# 第一批数据
x1_part1 = [4, 6, 8, 10]
y1_part1 = [0.044, 0.028, 0.0195, 0.0162]

x1_part2 = [4, 6, 8, 10]
y1_part2 = [0.0155, 0.0155, 0.0155, 0.0155]

# 第二批数据
x2_part1 = [4, 6, 8, 10]
y2_part1 = [0.294, 0.177, 0.148, 0.126]

x2_part2 = [4, 6, 8, 10]
y2_part2 = [0.12, 0.12, 0.12, 0.12]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# 第一个子图
axs[0].plot(x1_part1, y1_part1, marker='s', label='LDP',color='#B22222', markersize=8,linestyle='--',linewidth=2)
axs[0].plot(x1_part2, y1_part2, marker='*', label='Noise-Free',color='black', markersize=8,linewidth=2)

axs[0].legend()
axs[0].grid(True,linestyle='--')
axs[0].set_xlabel('epsilon')
axs[0].set_ylabel('Heterogeneity(sum of loss)')


# 第二个子图
axs[1].plot(x2_part1, y2_part1, marker='s', label='LDP',color='#B22222', markersize=8,linestyle='--',linewidth=2)
axs[1].plot(x2_part2, y2_part2, marker='*', label='Noise-Free',color='black', markersize=8,linewidth=2)
axs[1].set_xlabel('epsilon')
axs[1].set_ylabel('Heterogeneity(sum of loss)')

axs[1].legend()
axs[1].grid(True,linestyle='--')
# 调整布局
plt.tight_layout()

plt.sca(axs[0])
plt.xticks(x1_part1 + x1_part2)

plt.sca(axs[1])
plt.xticks(x2_part1 + x2_part2)
# 显示图形
fig.text(0.289, 0.009, '(a)IID', ha='center')
fig.text(0.782, 0.009, '(b)NO-IID', ha='center')
plt.subplots_adjust(bottom=0.13)
plt.show()
fig.savefig('my_plot.png', dpi=300)





























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
'''


