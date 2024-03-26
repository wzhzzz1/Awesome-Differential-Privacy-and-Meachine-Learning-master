'''
#论文图9

import matplotlib.pyplot as plt

# 第一批数据
x1_1 = [8,10,12,14]
y1_1 = [2.872	,2.478	,2.19	,1.978]

x1_2 = [8,10,12,14]
y1_2 = [2.716	,2.375	,2.105	,1.92]


# 第二批数据
x1_3 = [8,10,12,14]
y1_3 = [2.598	,2.254	,2.04	,1.89]



x2_1 = [8,10,12,14]
y2_1 = [1.72	,1.52	,1.325	,1.198]

x2_2 = [8,10,12,14]
y2_2 = [1.63	,1.43	,1.258	,1.147]


# 第二批数据
x2_3 = [8,10,12,14]
y2_3 = [1.55	,1.35,	1.204	,1.125]




x3_1 = [8,10,12,14]
y3_1 = [0.852	,0.621	,0.486,	0.397]

x3_2 = [8,10,12,14]
y3_2 = [0.721	,0.533	,0.428	,0.345]


# 第二批数据
x3_3 = [8,10,12,14]
y3_3 = [0.664	,0.492	,0.395	,0.333]



x4_1 = [8,10,12,14]
y4_1 = [0.218	,0.154	,0.112	,0.088445695]

x4_2 = [8,10,12,14]
y4_2 = [0.202	,0.145	,0.103	,0.082]


# 第二批数据
x4_3 = [8,10,12,14]
y4_3 = [0.189	,0.138	,0.098	,0.078]





fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5),dpi=300)

# 第一个子图
axs[0].plot(x1_1, y1_1, marker='s', label='Fedavg',color='#B22222', markersize=3,linestyle='--',linewidth=0.75)
axs[0].plot(x1_2, y1_2, marker='*', label='Privatefl',color='c', markersize=3,linestyle='-.',linewidth=0.75)
axs[0].plot(x1_3, y1_3, marker='+', label='PD-LDPFL',color='black', markersize=3,linestyle='-',linewidth=0.75)


axs[0].legend(loc='upper right')
axs[0].grid(True,linestyle='--')
axs[0].set_xlabel('Epsilon')
axs[0].set_ylabel('Heterogeneity(sum of loss)')
# 注释（标注）


# 第二个子图
axs[1].plot(x2_1, y2_1, marker='s', label='Fedavg',color='#B22222', markersize=3,linestyle='--',linewidth=0.75)
axs[1].plot(x2_2, y2_2, marker='*', label='Privatefl',color='c', markersize=3,linestyle='-.',linewidth=0.75)
axs[1].plot(x2_3, y2_3, marker='+', label='PD-LDPFL',color='black', markersize=3,linestyle='-',linewidth=0.75)


axs[1].legend(loc='upper right')
axs[1].grid(True,linestyle='--')
axs[1].set_xlabel('Epsilon')
axs[1].set_ylabel('Heterogeneity(sum of loss)')


# 第三个子图
axs[2].plot(x3_1, y3_1, marker='s', label='Fedavg',color='#B22222', markersize=3,linestyle='--',linewidth=0.75)
axs[2].plot(x3_2, y3_2, marker='*', label='Privatefl',color='c', markersize=3,linestyle='-.',linewidth=0.75)
axs[2].plot(x3_3, y3_3, marker='+', label='PD-LDPFL',color='black', markersize=3,linestyle='-',linewidth=0.75)


axs[2].legend(loc='upper right')
axs[2].grid(True,linestyle='--')
axs[2].set_xlabel('Epsilon')
axs[2].set_ylabel('Heterogeneity(sum of loss)')


# 第四个子图
axs[3].plot(x4_1, y4_1, marker='s', label='Fedavg',color='#B22222', markersize=3,linestyle='--',linewidth=0.75)
axs[3].plot(x4_2, y4_2, marker='*', label='Privatefl',color='c', markersize=3,linestyle='-.',linewidth=0.75)
axs[3].plot(x4_3, y4_3, marker='+', label='PD-LDPFL',color='black', markersize=3,linestyle='-',linewidth=0.75)


axs[3].legend(loc='upper right')
axs[3].grid(True,linestyle='--')
axs[3].set_xlabel('Epsilon')
axs[3].set_ylabel('Heterogeneity(sum of loss)')



axs[0].set_yticks(range(1, 4, 1))  # 设置 y 轴刻度为 10 的倍数，范围为 10 到 100

# 调整布局
axs[1].set_yticks(range(1, 4, 1))  # 设置 y 轴刻度为 10 的倍数，范围为 10 到 100

plt.tight_layout()




# 显示图形
fig.text(0.134, 0.009, r'(a) $\beta$ = 0.5', ha='center')
fig.text(0.385, 0.009, r'(b) $\beta$ = 1', ha='center')
fig.text(0.635, 0.009, r'(c) $\beta$ = 5', ha='center')
fig.text(0.885, 0.009, r'(d) $\beta$ = 100', ha='center')
plt.subplots_adjust(bottom=0.128)
plt.show()
fig.savefig('my_plot.png', dpi=1200)
'''





'''大论文第四章图
import matplotlib.pyplot as plt
import numpy as np

# 防止画图中的乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置 柱状图 有几组
size = 4

x = np.arange(size)

# 有 a/b 两种类型的数据，n 设置为 2 0.6 不需要改，n 设置为要对比机组
total_width, n = 0.6, 2
# 每种类型的柱状图宽度，不需要改
width = total_width / n

# list1 代表样本 a，list2 代表样本 b
list1 = [0.103, 0.103, 0.103, 0.103]
list2 = [0.28, 0.175, 0.148, 0.125]

x = x - (total_width - width) / 2

# 画柱状图，并设置阴影效果
plt.bar(x, list1, width=width, label="Noise-Free", color='#0066cc', edgecolor='grey', linewidth=0.5,alpha=0.7)
plt.bar(x + width, list2, width=width, label="LDP", color='#9ACD32', edgecolor='grey', linewidth=0.5,alpha=0.7)

plt.xticks(np.arange(4), ('4', '6', '8', '10'))
plt.yticks([])  # 隐藏 y 轴坐标

# 显示图例,font1 为了解决图例中文显示为方框的问题
font1 = {'family': 'SimHei', 'weight': 'normal'}
plt.legend(loc='upper right', prop=font1)
plt.xlabel("隐私预算")
plt.ylabel("异构性(损失方差累计和)")

# 保存频率图，当然也可以保存矢量图  svg 的格式，然后用 origin 软件转换之后，可以放入 word 写论文用
plt.savefig('pinlv.png', dpi=500)

# 显示柱状图
plt.show()
'''



'''
#大论文第三章实验图

import matplotlib.pyplot as plt

# 第一批数据
x1_1 = [i for i in range(1,51)]
y1_1 = [16.33, 19.93, 36.01, 39.26, 42.66, 44.62, 45.14, 48.03, 52.12, 56.38,
    58.18, 60.8, 62.99, 65.09, 68.91, 69.49, 71.02, 72.18, 72.09, 74.13,
    74.98, 75.15, 76.16, 76.67, 76.1, 77.52, 77.06, 78.43, 78.71, 78.07,
    79.15, 79.41, 80.17, 80.94, 81.65, 81.3, 81.01, 81.49, 81.31, 81.28,
    81.53, 81.68, 82.16, 82.02, 82.15, 81.56, 82.16, 81.91, 82.35, 82.81]

x1_2 = [i for i in range(1,51)]
y1_2 = [18.17, 19.22, 29.76, 31.91, 35.4, 39.27, 41.51, 43.62, 47.19, 50.09,
    53.02, 54.23, 55.97, 57.67, 58.73, 60.52, 61.76, 63.48, 64.72, 66.63,
    69.17, 69.4, 69.42, 69.38, 69.79, 71.58, 72.76, 71.53, 71.73, 72.19,
    73.13, 73.34, 73.76, 74.17, 75.16, 74.63, 75.17, 75.47, 76.01, 75.86,
    76.24, 77.34, 76.8, 76.25, 76.09, 76.32, 77.18, 77.49, 77.1, 77.48]


# 第二批数据
x1_3 = [i for i in range(1,51)]
y1_3 = [15.24, 18.78, 25.64, 29.28, 32.3, 34.5, 38.16, 40.49, 43.22, 47.15,
    49.82, 52.49, 53.16, 54, 55.87, 56.37, 58.4, 61.13, 62.78, 63.4,
    65.85, 65.32, 66.21, 67.77, 67.61, 68.47, 68.5, 67.9, 68.29, 69.11,
    71.84, 71.87, 71.75, 72.54, 72.58, 73.15, 72.95, 73.19, 73.39, 73.65,
    73.68, 74.3, 74.34, 74.23, 74.69, 74.86, 74.8, 75.29, 75.81, 75.63]

x1_4 = [i for i in range(1,51)]
y1_4 = [28.42, 51.03, 61.37, 64.73, 68.67, 72.61, 74.47, 76.88, 80.03, 81.26,
    82.66, 82.89, 84.46, 84.77, 86.47, 86.43, 87.15, 87.43, 87.92, 88.39,
    88.88, 89.67, 89.29, 89.38, 90.17, 89.92, 90.25, 90.45, 90.48, 90.65,
    90.82, 90.72, 91.34, 91.28, 91.13, 90.95, 91.04, 90.98, 91.18, 91.16,
    91.53, 91.74, 91.67, 91.86, 92.15, 92.32, 92.53, 92.44, 92.61, 92.87]

x2_1 = [i for i in range(1,51)]
y2_1 = [28.3, 40.94, 51.36, 54.78, 57.48, 61.52, 64.14, 67.96, 68.52, 71.44,
    73.53, 76.09, 76.92, 78.06, 77.91, 78.58, 79.85, 79.06, 80.1, 80.37,
    81.2, 81.44, 82.51, 82.91, 82.26, 82.76, 83.33, 82.93, 84.2, 84.18,
    84.54, 84.27, 83.9, 83.96, 84.36, 83.77, 84.25, 85.41, 85.18, 86.03,
    85.87, 85.69, 86.07, 86.27, 86.36, 86.59, 86.64, 87.01, 86.25, 85.28]

x2_2 = [i for i in range(1,51)]
y2_2 = [31.32, 38.19, 42.93, 47.35, 52.61, 54.79, 56.98, 61.0, 63.58, 67.49,
    69.62, 71.68, 72.26, 72.4, 73.47, 73.89, 74.5, 75.08, 76.33, 77.32,
    77.16, 77.2, 77.52, 77.98, 79.16, 78.53, 79.21, 80.56, 79.71, 80.44,
    80.65, 81.06, 80.7, 80.83, 81.08, 81.74, 81.93, 81.6, 81.76, 82.01,
    82.29, 82.71, 83.27, 82.93, 83.17, 83.25, 83.48, 83.56, 83.55, 83.61]


# 第二批数据
x2_3 = [i for i in range(1,51)]
y2_3 = [25.28, 29.87, 35.81, 37.28, 46.39, 48.22, 52.65, 55.27, 59.56, 62.15,
    65.44, 68.97, 68.82, 68.35, 70.64, 71.68, 72.95, 73.51, 74.1, 75.02,
    76.38, 75.73, 76.16, 76.28, 76.87, 77.27, 77.43, 78.01, 78.21, 79.08,
    78.36, 78.61, 78.86, 79.13, 78.88, 78.71, 79.52, 79.22, 80.45, 80.75,
    79.95, 81.28, 81.4, 81.65, 81.9, 82.07, 81.23, 82.06, 82.63, 82.8]

x2_4 = [i for i in range(1,51)]
y2_4 = [33.08, 56.27, 65.64, 68.52, 71.53, 76.66, 78.66, 80.64, 82.45, 82.99,
    84.72, 84.95, 85.71, 86.58, 87.22, 87.88, 87.85, 88.43, 88.91, 89.36,
    89.67, 90.41, 90.29, 90.23, 90.87, 90.28, 90.84, 91.1, 90.98, 91.18,
    91.47, 91.5, 91.25, 92.05, 91.92, 91.89, 91.57, 91.84, 91.6, 91.65,
    92.07, 91.97, 91.92, 92.24, 92.47, 92.55, 92.81, 93.2, 93.58, 93.75]


x3_1 = [i for i in range(1,51)]
y3_1 = [36.76, 45.62, 49.9, 52.05, 55.7, 62.89, 67.23, 68.82, 71.21, 73.08,
    74.89, 74.26, 75.33, 78.37, 79.18, 78.15, 80.59, 81.12, 80.92, 82.05,
    82.5, 83.68, 82.91, 83.43, 83.32, 84.48, 84.12, 84.98, 85.21, 86.08,
    86.1, 85.61, 86.11, 87.07, 86.36, 86.91, 87.19, 86.92, 87.26, 87.91,
    88.04, 87.84, 88.36, 88.11, 87.44, 88.68, 88.54, 88.82, 88.93, 89.25]

x3_2 = [i for i in range(1,51)]
y3_2 = [34.7, 42.53, 45.2, 47.5, 52.09, 57.21, 58.79, 62.65, 63.55, 67.19,
    70.12, 71.63, 72.1, 73.17, 74.52, 74.38, 75.87, 76.67, 76.32, 77.68,
    78.7, 79.31, 80.71, 81.36, 81.67, 82.58, 82.38, 82.71, 83.42, 83.61,
    83.9, 83.38, 82.87, 84.11, 84.32, 83.49, 84.62, 84.51, 85.05, 83.47,
    85.46, 84.4, 85.38, 85.58, 84.83, 85.38, 86.09, 86.44, 86.78, 87.23]


# 第二批数据
x3_3 = [i for i in range(1,51)]
y3_3 = [31.26, 38.18, 43.31, 44.17, 45.02, 51.17, 54.87, 58.96, 60.9, 63.86,
    64.15, 68.01, 69.25, 70.52, 71.66, 71.48, 72.16, 73.46, 73.17, 74.24,
    76.52, 75.93, 76.04, 77.22, 77.35, 78.66, 78.26, 79.71, 80.05, 80.6,
    81.22, 81.03, 80.46, 81.11, 81.47, 80.97, 81.67, 81.22, 81.66, 81.16,
    82.01, 82.06, 82.18, 82.66, 83.27, 83.01, 83.85, 83.47, 84.1, 84.59]

x3_4 = [i for i in range(1,51)]
y3_4 = [34.79, 39.35, 45.56, 53.2, 63.82, 64.34, 67.14, 66.7, 70.29, 68.74,
    71.93, 72.77, 73.66, 74.26, 74.76, 75.32, 74.81, 75.95, 76.68, 77.29,
    77.77, 77.58, 78.5, 78.87, 78.5, 78.86, 78.96, 78.76, 79.34, 79.3,
    79.6, 79.89, 80.6, 80.98, 81.11, 81.39, 80.97, 81.13, 81.47, 81.87,
    81.53, 81.74, 81.45, 81.76, 81.58, 81.98, 82.19, 82.39, 82.73, 82.6]

x4_1 = [i for i in range(1,51)]
y4_1 = [38, 51.38, 54.02, 61.89, 67.85, 68.09, 72.33, 75.26, 78.37, 79.56,
    80.31, 82.07, 83.51, 84.09, 84.74, 84.22, 86.3, 85.89, 87.3, 87.64,
    87.97, 87.24, 88.14, 87.61, 88.36, 89.04, 88.82, 89.36, 89.13, 89.53,
    88.67, 89.26, 89.84, 88.99, 90.02, 90.17, 90.12, 90.42, 90.11, 90.59,
    90.69, 90.35, 90.81, 90.51, 91.01, 91.06, 90.55, 91.24, 91.32, 91.33]

x4_2 = [i for i in range(1,51)]
y4_2 = [34.78, 45.58, 52.31, 58.62, 60.37, 64.39, 70.33, 71.36, 73.63, 76.44,
    78.96, 79.52, 80.15, 80.54, 81.06, 83.26, 82.26, 84.7, 84.96, 85.15,
    84.55, 84.19, 84.86, 85.53, 85.16, 86.21, 86.38, 86.78, 87.42, 87.19,
    87.28, 87.42, 87.11, 87.41, 88.07, 87.95, 88.29, 88.12, 88.43, 88.67,
    88.99, 89.15, 88.86, 88.42, 89.02, 89.15, 89.01, 89.33, 89.2, 89.53]


# 第二批数据
x4_3 = [i for i in range(1,51)]
y4_3 = [32.37, 40.56, 44.47, 52.95, 55.19, 61.5, 65.73, 68.11, 71.2, 71.01, 72.93,
    74.67, 76.56, 77.93, 77.01, 78.87, 78.73, 79.04, 81.92, 80.81, 81.23, 82.15,
    82.84, 83.24, 82.97, 83.06, 83.37, 83.86, 84.34, 85.45, 85.92, 85.52, 85.64,
    85.99, 86.21, 85.46, 85.55, 86.15, 85.89, 86.06, 86.76, 87.34, 87.22, 87.38,
    86.42, 87.19, 87.28, 87.74, 87.23, 87.54]


x4_4 = [i for i in range(1,51)]
y4_4 = [36.42, 40.95, 47.18, 54.9, 65.37, 65.96, 69.21, 67.8, 71.91, 70.32,
    73.05, 74.19, 75.38, 76.1, 76.24, 76.8, 76.49, 77.11, 78.02, 78.4,
    79.27, 79.15, 80.15, 80.27, 80.3, 80.21, 80.64, 80.53, 80.92, 81.13,
    81.46, 81.17, 81.49, 81.92, 82.35, 82.71, 82.93, 82.73, 83.12, 83.48,
    83.29, 83.5, 83.71, 83.62, 83.29, 83.85, 83.58, 83.81, 83.92, 84.35]


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 16),dpi=300)

# 第一个子图
axs[0][0].plot(x1_1, y1_1, marker='s', label='Fedshuffle',color='#B22222', markersize=7,linestyle='--',linewidth=2)
axs[0][0].plot(x1_2, y1_2, marker='*', label='PM-FedAvg',color='c', markersize=7,linestyle='-.',linewidth=2)
axs[0][0].plot(x1_3, y1_3, marker='d', label='Gaussian-FedAvg',color='orange', markersize=7,linestyle=':',linewidth=2)
#axs[0][0].plot(x1_4, y1_4, marker='>', label='Liner&Liner(PD-LDPFL)',color='cornflowerblue', markersize=5,linestyle='-',linewidth=2)

axs[0][0].legend(loc='lower right', fontsize=16)
axs[0][0].grid(True,linestyle='--')
axs[0][0].set_xlabel('Epoch', fontsize=16)
axs[0][0].set_ylabel('Accuracy(global model)', fontsize=16)
axs[0][0].text(0.5, -0.15, r'$\alpha$ = 0.1', ha='center', transform=axs[0][0].transAxes, fontsize=16)
# 注释（标注）

axs[0].annotate(
    text='92.87',  # 标注的内容
    xy=(50, 92.87),  # 标注的坐标点
    xytext=(41, 97.5),  # 标注的内容的坐标点
    # 箭头
    arrowprops={
             'width': 1,  # 箭头线的宽度
        'headwidth': 6,  # 箭头头部的宽度
        'facecolor': 'cornflowerblue' , # 箭头的背景颜色
        'edgecolor':  'cornflowerblue'

    }
)

# 第二个子图
axs[0][1].plot(x2_1, y2_1, marker='s', label='Fedshuffle',color='#B22222', markersize=7,linestyle='--',linewidth=2)
axs[0][1].plot(x2_2, y2_2, marker='*', label='PM-FedAvg',color='c', markersize=7,linestyle='-.',linewidth=2)
axs[0][1].plot(x2_3, y2_3, marker='d', label='Gaussian-FedAvg',color='orange', markersize=7,linestyle=':',linewidth=2)
#axs[0][1].plot(x2_4, y2_4, marker='>', label='Liner&Liner(PD-LDPFL)',color='cornflowerblue', markersize=5,linestyle='-',linewidth=2)

axs[0][1].legend(loc='lower right', fontsize=16)
axs[0][1].grid(True,linestyle='--')
axs[0][1].set_xlabel('Epoch', fontsize=16)
axs[0][1].set_ylabel('Accuracy(global model)', fontsize=16)
axs[0][1].text(0.5, -0.15, r'$\alpha$ = 1', ha='center', transform=axs[0][1].transAxes, fontsize=16)

axs[1].annotate(
    text='93.75',  # 标注的内容
    xy=(50, 93.75),  # 标注的坐标点
    xytext=(41, 97.5),  # 标注的内容的坐标点
    # 箭头
    arrowprops={
             'width': 1,  # 箭头线的宽度
        'headwidth': 6,  # 箭头头部的宽度
        'facecolor': 'cornflowerblue' , # 箭头的背景颜色
        'edgecolor':  'cornflowerblue'

    }
)

# 第三个子图
axs[1][0].plot(x3_1, y3_1, marker='s', label='Fedshuffle',color='#B22222', markersize=7,linestyle='--',linewidth=2)
axs[1][0].plot(x3_2, y3_2, marker='*', label='PM-FedAvg',color='c', markersize=7,linestyle='-.',linewidth=2)
axs[1][0].plot(x3_3, y3_3, marker='d', label='Gaussian-FedAvg',color='orange', markersize=7,linestyle=':',linewidth=2)
#axs[1][0].plot(x3_4, y3_4, marker='>', label='Liner&Liner(PD-LDPFL)',color='cornflowerblue', markersize=5,linestyle='--',linewidth=2)

axs[1][0].legend(loc='lower right', fontsize=16)
axs[1][0].grid(True,linestyle='--')
axs[1][0].set_xlabel('Epoch', fontsize=16)
axs[1][0].set_ylabel('Accuracy(global model)', fontsize=16)
axs[1][0].text(0.5, -0.15, r'$\alpha$ = 10', ha='center', transform=axs[1][0].transAxes, fontsize=16)

axs[2].annotate(
    text='82.6',  # 标注的内容
    xy=(50, 82.6),  # 标注的坐标点
    xytext=(42, 88),  # 标注的内容的坐标点
    # 箭头
    arrowprops={
             'width': 1,  # 箭头线的宽度
        'headwidth': 6,  # 箭头头部的宽度
        'facecolor': 'cornflowerblue' , # 箭头的背景颜色
        'edgecolor':  'cornflowerblue'

    }
)

# 第四个子图
axs[1][1].plot(x4_1, y4_1, marker='s', label='Fedshuffle',color='#B22222', markersize=7,linestyle='--',linewidth=2)
axs[1][1].plot(x4_2, y4_2, marker='*', label='PM-FedAvg',color='c', markersize=7,linestyle='-.',linewidth=2)
axs[1][1].plot(x4_3, y4_3, marker='d', label='Gaussian-FedAvg',color='orange', markersize=7,linestyle=':',linewidth=2)
#axs[1][1].plot(x4_4, y4_4, marker='>', label='Liner&Liner(PD-LDPFL)',color='cornflowerblue', markersize=5,linestyle='--',linewidth=2)

axs[1][1].legend(loc='lower right', fontsize=16)
axs[1][1].grid(True,linestyle='--')
axs[1][1].set_xlabel('Epoch', fontsize=16)
axs[1][1].set_ylabel('Accuracy(global model)', fontsize=16)
axs[1][1].text(0.5, -0.15, r'$\alpha$ = 100', ha='center', transform=axs[1][1].transAxes, fontsize=16)

axs[3].annotate(
    text='84.35',  # 标注的内容
    xy=(50, 84.35),  # 标注的坐标点
    xytext=(42, 90),  # 标注的内容的坐标点
    # 箭头
    arrowprops={
             'width': 1,  # 箭头线的宽度
        'headwidth': 6,  # 箭头头部的宽度
        'facecolor': 'cornflowerblue' , # 箭头的背景颜色
        'edgecolor':  'cornflowerblue'

    }
)

for i in axs:
    for ax in i:
        ax.set_yticks(range(10, 101, 20))  # 设置 y 轴刻度为 10 的倍数，范围为 10 到 100
        ax.set_ylim(20, 100)  # 设置 y 轴范围从 10 到 100
        ax.tick_params(axis='both', which='major', labelsize=16)
# 调整布局



# 显示图形

plt.subplots_adjust(bottom=0.128)
plt.show()
fig.savefig('my_plot.png', dpi=300)
'''


''''''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

# 设置中文字体，如果不需要中文，则可不写该语句
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 写入数据
x_data = ['0.25', '0.5', '0.75','1.0','1.25', '1.5', '1.75','2.0','random']
y_data = [19.67,88.51,93.83,94.49,94.29,93.75,94.26,93.94,84.92]

# 设置颜色映射


# 设置图像尺寸
plt.figure(figsize=(10, 8))  # 调整图像宽度为10，高度为8

# 利用bar()函数设置柱状图的参数，
bars = plt.bar(x_data, y_data, color='#9ACD32', edgecolor='grey', linewidth=0.5,alpha=0.7)

plt.xlabel('初始化方式', fontsize=16)  # x轴的标签
plt.ylabel('准确率', fontsize=16)  # y轴的标签
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# 创建一个映射对象
sm = ScalarMappable(cmap=plt.cm.Blues)
sm.set_array(y_data)

# 添加颜色条


plt.tight_layout()  # 调整布局，防止标签重叠
plt.show()



'''
#大论文图7

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 创建随机数据
data = [
    [15.1, 17.29, 13.5, 11.4, 9.8, 17.1, 14.8, 12.3, 11.7],
    [71.3, 81.7, 82.07, 83.51, 92.3, 82.7, 81.22, 79.6, 67.8],
    [73.9, 79.2, 82.43, 83.29, 92.34, 84.7, 81.54, 79.8, 76.7],
    [76.5, 81.2, 85.1, 87.34, 92.8, 86.29, 82.3, 81.9, 76.5],
    [78.43, 84.9, 85.82, 91.33, 94.5, 92.57, 86.1, 84.7, 78.2],
    [81.76, 82.53, 84.36, 90.47, 92.41, 89.25, 83.9, 81.2, 81.86],
    [84.4, 81.3, 82.75, 89.41, 92.12, 88.39, 85.7, 81.9, 81.0],
    [82.31, 80.62, 84.29, 88.53, 91.48, 89.18, 86.89, 81.38, 81.4],
    [73.16, 74.57, 83.69, 88.4, 91.51, 89.44, 86.99, 77.66, 69.39]
]

# 绘制热图
data_array = np.array(data)

# 创建热图
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(data_array, annot=True, cmap='Greens', fmt='.1f')

# 设置刻度位置
x_ticks_positions = np.arange(0, len(data_array[0]), 1) + 0.5
x_ticks_labels = np.linspace(-1, 1, len(data_array[0]))
y_ticks_positions = np.arange(0, len(data_array), 1) + 0.5
y_ticks_labels = np.linspace(0, 2, len(data_array))

plt.xticks(x_ticks_positions, x_ticks_labels, fontsize=14)  # 设置 x 轴刻度的大小为 14
plt.yticks(y_ticks_positions, y_ticks_labels, fontsize=14)  # 设置 y 轴刻度的大小为 14

# 设置 x 轴和 y 轴标签的字体
font = FontProperties(family='Times New Roman')
plt.xlabel(r'Initial value of b', fontproperties=font, fontsize=16)  # 设置 x 轴标签的字体大小为 16
plt.ylabel(r'Initial value of a', fontproperties=font, fontsize=16)  # 设置 y 轴标签的字体大小为 16

# 去除刻度线
plt.tick_params(axis='both', which='both', length=0)

plt.tight_layout()
plt.show()
'''



'''
#大论文实验图

import matplotlib.pyplot as plt

# 第一部分数据
x1 = [i for i in range(1,51)]

y1 = [38.82, 51.69, 57.39, 60.11, 62.45, 67.26, 69.84, 72.08, 75.0, 76.42, 78.63, 78.96, 79.11, 81.29, 81.39, 81.23, 83.15, 82.72, 83.89, 84.5, 85.34, 85.19, 85.11, 85.79, 85.92, 85.88, 85.74, 86.53, 86.53, 87.28, 86.59, 87.81, 87.97, 88.81, 88.17, 88.24, 88.48, 88.65, 88.55, 88.61, 88.83, 88.96, 89.03, 89.23, 89.45, 89.15, 89.56, 89.51, 89.69, 89.72]
# 第二部分数据
x2 = [i for i in range(1,51)]

y2 = [48.36, 59.23, 63.87, 66.3, 75, 77.28, 79.21, 80.87, 82.24, 84, 84.91, 85.53, 86.85, 87.34, 88.38, 89.01, 89.26,90.18, 90.49, 90.31, 89.87, 89.74, 90.45, 90.52, 91.15, 90.97, 90.72, 91.25, 91.58, 91.04, 92.08, 91.51, 91.19,92.39, 92.32, 92.29, 93.11, 92.34, 92, 93.06, 93.35, 93.48, 93.78, 93.9, 94.14, 94.02, 93.85, 94.31, 94.52,94.38]

x3 =[i for i in range(1,51)]
y3 = [42.06, 56.85, 62.73, 67.59, 72.89, 76.74, 78.37, 80.77, 82.74, 82.47, 84.18, 84.47, 84.66, 85.46, 86.36, 87.46, 88.32, 88.26, 89.02, 89.67, 89.37, 89.68, 90.11, 90.31, 89.72, 90.64, 90.43, 90.83, 91.13, 91.13, 91.42, 91.26, 90.47, 90.85, 91.69, 91.97, 91.86, 91.75, 91.98, 92.35, 92.5, 92.12, 91.53, 92.8, 92.15, 92.76, 92.43, 92.54, 92.7, 92.92]

x4 =[i for i in range(1,51)]
y4 = [25.73, 26.91, 29.18, 32.64, 43.18, 54.25, 59.25, 61.07, 62.24, 62, 65.41, 70.94, 71.78, 70.16, 74.13, 73.19, 75.67, 79.36, 81.59, 82.61, 84.71, 85.57, 85.55, 85.03, 86.24, 85.22, 85.36, 85.47, 84.59, 85.19, 85.34, 86.97, 86.84, 86.6, 86.73, 86.92, 87.1, 87.26, 87.28, 87.44, 87.68, 87.69, 87.27, 87.18, 87.24, 87.79, 87.66, 87.54, 87.87, 87.75]
# 绘制曲线图

x5 = [i for i in range(1,51)]
y5 = [27.55, 54.05, 60.23, 64.54, 70.05, 74.34, 75.34, 77.91, 79.56, 80.91, 82.67, 83.15, 83.64, 84.92, 85.67, 86.04, 86.69, 87.05, 87.63, 87.42, 87.57, 88.72, 88.1, 88.56, 88.7, 88.13, 87.43, 87.57, 87.31, 88.66, 88.25, 88.89, 90.29, 89.36, 88.24, 89.25, 89.86, 90.21, 90.74, 90.67, 91.05, 91.94, 90.37, 91.65, 91.45, 91.81, 92.1, 91.11, 91.25, 91.56]

x6 = [i for i in range(1,51)]
y6 = [28.23, 29.67, 33.69, 42.18, 45.47, 52.7, 64.99, 68.61, 73.07, 71.51, 76.63, 80.6, 81.12, 81.85, 81.56, 81.39, 80.96, 81.26, 83.09, 82.71, 83.36, 83.28, 83.75, 85.45, 85.39, 85.44, 85.61, 86.5, 86.19, 86.58, 86.91, 87.17, 87.71, 87.86, 87.61, 87.97, 88.06, 88.69, 88.77, 89.6, 88.61, 88.68, 89.51, 89.56, 90.27, 90.66, 89.96, 90.45, 90.85, 90.25]
fig = plt.figure(figsize=(10, 8),dpi = 300)  # 可选：设置图形大小
plt.plot(x1, y1, marker='s', label='x&x',color='#B22222', markersize=7,linestyle='--',linewidth=2)
plt.plot(x2, y2, marker='*', label='liner&liner',color='cornflowerblue', markersize=7,linewidth=2,linestyle='--')
plt.plot(x3, y3, marker='p', label='liner&tanh',color='fuchsia', markersize=7,linestyle='--',linewidth=2)
plt.plot(x4, y4, marker='o', label='liner&sigmoid',color='silver', markersize=7,linestyle='--',linewidth=2)
plt.plot(x5, y5, marker='X', label='log&liner',color='orange', markersize=7,linestyle='--',linewidth=2)
plt.plot(x6, y6, marker='v', label='log&tanh',color='green', markersize=7,linestyle='--',linewidth=2)
plt.xlabel('迭代轮次', fontsize=16)
plt.ylabel('准确率', fontsize=16)
#plt.suptitle('(a)IID', x=0.5, y=0.08, ha='center', fontsize=14)
plt.legend(loc='lower right', fontsize=16)
plt.xticks(fontsize=14)  # 设置 x 轴刻度的大小为 14
plt.yticks(fontsize=14)  # 设置 y 轴刻度的大小为 14
plt.grid(True,linestyle='--')  # 添加网格线
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show()
fig.savefig('my_plot1.png', dpi=300)
'''


'''
#论文图3

import matplotlib.pyplot as plt

# 第一批数据
x1_part1 = [4, 6, 8, 10]
y1_part1 = [0.044, 0.028, 0.0195, 0.0169]

x1_part2 = [4, 6, 8, 10]
y1_part2 = [0.0155, 0.0155, 0.0155, 0.0155]

# 第二批数据
x2_part1 = [4, 6, 8, 10]
y2_part1 = [0.294, 0.177, 0.148, 0.126]

x2_part2 = [4, 6, 8, 10]
y2_part2 = [0.12, 0.12, 0.12, 0.12]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),dpi=300)

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


























'''
#论文图2

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


