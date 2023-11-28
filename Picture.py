'''
#论文图7

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
axs[0].plot(x1_1, y1_1, marker='s', label='x&x(Fedavg)',color='#B22222', markersize=3,linestyle='--',linewidth=0.75)
axs[0].plot(x1_2, y1_2, marker='*', label='Liner&x(Privatefl)',color='c', markersize=3,linestyle='-.',linewidth=0.75)
axs[0].plot(x1_3, y1_3, marker='+', label='Liner&Liner',color='black', markersize=3,linestyle='-',linewidth=0.75)


axs[0].legend(loc='upper right')
axs[0].grid(True,linestyle='--')
axs[0].set_xlabel('Epsilon')
axs[0].set_ylabel('Heterogeneity(sum of loss)')
# 注释（标注）


# 第二个子图
axs[1].plot(x2_1, y2_1, marker='s', label='x&x(Fedavg)',color='#B22222', markersize=3,linestyle='--',linewidth=0.75)
axs[1].plot(x2_2, y2_2, marker='*', label='Liner&x(Privatefl)',color='c', markersize=3,linestyle='-.',linewidth=0.75)
axs[1].plot(x2_3, y2_3, marker='+', label='Liner&Liner',color='black', markersize=3,linestyle='-',linewidth=0.75)


axs[1].legend(loc='upper right')
axs[1].grid(True,linestyle='--')
axs[1].set_xlabel('Epsilon')
axs[1].set_ylabel('Heterogeneity(sum of loss)')


# 第三个子图
axs[2].plot(x3_1, y3_1, marker='s', label='x&x(Fedavg)',color='#B22222', markersize=3,linestyle='--',linewidth=0.75)
axs[2].plot(x3_2, y3_2, marker='*', label='Liner&x(Privatefl)',color='c', markersize=3,linestyle='-.',linewidth=0.75)
axs[2].plot(x3_3, y3_3, marker='+', label='Liner&Liner',color='black', markersize=3,linestyle='-',linewidth=0.75)


axs[2].legend(loc='upper right')
axs[2].grid(True,linestyle='--')
axs[2].set_xlabel('Epsilon')
axs[2].set_ylabel('Heterogeneity(sum of loss)')


# 第四个子图
axs[3].plot(x4_1, y4_1, marker='s', label='x&x(Fedavg)',color='#B22222', markersize=3,linestyle='--',linewidth=0.75)
axs[3].plot(x4_2, y4_2, marker='*', label='Liner&x(Privatefl)',color='c', markersize=3,linestyle='-.',linewidth=0.75)
axs[3].plot(x4_3, y4_3, marker='+', label='Liner&Liner',color='black', markersize=3,linestyle='-',linewidth=0.75)


axs[3].legend(loc='upper right')
axs[3].grid(True,linestyle='--')
axs[3].set_xlabel('Epsilon')
axs[3].set_ylabel('Heterogeneity(sum of loss)')



axs[0].set_yticks(range(1, 4, 1))  # 设置 y 轴刻度为 10 的倍数，范围为 10 到 100
axs[0].set_ylim(1, 3)  # 设置 y 轴范围从 10 到 100
# 调整布局
axs[1].set_yticks(range(1, 4, 1))  # 设置 y 轴刻度为 10 的倍数，范围为 10 到 100
axs[1].set_ylim(1, 3)  # 设置 y 轴范围从 10 到 100
plt.tight_layout()




# 显示图形
fig.text(0.134, 0.009, r'(a) $\beta$ = 0.5', ha='center')
fig.text(0.385, 0.009, r'(a) $\beta$ = 1', ha='center')
fig.text(0.635, 0.009, r'(a) $\beta$ = 5', ha='center')
fig.text(0.885, 0.009, r'(a) $\beta$ = 100', ha='center')
plt.subplots_adjust(bottom=0.128)
plt.show()
fig.savefig('my_plot.png', dpi=1200)
'''


#论文图6

import matplotlib.pyplot as plt

# 第一批数据
x1_1 = [i for i in range(1,51)]
y1_1 = [28.3, 40.94, 51.36, 54.78, 57.48, 61.52, 64.14, 67.96, 68.52, 71.44,
    73.53, 76.09, 76.92, 78.06, 77.91, 78.58, 79.85, 79.06, 80.1, 80.37,
    81.2, 81.44, 82.51, 82.91, 82.26, 82.76, 83.33, 82.93, 84.2, 84.18,
    84.54, 84.27, 83.9, 83.96, 84.36, 83.77, 85.25, 85.41, 85.18, 86.03,
    85.87, 85.69, 86.07, 86.27, 86.36, 86.59, 86.64, 87.01, 86.25, 87.28]

x1_2 = [i for i in range(1,51)]
y1_2 = [43.17, 50.02, 58.86, 61.21, 65.4, 70.27, 74.51, 76.62, 79.19, 81.09,
    81.02, 82.23, 82.97, 83.67, 83.73, 84.52, 84.76, 85.48, 84.72, 85.63,
    86.17, 86.4, 86.42, 86.38, 86.79, 87.58, 87.76, 88.53, 88.73, 88.79,
    89.13, 89.14, 88.76, 88.77, 89.16, 89.63, 89.77, 89.47, 89.56, 89.86,
    90.64, 90.54, 89.8, 90.25, 90.09, 90.32, 90.18, 90.79, 90.1, 90.48]


# 第二批数据
x1_3 = [i for i in range(1,51)]
y1_3 = [26.24, 49.78, 58.64, 60.28, 62.3, 66.5, 71.16, 74.49, 76.22, 77.15,
    79.82, 80.49, 82.16, 82, 82.87, 83.37, 83.4, 84.13, 84.78, 85.4,
    84.85, 85.32, 86.21, 86.77, 87.61, 87.47, 86.5, 86.9, 88.29, 88.11,
    87.84, 87.87, 88.75, 88.54, 88.58, 89.15, 88.95, 89.19, 89.39, 89.65,
    89.68, 90.3, 90.34, 90.23, 90.69, 90.86, 89.8, 90.29, 90.81, 90.63]

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
    84.54, 84.27, 83.9, 83.96, 84.36, 83.77, 85.25, 85.41, 85.18, 86.03,
    85.87, 85.69, 86.07, 86.27, 86.36, 86.59, 86.64, 87.01, 86.25, 87.28]

x2_2 = [i for i in range(1,51)]
y2_2 = [43.32, 50.19, 58.93, 61.35, 65.61, 70.79, 74.98, 77.0, 79.58, 81.49,
    81.62, 82.68, 83.26, 84.4, 84.47, 84.89, 85.5, 86.08, 85.33, 86.32,
    86.86, 87.3, 87.33, 87.28, 87.66, 88.53, 88.61, 89.56, 89.61, 89.74,
    89.95, 90.06, 89.7, 89.83, 90.08, 90.74, 90.93, 90.6, 90.76, 91.01,
    91.29, 91.31, 91.27, 91.33, 91.17, 91.25, 91.48, 91.56, 91.55, 91.61]


# 第二批数据
x2_3 = [i for i in range(1,51)]
y2_3 = [35.1, 50.9, 60.15, 64.78, 67.88, 70.41, 74.6, 78.09, 79.11, 80.01,
    81.86, 82.35, 83.36, 83.91, 84.5, 84.8, 85.25, 85.55, 85.77, 86.59,
    86.54, 86.46, 87.12, 87.2, 88.17, 88.04, 88.02, 87.55, 88.85, 88.13,
    88.54, 88.19, 88.88, 89.19, 88.92, 89.37, 89.57, 89.77, 89.42, 89.63,
    89.75, 90.02, 90.48, 90.2, 90.63, 90.83, 89.85, 90.41, 90.88, 90.73]

x2_4 = [i for i in range(1,51)]
y2_4 = [33.08, 56.27, 65.64, 68.52, 71.53, 76.66, 78.66, 80.64, 82.45, 82.99,
    84.72, 84.95, 85.71, 86.58, 87.22, 87.88, 87.85, 88.43, 88.91, 89.36,
    89.67, 90.41, 90.29, 90.23, 90.87, 90.28, 90.84, 91.1, 90.98, 91.18,
    91.47, 91.5, 91.25, 92.05, 91.92, 91.89, 91.57, 91.84, 91.6, 91.65,
    92.07, 91.97, 91.92, 92.24, 92.47, 92.55, 92.81, 93.2, 93.58, 93.75]


x3_1 = [i for i in range(1,51)]
y3_1 = [30.43, 31.21, 38.18, 42.87, 47.61, 51.28, 52.39, 55.58, 55.66, 57.06,
    56.89, 57.75, 57.32, 58.96, 61.39, 60.77, 62.72, 64.5, 66.32, 68.44,
    69.73, 70.25, 70.57, 70.97, 71.25, 71.65, 71.91, 71.36, 71.9, 72.48,
    73.01, 73.58, 74.61, 74.98, 75.12, 75.51, 75.83, 76.29, 76.81, 77.02,
    77.23, 77.11, 77.47, 77.11, 77.27, 76.44, 77.62, 77.71, 77.91, 77.31]

x3_2 = [i for i in range(1,51)]
y3_2 = [34.7, 42.53, 48.2, 53.5, 55.09, 57.21, 58.79, 62.65, 63.55, 67.19,
    68.12, 70.63, 73.1, 71.17, 73.52, 73.38, 74.87, 74.07, 75.32, 75.68,
    75.7, 77.31, 74.71, 77.36, 75.67, 77.58, 76.38, 77.71, 77.42, 77.61,
    77.9, 77.38, 76.87, 78.11, 78.32, 77.49, 78.62, 78.51, 79.05, 77.47,
    79.46, 78.4, 79.38, 79.58, 78.83, 79.38, 80.09, 80.44, 80.78, 81.23]


# 第二批数据
x3_3 = [i for i in range(1,51)]
y3_3 = [36.26, 38.18, 46.31, 50.17, 54.02, 58.17, 62.87, 64.96, 67.9, 67.86,
    68.15, 71.01, 70.25, 71.03, 71.66, 73.98, 74.16, 74.46, 74.87, 74.24,
    76.52, 75.93, 77.04, 76.52, 77.55, 77.26, 77.46, 76.91, 78.05, 78,
    78.22, 78.03, 78.46, 79.11, 78.47, 78.97, 78.67, 79.22, 79.66, 79.16,
    80.01, 80.06, 80.18, 79.66, 80.27, 79.01, 79.85, 79.47, 80.1, 80.59]

x3_4 = [i for i in range(1,51)]
y3_4 = [34.79, 39.35, 45.56, 53.2, 63.82, 64.34, 67.14, 66.7, 70.29, 68.74,
    71.93, 72.77, 73.66, 74.26, 74.76, 75.32, 74.81, 75.95, 76.68, 77.29,
    77.77, 77.58, 78.5, 78.87, 78.5, 78.86, 78.96, 78.76, 79.34, 79.3,
    79.6, 79.89, 80.6, 80.98, 81.11, 81.39, 80.97, 81.13, 81.47, 81.87,
    81.53, 81.74, 81.45, 81.76, 81.58, 81.98, 82.19, 82.39, 82.73, 82.6]

x4_1 = [i for i in range(1,51)]
y4_1 = [30.43, 31.21, 38.18, 42.87, 47.61, 51.28, 52.39, 55.58, 55.66, 57.06,
    56.89, 57.75, 57.32, 58.96, 61.39, 60.77, 62.72, 64.5, 66.32, 68.44,
    69.73, 70.25, 70.57, 70.97, 71.25, 71.65, 71.91, 71.36, 71.9, 72.48,
    73.01, 73.58, 74.61, 74.98, 75.12, 75.51, 75.83, 76.29, 76.81, 77.02,
    77.23, 77.11, 77.47, 77.11, 77.27, 76.44, 77.62, 77.71, 77.91, 77.31]

x4_2 = [i for i in range(1,51)]
y4_2 = [34.78, 52.58, 53.31, 55.62, 55.37, 57.39, 59.33, 63.36, 64.63, 68.44,
    69.96, 71.52, 74.15, 73.04, 74.06, 75.26, 76.26, 75.7, 76.96, 77.15,
    77.55, 78.59, 76.56, 78.53, 78.25, 79.01, 78.38, 79.18, 79.18, 79.19,
    79.28, 79.28, 79.05, 79.61, 80.27, 79.85, 80.29, 80.08, 80.53, 80.17,
    80.69, 80.45, 80.6, 80.72, 80.8, 80.65, 80.83, 81.03, 81, 82.13]


# 第二批数据
x4_3 = [i for i in range(1,51)]
y4_3 = [37.37, 40.56, 47.47, 52.95, 57.19, 61.5, 65.73, 68.11, 71.2, 70.01, 69.93,
    71.67, 70.56, 70.93, 73.01, 73.87, 74.73, 75.04, 74.92, 74.81, 76.23, 76.15,
    76.84, 77.24, 77.97, 78.06, 78.37, 78.86, 78.34, 78.45, 78.92, 78.52, 78.64,
    78.99, 79.21, 79.46, 79.55, 79.35, 79.89, 80.06, 81.26, 81.34, 81.22, 81.98,
    82.42, 82.19, 82.28, 81.74, 82.23, 82.34]


x4_4 = [i for i in range(1,51)]
y4_4 = [36.42, 40.95, 47.18, 54.9, 65.37, 65.96, 69.21, 67.8, 71.91, 70.32,
    73.05, 74.19, 75.38, 76.1, 76.24, 76.8, 76.49, 77.11, 78.02, 78.4,
    79.27, 79.15, 80.15, 80.27, 80.3, 80.21, 80.64, 80.53, 80.92, 81.13,
    81.46, 81.17, 81.49, 81.92, 82.35, 82.71, 82.93, 82.73, 83.12, 83.48,
    83.29, 83.5, 83.71, 83.62, 83.29, 83.85, 83.58, 83.81, 83.92, 84.35]


fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5),dpi=300)

# 第一个子图
axs[0].plot(x1_1, y1_1, marker='s', label='x&x(Fedavg)',color='#B22222', markersize=3,linestyle='--',linewidth=0.75)
axs[0].plot(x1_2, y1_2, marker='*', label='Liner&x(Privatefl)',color='c', markersize=3,linestyle='-.',linewidth=0.75)
axs[0].plot(x1_3, y1_3, marker='+', label='x&Liner',color='black', markersize=3,linestyle=':',linewidth=0.75)
axs[0].plot(x1_4, y1_4, marker='>', label='Liner&Liner(PDPFL)',color='cornflowerblue', markersize=3,linestyle='-',linewidth=0.75)

axs[0].legend(loc='lower right')
axs[0].grid(True,linestyle='--')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy(global model)')
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
axs[1].plot(x2_1, y2_1, marker='s', label='x&x(Fedavg)',color='#B22222', markersize=3,linestyle='--',linewidth=0.75)
axs[1].plot(x2_2, y2_2, marker='*', label='Liner&x(Privatefl)',color='c', markersize=3,linestyle='-.',linewidth=0.75)
axs[1].plot(x2_3, y2_3, marker='+', label='x&Liner',color='black', markersize=3,linestyle=':',linewidth=0.75)
axs[1].plot(x2_4, y2_4, marker='>', label='Liner&Liner(PDPFL)',color='cornflowerblue', markersize=3,linestyle='-',linewidth=0.75)

axs[1].legend(loc='lower right')
axs[1].grid(True,linestyle='--')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel(' Max accuracy(local model)')

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
axs[2].plot(x3_1, y3_1, marker='s', label='x&x(Fedavg)',color='#B22222', markersize=3,linestyle='--',linewidth=0.75)
axs[2].plot(x3_2, y3_2, marker='*', label='Liner&x(Privatefl)',color='c', markersize=3,linestyle='-.',linewidth=0.75)
axs[2].plot(x3_3, y3_3, marker='+', label='x&Liner',color='black', markersize=3,linestyle=':',linewidth=0.75)
axs[2].plot(x3_4, y3_4, marker='>', label='Liner&Liner(PDPFL)',color='cornflowerblue', markersize=3,linestyle='--',linewidth=0.75)

axs[2].legend(loc='lower right')
axs[2].grid(True,linestyle='--')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Accuracy(global model)')

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
axs[3].plot(x4_1, y4_1, marker='s', label='x&x(Fedavg)',color='#B22222', markersize=3,linestyle='--',linewidth=0.75)
axs[3].plot(x4_2, y4_2, marker='*', label='Liner&x(Privatefl)',color='c', markersize=3,linestyle='-.',linewidth=0.75)
axs[3].plot(x4_3, y4_3, marker='+', label='x&Liner',color='black', markersize=3,linestyle=':',linewidth=0.75)
axs[3].plot(x4_4, y4_4, marker='>', label='Liner&Liner(PDPFL)',color='cornflowerblue', markersize=3,linestyle='--',linewidth=0.75)

axs[3].legend(loc='lower right')
axs[3].grid(True,linestyle='--')
axs[3].set_xlabel('Epoch')
axs[3].set_ylabel(' Max accuracy(local model)')

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

for ax in axs:
    ax.set_yticks(range(10, 101, 20))  # 设置 y 轴刻度为 10 的倍数，范围为 10 到 100
    ax.set_ylim(20, 100)  # 设置 y 轴范围从 10 到 100
# 调整布局
plt.tight_layout()


# 显示图形
fig.text(0.142, 0.009, '(a) MNIST dataset', ha='center')
fig.text(0.390, 0.009, '(b) MNIST dataset', ha='center')
fig.text(0.638, 0.009, '(c) FMNIST dataset', ha='center')
fig.text(0.888, 0.009, '(d) FMNIST dataset', ha='center')
plt.subplots_adjust(bottom=0.128)
plt.show()
fig.savefig('my_plot.png', dpi=1200)






#论文图5
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 创建随机数据（这里用随机数填充一个5x5的矩阵）
data = [
    [14.2, 16.18, 10.8, 12.6, 10.5, 14.6, 12.5, 11.7, 9.6],
    [69.7, 80.0, 80.07, 82.21, 91.9, 81.9, 79.62, 78.5, 65.2],
    [72.1, 78.6, 81.53, 81.89, 91.84, 82.4, 80.94, 79.6, 74.1],
    [74.9, 80.2, 84.0, 86.64, 91.7, 84.59, 83.2, 81.2, 75.8],
    [76.03, 82.5, 85.62, 90.03, 93.7, 89.17, 84.9, 82.1, 76.9],
    [82.66, 81.73, 82.76, 89.07, 92.11, 88.75, 83.2, 80.5, 81.66],
    [84.2, 80.9, 83.45, 88.91, 92.42, 87.59, 84.45, 80.1, 83.5],
    [82.41, 80.32, 85.59, 88.23, 92.78, 88.68, 85.59, 80.28, 82.7],
    [71.66, 72.27, 85.39, 88.7, 91.81, 89.14, 85.39, 75.36, 68.59]
]

# 绘制热图
data_array = np.array(data)

# 创建热图
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(data_array, annot=True, cmap='Blues', fmt='.2f')


# 设置刻度位置为每两个数据点中间
x_ticks_positions = np.arange(0, len(data_array[0]), 1) +0.5
x_ticks_labels = np.linspace(-1, 1, len(data_array[0]) )
print(x_ticks_labels)
y_ticks_positions = np.arange(0, len(data_array), 1)+0.5
y_ticks_labels = np.linspace(0, 2, len(data_array))

plt.xticks(x_ticks_positions, x_ticks_labels)
plt.yticks(y_ticks_positions, y_ticks_labels)

for tick in plt.gca().get_xticklabels():
    tick.set_horizontalalignment('center')
font = FontProperties(family='Times New Roman')
plt.xlabel(r'Initial value of b', fontproperties=font)
plt.ylabel(r'Initial value of $\alpha$', fontproperties=font)
plt.tick_params(axis='both', which='both', length=0)
plt.tight_layout()
plt.show()
'''



#论文图4
'''
import matplotlib.pyplot as plt

# 第一部分数据
x1 = [i for i in range(1,51)]

y1 = [36.82, 51.69, 57.39, 60.11, 62.45, 67.26, 69.84, 72.08, 75.0, 76.42, 78.63, 78.96, 79.11, 81.29, 81.39, 81.23, 83.15, 82.72, 83.89, 84.5, 85.34, 85.19, 85.11, 85.79, 85.92, 85.88, 85.74, 86.53, 86.53, 87.28, 86.59, 87.81, 87.97, 88.81, 88.17, 88.24, 88.48, 88.65, 88.55, 88.61, 88.83, 88.96, 89.03, 89.23, 89.45, 89.15, 89.56, 89.51, 89.69, 89.72]
# 第二部分数据
x2 = [i for i in range(1,51)]

y2 = [47.36, 59.23, 61.87, 66.3, 74, 77.28, 79.21, 80.87, 82.24, 84, 84.91, 85.53, 85.85, 87.34, 87.38, 88.01, 89.26,
        89.68, 89.99, 88.91, 88.47, 88.54, 88.85, 89.52, 90.15, 89.97, 89.72, 90.25, 90.58, 90.04, 91.08, 90.51, 90.19,
        91.39, 91.32, 91.29, 92.11, 91.34, 92, 92.06, 92.35, 92.48, 92.78, 92.9, 93.14, 93.42, 93.52, 93.79, 93.88,
        93.7]

x3 =[i for i in range(1,51)]
y3 = [42.06, 56.85, 62.73, 67.59, 72.89, 76.74, 78.37, 80.77, 82.74, 82.47, 84.18, 84.47, 84.66, 85.46, 86.36, 87.46, 88.32, 88.26, 89.02, 89.67, 89.37, 89.68, 90.11, 90.31, 89.72, 90.64, 90.43, 90.83, 91.13, 91.13, 91.42, 91.26, 90.47, 90.85, 91.69, 91.97, 91.86, 91.75, 91.98, 92.35, 92.5, 92.12, 91.53, 92.8, 92.15, 92.76, 92.23, 92.34, 92.5, 92.62]

x4 =[i for i in range(1,51)]
y4 = [18.73, 22.91, 24.18, 26.64, 43.18, 54.25, 61.25, 61.07, 62.24, 63, 66.41, 69.94, 70.78, 68.16, 74.13, 68.19, 76.67, 71.36, 80.59, 81.61, 83.71, 84.57, 84.55, 84.03, 85.24, 84.22, 84.36, 84.47, 83.59, 84.19, 84.34, 85.97, 85.84, 85.6, 85.73, 85.92, 86.1, 86.26, 86.28, 86.44, 86.68, 86.69, 86.27, 86.18, 86.24, 86.79, 86.66, 86.54, 86.87, 86.75]
# 绘制曲线图

x5 = [i for i in range(1,51)]
y5 = [25.55, 54.05, 60.23, 64.54, 70.05, 74.34, 75.34, 77.91, 79.56, 80.91, 82.67, 83.15, 83.64, 84.92, 85.67, 86.04, 86.69, 87.05, 87.63, 87.42, 87.57, 88.72, 88.1, 88.56, 88.7, 88.13, 87.43, 87.57, 87.31, 88.66, 88.25, 88.89, 90.29, 76.76, 80.94, 85.25, 88.86, 90.21, 90.74, 90.67, 91.05, 91.94, 90.37, 91.65, 91.45, 91.81, 92.1, 91.11, 91.25, 91.56]

x6 = [i for i in range(1,51)]
y6 = [25.23, 17.67, 18.69, 53.18, 40.47, 61.7, 70.99, 45.61, 73.07, 64.51, 76.63, 80.6, 80.12, 80.85, 81.56, 81.39, 80.96, 81.26, 83.09, 81.71, 83.36, 83.28, 83.75, 85.45, 85.39, 85.44, 85.61, 86.5, 86.19, 86.58, 86.91, 87.17, 87.71, 87.86, 87.61, 87.97, 88.06, 87.69, 88.77, 88.6, 88.61, 88.68, 89.51, 89.56, 90.27, 89.66, 89.96, 90.45, 89.85, 89.25]
plt.figure(figsize=(10, 6),dpi=300)  # 可选：设置图形大小
plt.plot(x1, y1, marker='s', label='x&x',color='#B22222', markersize=4,linestyle='--',linewidth=2)
plt.plot(x2, y2, marker='*', label='liner&liner',color='cornflowerblue', markersize=4,linewidth=2,linestyle='--')
plt.plot(x3, y3, marker='+', label='liner&tanh',color='black', markersize=4,linestyle='--',linewidth=2)
plt.plot(x4, y4, marker='o', label='liner&sigmoid',color='silver', markersize=4,linestyle='--',linewidth=2)
plt.plot(x5, y5, marker='X', label='log&liner',color='bisque', markersize=4,linestyle='--',linewidth=2)
plt.plot(x6, y6, marker='v', label='log&tanh',color='limegreen', markersize=4,linestyle='--',linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.suptitle('(a)IID', x=0.5, y=0.08, ha='center', fontsize=14)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True,linestyle='--')  # 添加网格线
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show()
'''




#论文图3
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

























#论文图2
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


