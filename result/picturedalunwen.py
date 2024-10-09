import matplotlib.pyplot as plt
import numpy as np

# 防止画图中的乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置 柱状图 有几组
size = 5

x = np.arange(size)

# 有 a/b 两种类型的数据，n 设置为 2 0.6 不需要改，n 设置为要对比机组
total_width, n = 0.6, 3
# 每种类型的柱状图宽度，不需要改
width = total_width / n

# list1 代表样本 a，list2 代表样本 b
list1 = [81.8, 82.2, 85.2, 83.4, 82.7]
list2 = [89.3, 89.4, 88.9, 88.7, 89.5]


x = x - (total_width - width) / 2
plt.figure(figsize=(20, 8),dpi=300)
# 创建左侧子图
plt.subplot(1, 2, 1)
# 画柱状图，并设置阴影效果
plt.bar(x, list1, width=width, label=r'$\alpha$ = 1', color='orange', edgecolor='grey', linewidth=0.5,alpha=0.7)
plt.bar(x + width, list2, width=width, label=r'$\alpha$ = 100', color='#9ACD32', edgecolor='grey', linewidth=0.5,alpha=0.7)

plt.xticks(np.arange(5), ('0.3', '0.4', '0.5', '0.6', '0.7'), fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right', prop={'family': 'SimHei', 'weight': 'normal', 'size': 16})
plt.xlabel(r'阈值$\lambda$', fontsize=16)
plt.ylabel("全局模型的准确率", fontsize=16)



list4 = [85.3, 84.7, 84.2, 83.3, 81.4]
list5 = [89.4, 88.8, 89.2, 89.3, 89.1]

# 创建右侧子图
plt.subplot(1, 2, 2)
# 在右侧子图中再次绘制柱状图
plt.bar(x, list4, width=width, label=r'$\alpha$ = 1', color='orange', edgecolor='grey', linewidth=0.5,alpha=0.7)
plt.bar(x + width, list5, width=width, label=r'$\alpha$ = 100', color='#9ACD32', edgecolor='grey', linewidth=0.5,alpha=0.7)

plt.xticks(np.arange(5), ('2', '4', '7', '10', '20'), fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right', prop={'family': 'SimHei', 'weight': 'normal', 'size': 16})
plt.xlabel("尺度m", fontsize=16)
plt.ylabel("全局模型的准确率", fontsize=16)

# 调整子图间的距离
plt.tight_layout()

# 显示图形
plt.show()


'''
import matplotlib.pyplot as plt
import numpy as np

# 防止画图中的乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置 柱状图 有几组
size = 5

x = np.arange(size)

# 有 a/b 两种类型的数据，n 设置为 2 0.6 不需要改，n 设置为要对比机组
total_width, n = 0.6, 3
# 每种类型的柱状图宽度，不需要改
width = total_width / n

# list1 代表样本 a，list2 代表样本 b
list1 = [72.8, 74.1, 77.5, 80.4, 82.2]
list2 = [75.3, 78.0, 81.4, 83.9, 86.5]
list3 = [79.9, 82.7, 85.6, 88.1, 90.3]

x = x - (total_width - width) / 2
plt.figure(figsize=(20, 8),dpi=300)
# 创建左侧子图
plt.subplot(1, 2, 1)
# 画柱状图，并设置阴影效果
plt.bar(x, list1, width=width, label="Fedshuffle", color='#0066cc', edgecolor='grey', linewidth=0.5,alpha=0.7)
plt.bar(x + width, list2, width=width, label="shuffle-Privatefl", color='#9ACD32', edgecolor='grey', linewidth=0.5,alpha=0.7)
plt.bar(x + 2*width, list3, width=width, label="shuffle-Fedpd", color='yellow', edgecolor='grey', linewidth=0.5,alpha=0.7)
plt.xticks(np.arange(5), ('10', '20', '30', '40', '50'), fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right', prop={'family': 'SimHei', 'weight': 'normal', 'size': 16})
plt.xlabel("MNIST", fontsize=16)
plt.ylabel("全局模型的准确率", fontsize=16)
list4 = [51.6, 54.3, 57.2, 60.5, 63.7]
list5 = [57.4, 60.8, 63.1, 66.2, 69.5]
list6 = [64.9, 68.7, 71.6, 73.1, 76.3]
# 创建右侧子图
plt.subplot(1, 2, 2)
# 在右侧子图中再次绘制柱状图
plt.bar(x, list4, width=width, label="Fedshuffle", color='#0066cc', edgecolor='grey', linewidth=0.5,alpha=0.7)
plt.bar(x + width, list5, width=width, label="shuffle-Privatefl", color='#9ACD32', edgecolor='grey', linewidth=0.5,alpha=0.7)
plt.bar(x + 2*width, list6, width=width, label="shuffle-Fedpd", color='yellow', edgecolor='grey', linewidth=0.5,alpha=0.7)
plt.xticks(np.arange(5), ('10', '20', '30', '40', '50'), fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right', prop={'family': 'SimHei', 'weight': 'normal', 'size': 16})
plt.xlabel("Fashion-MNIST", fontsize=16)
plt.ylabel("全局模型的准确率", fontsize=16)

# 调整子图间的距离
plt.tight_layout()

# 显示图形
plt.show()

'''


'''

#大论文第三章实验图

import matplotlib.pyplot as plt

# 第一批数据
x1_1 = [i for i in range(1,51)]
y1_1 = [19.94, 23.13, 28.94, 33.97, 37.13, 41.72, 47.21, 50.72, 55.75, 63.79,
    65.3, 67.82, 68.81, 71.38, 70.48, 73.54, 72.83, 75.47, 74.14, 75.79,
    76.34, 78.8, 77.3, 78.45, 79.24, 80.28, 79.48, 80.25, 81.58, 81,
    81.77, 81.66, 82.54, 81.38, 83.14, 82.54, 83.02, 82.71, 82.94, 83.76,
    84.72, 84.34, 85.26, 83.8, 85.04, 84.25, 85.82, 84.35, 85.62, 85.96]

x1_2 = [i for i in range(1,51)]
y1_2 = [ 11.86, 29.18, 38.88, 40.54, 41.33, 40.75, 42.06, 43.92, 44.9, 47.7,
    49.08, 50.11, 52.14, 51.3, 52.69, 53.45, 54.73, 54.1, 55.94, 55.36,
    56.4, 57.72, 56.88, 58.29, 57.32, 59.33, 60.5, 61.77, 60.84, 61.91,
    62.74, 62.93, 63.63, 65.57, 64.77, 65.81, 65.25, 66.05, 65.14, 66.8,
    67.4, 67.54, 68.58, 67.74, 68.33, 69.26, 70.17, 69.87, 69.54, 70.14]


# 第二批数据
x1_3 = [i for i in range(1,51)]
y1_3 = [24.19, 24.15, 29.86, 39.93, 41.66, 48.58, 52.29, 55.96, 63.1, 65.63,
    70.82, 72.52, 73.44, 75.32, 74.55, 76.46, 77.21, 78.56, 79.16, 78.21,
    79.73, 80.29, 81.48, 80.35, 81.83, 82.28, 81.45, 83.15, 84.19, 83.49,
    84.27, 85.15, 84.29, 85.19, 85.53, 86.33, 85.77, 86.2, 87.42, 86.68,
    87.31, 87.15, 86.38, 87.24, 88.41, 88.33, 88.15, 88.62, 88.47, 88.61
]

x1_4 = [i for i in range(1,51)]
y1_4 = [24.75, 24.89, 29.07, 35.9, 41.22, 52.85, 56.02, 59.91, 66.96, 68.71,
    71.65, 73.31, 74.93, 76.25, 75.34, 78.37, 79.06, 78.59, 79.29, 80,
    80.53, 81.7, 81.37, 82.88, 82.37, 81.74, 83.25, 82.26, 84.12, 84.43,
    84.76, 84.58, 85.89, 85.44, 86.65, 85.34, 86.19, 85.26, 86.41, 86.44,
    87.16, 87.44, 88.9, 87.81, 88.63, 88.29, 88.74, 88.29, 88.82, 88.5]

x1_5 = [i for i in range(1,51)]
y1_5 = [27.6, 23.35, 39.17, 42.87, 47.42, 58.49, 65.71, 70.26, 72.62, 74.66,
    75.84, 77.11, 78, 77.39, 79.23, 81.32, 80.5, 81.62, 82.25, 81.4,
    83.14, 84.74, 84.22, 85.76, 85.12, 85.44, 86.22, 85.33, 86.79, 87.33,
    86.89, 87.41, 88.24, 88.67, 89.55, 88.31, 89.21, 89.69, 88.66, 89.87,
    90.71, 90.41, 90.59, 90.92, 90.86, 91.04, 91.65, 90.4, 91.67, 91.46]

x1_6 = [i for i in range(1,51)]
y1_6 = [14.04, 33.48, 49, 61.79, 64.5, 71.4, 74.89, 76.68, 75.82, 78.31,
    80.49, 81.33, 83.36, 82.6, 84, 83.76, 85.41, 86.3, 85.57, 87.21,
    87.4, 88.19, 89.29, 88.77, 89.48, 90.89, 89.78, 90.36, 90.96, 91.23,
    90.41, 91.21, 92.34, 91.74, 92.58, 92.62, 93.89, 93.03, 93.48, 93.18,
    93.54, 93.67, 93.42, 94.04, 93.41, 94.21, 94.57, 94.48, 94.73, 94.34]


x2_1 = [i for i in range(1,51)]
y2_1 = [16.21, 19.34, 28.37, 34.56, 37.2, 41.11, 42.38, 43.33, 44.11, 46.5,
    48.23, 49, 49.3, 51.83, 52.26, 54.19, 54.9, 53.36, 56.96, 55.92, 57.47,
    58.65, 59.79, 61.25, 60.03, 60.83, 61.55, 60.95, 62.18, 62.85, 61.51,
    63.25, 63.73, 64.12, 64.37, 63.54, 64.88, 63.67, 63.52, 64.64, 63.98,
    64.83, 64.15, 65.27, 65.5, 64.47, 66.51, 65.75, 66.37, 66.03]

x2_2 = [i for i in range(1,51)]
y2_2 = [8.73, 26.78, 27.65, 27.92, 29.12, 31.92, 36.48, 37.99, 38.89, 39.58,
    40.29, 41.34, 41.52, 43.13, 44.35, 46.74, 47.62, 46.26, 47.62, 48.15,
    49.06, 48.32, 49.41, 50.3, 51.17, 51.06, 50.55, 52.26, 51.9, 51.46,
    52.15, 52.72, 52.14, 53.14, 52.21, 53.43, 51.2, 53.48, 54.17, 53.32,
    55.22, 54.42, 55.28, 54.92, 55.21, 56.21, 55.88, 55.69, 56.56, 56.17]


# 第二批数据
x2_3 = [i for i in range(1,51)]
y2_3 = [10, 18, 27.38, 35.87, 38.58, 41.15, 45.43, 48.31, 50.26, 55.63,
    53.22, 54.7, 56.29, 57.7, 58.9, 58.49, 60.29, 60.41, 61.56, 61.69,
    62.12, 62.48, 62.43, 62.35, 63.13, 62.36, 63.81, 63.51, 64.1, 64.55,
    64.84, 65.41, 65.12, 65.38, 66.52, 65.65, 66.24, 65.73, 66.35, 66.59,
    66.8, 66.4, 67.21, 67.5, 66.36, 67.58, 67.94, 67.57, 68.42, 68.14]

x2_4 = [i for i in range(1,51)]
y2_4 = [12.58, 22.02, 32.52, 36.21, 37.41, 41.97, 42.52, 45.4, 49.53, 51.25,
    53.04, 54.62, 56.1, 57.25, 58.27, 59.28, 59.54, 60.23, 61.22, 61.47,
    62.94, 62.62, 62.38, 63.97, 63.48, 63.79, 65.13, 64.72, 64.84, 65.08,
    65.29, 65.76, 65.23, 65.94, 66.5, 66.33, 66.94, 66.77, 66.8, 67.23, 67.45,
    67.49, 67.74, 67.81, 67.49, 68.2, 68.3, 68.79, 68.49, 68.67]

x2_5 = [i for i in range(1,51)]
y2_5 = [10, 30.16, 35.77, 37.68, 41.2, 45.96, 49.69, 52.42, 56.15, 57.15,
    58.61, 59.63, 60.37, 61.07, 61.56, 61.87, 60.21, 62.98, 62.51, 63.14,
    62.59, 63.8, 65, 64.37, 64.65, 63.89, 65.09, 64.48, 65.23, 65.86,
    65.31, 66.62, 67.82, 66.63, 66.48, 67.95, 68.86, 67.55, 68.33, 68.48,
    68.94, 68.21, 68.91, 69.27, 69.76, 69.26, 70.25, 69.93, 70.61, 70.25]

x2_6 = [i for i in range(1,51)]
y2_6 = [21.28, 35.44, 37.51, 47.05, 51.82, 53.54, 57.29, 60.09, 59.59, 60.86,
    61.36, 62.65, 61.7, 63.98, 62.84, 65.28, 64.59, 65.59, 64.58, 66.27,
    66.63, 67.64, 67.09, 68.02, 67.66, 68.92, 67.78, 69.29, 68.93, 70.14,
    70.23, 70.44, 70.26, 71.9, 71.03, 72.24, 70.86, 71.76, 72.35, 73.76,
    72.63, 73.83, 73.65, 74.7, 74.24, 74.84, 74.84, 75.16, 76.06, 74.87]


x3_1 = [i for i in range(1,51)]   #啥都没有
y3_1 = [
    3.908653846, 6.322115385, 8.25, 11.3125, 14.14423077, 16.72596154, 18.49038462,
    21.89423077, 23.83653846, 24.45192308, 25.36153846, 28.53365385, 29.85576923,
    31.21153846, 34.37019231, 32.45192308, 33.31057692, 35.83653846, 34.74134615,
    37.87019231, 38.25865385, 38.48076923, 40.23557692, 42.89903846, 41.54038462,
    42.26730769, 42.57403846, 43.19038462, 44.58846154, 43.48557692, 45.74038462,
    46.37596154, 48.225, 47.55480769, 48.52788462, 49.51442308, 50.22115385,
    48.89903846, 49.79038462, 52.09134615, 51.68269231, 52.99807692, 53.32307692,
    54.22403846, 55.74326923, 56.20096154, 55.49615385, 57.15384615, 57.69711538,
    57.46346154]

x3_2 = [i for i in range(1,51)]   #per
y3_2 = [3.235576923, 6.903846154, 5.341346154, 5.504807692, 8.706730769, 11.64903846,
    14.35576923, 16.63461538, 19.07692308, 21.31730769, 24.21634615, 25.02403846,
    26.63461538, 28.58365385, 29.69711538, 30.93269231, 31.14423077, 33.19711538,
    34.4625, 34.90384615, 35.69230769, 36.32211538, 37.73461538, 37.49807692,
    38.46153846, 39.41442308, 39.51442308, 39.63269231, 40.5, 40.92307692,
    41.41346154, 42.3125, 41.77788462, 42.825, 42.589423077, 43.36538462,
    43.73557692, 44.04326923, 43.57019231, 44.38846154, 45.11057692, 44.74230769,
    45.70673077, 46.37211538, 46.14615385, 46.77692308, 46.20769231, 47.27692308,
    47.62692308, 47.19134615]


# 第二批数据  #privatefl
x3_3 = [i for i in range(1,51)]
y3_3 = [8.129807692, 6.730769231, 9.466346154, 16.92307692, 19.67307692, 21.97596154,
    25.96634615, 26.62019231, 31.09615385, 32.3125, 35.41826923, 36.20192308,
    39.02403846, 42.83173077, 41.60096154, 43.09134615, 45.63461538, 47.07692308,
    48.55769231, 49.95192308, 54.28365385, 52.52403846, 53.6875, 54.84134615,
    55.88942308, 57.07211538, 56.13461538, 57.9875, 60.11057692, 59.71442308,
    61.6875, 62.40865385, 63.79711538, 64.15288462, 63.78846154, 64.82307692,
    65.93365385, 67.225, 65.64038462, 67.66826923, 68.75480769, 67.71634615,
    69.16346154, 69.65288462, 70.4375, 71.26442308, 70.825, 71.45288462,
    71.7375, 71.83653846]

x3_4 = [i for i in range(1,51)]  #x&liner
y3_4 = [6.293269231, 4.432692308, 12.408653846, 14.47115385, 19.5625, 22.33653846,
    25.6875, 27.63461538, 31.16346154, 32.54807692, 37.48076923, 36.49519231,
    39.47788462, 40.36634615, 42.48076923, 44.10096154, 43.46634615, 46.85096154,
    48.15384615, 49.32788462, 50.71153846, 54.95673077, 52.20192308, 54.32211538,
    55.30288462, 57.38461538, 59.45192308, 58.32692308, 59.65865385, 60.08173077,
    61.96634615, 61.69711538, 61.99519231, 63.53557692, 62.89423077, 64.65384615,
    65.32692308, 65.93269231, 66.52403846, 65.09615385, 67.364423077, 68.17307692,
    68.57211538, 69.53269231, 69.12307692, 69.90865385, 70.74134615, 70.34519231,
    71.21153846, 71.63942308]

x3_5 = [i for i in range(1,51)]  #liner&liner
y3_5 = [4.567307692, 5.302884615, 17.74038462, 22.25480769, 26.10576923, 30.48557692,
    33.79326923, 35.87980769, 37.88942308, 43.65865385, 45.09134615, 47.18269231,
    49.01923077, 51.85576923, 52.70673077, 54.58846154, 55.12980769, 54.40673077,
    58.66346154, 59.91346154, 60.10096154, 62.11538462, 61.57307692, 63.04326923,
    64.94711538, 64.37115385, 65.78365385, 67.62019231, 67.98365385, 68.4375,
    69.42307692, 70.91442308, 70.19615385, 71.17307692, 70.39230769, 72.425,
    72.34807692, 71.45576923, 73.23557692, 73.52884615, 73.8125, 73.49134615,
    74.6, 73.85096154, 75.0625, 75.39423077, 74.85769231, 75.30769231, 75.98076923,
    76.21634615]

x3_6 = [i for i in range(1,51)] #liner&liner&W
y3_6 = [4.230769231, 15.44230769, 23.42788462, 26.52884615, 39.64903846, 49.46153846,
    51.625, 54.99519231, 55.95673077, 58.78365385, 62.09134615, 64.76442308,
    66.30769231, 67.89423077, 68.95192308, 68.44230769, 70.58173077, 70.63461538,
    72.19711538, 72.92788462, 73.85673077, 72.97115385, 74.70192308, 75.09615385,
    74.78076923, 76.12980769, 76.75, 75.87019231, 77.98173077, 77.32980769,
    77.72596154, 76.63942308, 78.08173077, 79.28365385, 78.49038462, 78.875,
    79.09615385, 79.40384615, 79.3125, 80.53365385, 79.71634615, 79.78846154,
    80.19230769, 80.54326923, 79.75480769, 80.95673077, 81.01923077, 81.08173077,
    80.48557692, 80.72115385]

x4_1 = [i for i in range(1,51)]
y4_1 = [4.567307692, 4.447115385, 7.384615385, 13.70192308, 21.19711538,
    23.66826923, 32.32211538, 35.375, 38.35576923, 37.90384615,
    39.91346154, 41.60096154, 40.83173077, 42.14903846, 43.41826923,
    45.26923077, 45.16346154, 45.875, 48.63461538, 47.10096154,
    50.55288462, 49.92307692, 49.85576923, 48.25, 50.30288462,
    49.48076923, 49.75, 50.10096154, 49.31730769, 50.44230769,
    51.0625, 52.125, 53.73557692, 52.125, 52.26923077, 52.04807692,
    53.15865385, 52.42307692, 53.07692308, 52.76923077, 53.26923077,
    53.53846154, 55.14903846, 54.9375, 53.72115385, 54.47115385,
    54.88461538, 56.1875, 55.27884615, 57.35576923]

x4_2 = [i for i in range(1,51)]
y4_2 = [3.884615385, 3.870192308, 4.317307692, 7.692307692, 12.84134615,
    15.76442308, 17.78365385, 19.89903846, 21.24519231, 22.22115385,
    23.87980769, 25.1875, 25.62019231, 26.12980769, 27.00961538,
    27.95192308, 29.36057692, 30.07211538, 30.42788462, 31, 32.88461538,
    34.4375, 34.625, 35.12019231, 35.43269231, 36.88461538, 37.22596154,
    37.6875, 38.27403846, 39.73076923, 39.81730769, 39.37980769,
    39.44230769, 39.79807692, 40.94230769, 41.27403846, 41.49519231,
    41.73557692, 41.80288462, 42.44230769, 42.00961538, 41.70192308,
    41.89423077, 43.41826923, 43.20673077, 42.87980769, 42.98557692,
    44.69230769, 45.01923077, 44.5625]


# 第二批数据
x4_3 = [i for i in range(1,51)]
y4_3 = [4.408653846, 8.365384615, 15.40384615, 27.25480769, 35.38942308,
    41.21153846, 45, 44.5, 47.27884615, 49.54807692, 51.53365385,
    52.39423077, 52.46634615, 57.24038462, 56.09615385, 56.89423077,
    58.84615385, 59.46634615, 58.87980769, 59.17788462, 58.375,
    61.91346154, 61.88461538, 62.80769231, 61.91346154, 65.08653846,
    63.90865385, 64.82692308, 64.79807692, 66.41826923, 66.52403846,
    67.00961538, 65.55769231, 66.14423077, 66.06730769, 65.63461538,
    68.02884615, 68.38942308, 68.53846154, 67.23557692, 65.52884615,
    68.80288462, 67.90865385, 65.53365385, 66.20192308, 65.72596154,
    66.86538462, 67.37980769, 66.02884615, 65.39903846]


x4_4 = [i for i in range(1,51)]
y4_4 = [7.240384615, 5.903846154, 13.34134615, 25.45192308, 35.6875,
    40.36538462, 41.875, 43.93269231, 45.27884615, 47.06730769,
    48.45673077, 51.53846154, 51.92788462, 52.27403846, 54.73076923,
    55.125, 54.02403846, 56.55288462, 57.04807692, 59.59615385,
    57.83653846, 60.51923077, 62.74519231, 61.26442308, 60.51442308,
    63.42788462, 63.52403846, 61.97596154, 63.38942308, 65.08173077,
    65.99519231, 67.27403846, 65.34615385, 66.84134615, 66.52884615,
    66.91826923, 66.43269231, 67.9375, 67.20192308, 68.30288462,
    66.07692308, 67.14903846, 66.44711538, 66.58653846, 65.31730769,
    66.9375, 67.80288462, 67.37019231, 66.51923077, 67.05288462]

x4_5 = [i for i in range(1,51)]
y4_5 = [7.081730769, 15.33653846, 26.68269231, 40.02884615, 43.56730769,
    49.23076923, 49.36057692, 53.70673077, 56.88942308, 57.59134615,
    60.13942308, 59.24519231, 62.04326923, 63.96153846, 63.95673077,
    63.00961538, 63.83173077, 66.00480769, 64.52403846, 65.75,
    67.65384615, 68.15384615, 70.83173077, 69.12019231, 70.12019231,
    71.5625, 70.27884615, 69.76442308, 70.29807692, 70.54326923,
    71.33173077, 71.25480769, 72.07211538, 69.36057692, 71.26442308,
    69.92307692, 70.34711538, 69.55288462, 70.06730769, 71.87019231,
    70.34615385, 71.70673077, 72.25961538, 70.84134615, 69.63461538,
    71.13942308, 71.34134615, 70.78365385, 70.81730769, 70.15384615]

x4_6 = [i for i in range(1,51)]
y4_6 = [8.913461538, 26.83653846, 43.86057692, 51.19711538, 54.72115385,
    59.5625, 62.76923077, 65.01442308, 67.59134615, 68.89423077,
    67.98076923, 70.21634615, 70.66346154, 72.03365385, 72.86057692,
    73.1875, 73, 73.48076923, 74.62019231, 73.00480769, 75.6875,
    76.11057692, 74.94711538, 74.98557692, 74.6875, 75.91346154,
    75.98076923, 74.40865385, 76.29326923, 75.84134615, 74.59134615,
    75.60096154, 75.75961538, 74.62019231, 74.47596154, 75.04326923,
    74.58653846, 74.17307692, 74.31730769, 75.72596154, 75.96634615,
    75.49038462, 75.85096154, 75.92788462, 76.22596154, 76.11057692,
    75.98557692, 75.52884615, 75.97596154, 76.14903846]
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 16),dpi=300)

# 第一个子图
axs[0][0].plot(x1_1, y1_1, marker='s', label='Fedshuffle',color='#B22222', markersize=7,linestyle='--',linewidth=2)
axs[0][0].plot(x1_2, y1_2, marker='*', label='shuffle-Fedper',color='c', markersize=7,linestyle='-.',linewidth=2)
axs[0][0].plot(x1_3, y1_3, marker='d', label='shuffle-Privatefl(liner&x)',color='orange', markersize=7,linestyle=':',linewidth=2)
axs[0][0].plot(x1_4, y1_4, marker='v', label='shuffle-Fedpd(x&liner)',color='cornflowerblue', markersize=7,linestyle='--',linewidth=2)
axs[0][0].plot(x1_5, y1_5, marker='X', label='shuffle-Fedpd(liner&liner)',color='limegreen', markersize=7,linestyle='--',linewidth=2)
axs[0][0].plot(x1_6, y1_6, marker='o', label='shuffle-Fedpd(liner&liner&W)',color='violet', markersize=7,linestyle='--',linewidth=2)

axs[0][0].legend(loc='lower right', fontsize=16)
axs[0][0].grid(True,linestyle='--')
axs[0][0].set_xlabel('Epoch', fontsize=16)
axs[0][0].set_ylabel('Accuracy(global model)', fontsize=16)
axs[0][0].text(0.5, -0.15, '(a)MNIST', ha='center', transform=axs[0][0].transAxes, fontsize=16)
# 注释（标注）



# 第二个子图
axs[0][1].plot(x2_1, y2_1, marker='s', label='Fedshuffle',color='#B22222', markersize=7,linestyle='--',linewidth=2)
axs[0][1].plot(x2_2, y2_2, marker='*', label='shuffle-Fedper',color='c', markersize=7,linestyle='-.',linewidth=2)
axs[0][1].plot(x2_3, y2_3, marker='d', label='shuffle-Privatefl(liner&x)',color='orange', markersize=7,linestyle=':',linewidth=2)
axs[0][1].plot(x2_4, y2_4, marker='v', label='shuffle-Fedpd(x&liner)',color='cornflowerblue', markersize=7,linestyle='--',linewidth=2)
axs[0][1].plot(x2_5, y2_5, marker='X', label='shuffle-Fedpd(liner&liner)',color='limegreen', markersize=7,linestyle='--',linewidth=2)
axs[0][1].plot(x2_6, y2_6, marker='o', label='shuffle-Fedpd(liner&liner&W)',color='violet', markersize=7,linestyle='--',linewidth=2)

axs[0][1].legend(loc='lower right', fontsize=16)
axs[0][1].grid(True,linestyle='--')
axs[0][1].set_xlabel('Epoch', fontsize=16)
axs[0][1].set_ylabel('Accuracy(global model)', fontsize=16)
axs[0][1].text(0.5, -0.15, '(b)Fashion-MNIST', ha='center', transform=axs[0][1].transAxes, fontsize=16)



# 第三个子图
axs[1][0].plot(x3_1, y3_1, marker='s', label='Fedshuffle',color='#B22222', markersize=7,linestyle='--',linewidth=2)
axs[1][0].plot(x3_2, y3_2, marker='*', label='shuffle-Fedper',color='c', markersize=7,linestyle='-.',linewidth=2)
axs[1][0].plot(x3_3, y3_3, marker='d', label='shuffle-Privatefl(liner&x)',color='orange', markersize=7,linestyle=':',linewidth=2)
axs[1][0].plot(x3_4, y3_4, marker='v', label='shuffle-Fedpd(x&liner)',color='cornflowerblue', markersize=7,linestyle='--',linewidth=2)
axs[1][0].plot(x3_5, y3_5, marker='X', label='shuffle-Fedpd(liner&liner)',color='limegreen', markersize=7,linestyle='--',linewidth=2)
axs[1][0].plot(x3_6, y3_6, marker='o', label='shuffle-Fedpd(liner&liner&W)',color='violet', markersize=7,linestyle='--',linewidth=2)
axs[1][0].legend(loc='lower right', fontsize=16)
axs[1][0].grid(True,linestyle='--')
axs[1][0].set_xlabel('Epoch', fontsize=16)
axs[1][0].set_ylabel('Accuracy(global model)', fontsize=16)
axs[1][0].text(0.5, -0.15, '(c)EMNIST', ha='center', transform=axs[1][0].transAxes, fontsize=16)


# 第四个子图
axs[1][1].plot(x4_1, y4_1, marker='s', label='Fedshuffle',color='#B22222', markersize=7,linestyle='--',linewidth=2)
axs[1][1].plot(x4_2, y4_2, marker='*', label='shuffle-Fedper',color='c', markersize=7,linestyle='-.',linewidth=2)
axs[1][1].plot(x4_3, y4_3, marker='d', label='shuffle-Privatefl(liner&x)',color='orange', markersize=7,linestyle=':',linewidth=2)
axs[1][1].plot(x4_4, y4_4, marker='v', label='shuffle-Fedpd(x&liner)',color='cornflowerblue', markersize=7,linestyle='--',linewidth=2)
axs[1][1].plot(x4_5, y4_5, marker='X', label='shuffle-Fedpd(liner&liner)',color='limegreen', markersize=7,linestyle='--',linewidth=2)
axs[1][1].plot(x4_6, y4_6, marker='o', label='shuffle-Fedpd(liner&liner&W)',color='violet', markersize=7,linestyle='--',linewidth=2)

axs[1][1].legend(loc='lower right', fontsize=16)
axs[1][1].grid(True,linestyle='--')
axs[1][1].set_xlabel('Epoch', fontsize=16)
axs[1][1].set_ylabel('Accuracy(global model)', fontsize=16)
axs[1][1].text(0.5, -0.15, r'(d)Cifar-10', ha='center', transform=axs[1][1].transAxes, fontsize=16)



for i in axs:
    for ax in i:
        ax.set_yticks(range(0, 101, 10))  # 设置 y 轴刻度为 10 的倍数，范围为 10 到 100
        ax.set_ylim(10, 100)  # 设置 y 轴范围从 10 到 100
        ax.tick_params(axis='both', which='major', labelsize=16)
# 调整布局



# 显示图形

plt.subplots_adjust(bottom=0.128)
plt.show()
fig.savefig('my_plot.png')

'''

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


'''大论文
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

