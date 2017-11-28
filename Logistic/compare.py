# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: compare.py
@Date: 2017/11/26 17:01
"""
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
"""
函数名：loadDataSet()
函数说明：加载数据

Parameters:
    无
Return:
    dataSet - 数据列表
    label - 标签列表
Modify:
    2017-11-25

"""
def loadDataSet():
    #打开文件
    fr = open("testSet.txt")
    #数据矩阵
    dataSet = []
    #标签数组
    labels = []
    for line in fr.readlines():
        #获取一行数据，按照空格分开
        lineArr = line.strip().split()
        #添加数据
        dataSet.append([1.0, float(lineArr[0]), float(lineArr[1])])
        #添加标签
        labels.append(int(lineArr[2]))
    #关闭文件
    fr.close()
    #返回数据矩阵和标签数组
    return dataSet, labels

"""
函数名：sigmoid(inX)
函数说明：sigmoid函数

Parameters:
    inX - 数据
Return:
    sigmoid函数
Modify:
    2017-11-26
"""
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


"""
函数名：gradientAscent(dataSet, labels)
函数说明：梯度上升算法

Parameters:
    dataSet - 数据集
    labels - 数据标签
Return:
    weights.getA() - 权重数组
    weights_array - 每次更新的回归系数
Modify:
    2017-11-26
"""
def gradientAscent(dataSet, labels):
    #把数据集转化为numpy类型的矩阵
    dataSetMat = mat(dataSet)
    #把标签转化为numpy类型的矩阵，并进行转置
    labelsMat = mat(labels).transpose()
    #获取矩阵参数（行、列）
    m, n = shape(dataSet)
    #移动步长，控制更新幅度
    alpha = 0.01
    #最大迭代次数
    maxCycles = 500
    #初始化权重向量
    weights = ones((n, 1))
    #初始化数组，存储每次更新的回归参数
    weights_array = array([])
    for k in range(maxCycles):
        #梯度上升矢量化公式
        weights = weights + alpha * dataSetMat.transpose() * (labelsMat - sigmoid(dataSetMat * weights))
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    return weights.getA(), weights_array

"""
函数名：stocGradientAscent1(dataSet, labels, numIter = 150)
函数说明:改进的随机机梯度上升算法

Parameters:
    dataSet - 数据集
    labels - 数据标签
    numIter - 迭代次数
Return:
    weights - 权重数组
    weights_array - 每次迭代更新的回归系数
Modify:
    2017-11-26
"""
def gradientAscent2(dataSet, labels, numIter = 150):
    #把数据集转化为numpy型数组
    dataMatrix = array(dataSet)
    #获取数据集大小
    m, n = shape(dataMatrix)
    #初始化权重数组
    weights = ones(n)
    #存储每次迭代更新的回归系数
    weights_array = array([])
    #改进的随机梯度上升算法
    for j in range(numIter):
        #获取数据集的行（0 - m）
        dataIndex = list(range(m))
        for i in range(m):
            #改变更新幅度
            alpha = 4 / (1.0 + i + j) + 0.01
            #选取随机数
            randIndex = int(random.uniform(0, len(dataIndex)))
            #更新回归系数
            # h = sigmoid(sum(dataMatrix[randIndex] * weights))
            # error = labels[randIndex] - h
            # weights = weights + alpha * error *dataMatrix[randIndex]
            weights = weights + alpha * dataMatrix[randIndex] * (labels[randIndex] - sigmoid(sum(dataMatrix[randIndex] * weights)))
            weights_array = np.append(weights_array, weights)
            #删除已使用样本
            del(dataIndex[randIndex])
    weights_array = weights_array.reshape(numIter * m, n)
    #返回回归系数
    return weights, weights_array

# dataSet, labels = loadDataSet()
# weights = gradientAscent2(array(dataSet), labels, numIter = 150)
# plotBestFit(weights)
"""
函数名：plotWeights(weights_array1, weights_array2)
函数说明:绘制回归系数与迭代次数的关系

Parameters:
    weights_array1 - 回归系数组1（梯度上升算法）
    weights_array2 - 回归系数组2（改进的随机梯度上升算法）
Return:
    无
Modify:
    2017-11-26
"""
def plotWeights(weights_array1, weights_array2):
    #设置汉字格式
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size = 14)
    #创建画布
    fig = plt.figure(figsize=(20,10))
    #添加子图,绘制X0与迭代次数的关系
    ax = fig.add_subplot(321)
    x1 = arange(0, len(weights_array1), 1)
    ax.plot(x1, weights_array1[:,0])
    ax_title_text = ax.set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'W0',FontProperties=font)
    plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'black')
    plt.setp(ax_ylabel_text, size = 20, weight = 'bold', color = 'black')

    # 添加子图,绘制X0与迭代次数的关系
    x2 = arange(0, len(weights_array2), 1)
    ax = fig.add_subplot(322)
    ax.plot(x2, weights_array2[:,0])
    ax_title_text = ax.set_title(u'梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'W0', FontProperties=font)
    plt.setp(ax_title_text, size=20, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=20, weight='bold', color='black')

    #绘制W1与迭代次数的关系
    ax = fig.add_subplot(323)
    ax.plot(x1, weights_array1[:,1])
    ax_ylabel_text = ax.set_ylabel(u'W1',FontProperties=font)
    plt.setp(ax_ylabel_text, size = 20, weight = 'bold', color = 'black')

    # 绘制W1与迭代次数的关系
    ax = fig.add_subplot(324)
    ax.plot(x2, weights_array2[:,1])
    ax_ylabel_text = ax.set_ylabel(u'W1', FontProperties=font)
    plt.setp(ax_ylabel_text, size=20, weight='bold', color='black')

    # 绘制W2与迭代次数的关系
    ax = fig.add_subplot(325)
    ax.plot(x1, weights_array1[:,2])
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'W2', FontProperties=font)
    plt.setp(ax_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=20, weight='bold', color='black')

    # 绘制W2与迭代次数的关系
    ax = fig.add_subplot(326)
    ax.plot(x2, weights_array2[:,2])
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'W2', FontProperties=font)
    plt.setp(ax_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=20, weight='bold', color='black')
    plt.show()

# dataSet, labels = loadDataSet()
# weights1,weights_array1 = gradientAscent2(dataSet, labels)
# weights2,weights_array2 = gradientAscent(dataSet, labels)
# # print len(weights_array2[:,1])
# plotWeights(weights_array1, weights_array2)
