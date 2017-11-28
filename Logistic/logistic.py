# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: logistic.py
@Date: 2017/11/25 20:21
"""
from numpy import *
import matplotlib.pyplot as plt
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
    fr.close()
    #返回数据矩阵和标签数组
    return dataSet, labels

# dataSet, label = loadDataSet()

"""
函数名：
函数说明：绘制数据集

Parameters:
    无
Return:
    无
Modify:
    2017-11-25
"""
def plotDataSet():
    #获取样本数据
    dataSet, label = loadDataSet()
    #把矩阵转化为数组
    dataArr = array(dataSet)
    #存储正样本坐标
    x1 = []
    y1 = []
    #存储负样本坐标
    x2 = []
    y2 = []
    for i in range(len(label)):
        #1为正样本
        if label[i] == 1:
            x1.append(dataArr[i,1])
            y1.append(dataArr[i,2])
        #0为负样本
        else:
            x2.append(dataArr[i,1])
            y2.append(dataArr[i,2])
    #建立空白图
    fig = plt.figure()
    #也可以指定绘图的大小
    # fig = plt.figure(figsize=(4,2))
    #添加子图,111分别代表在y轴方向、x轴的个数及当前所画图的焦点
    ax = fig.add_subplot(111)
    #绘制正样本
    ax.scatter(x1, y1, s = 20, c = 'red', marker = 's', alpha =.5)
    #绘制负样本
    ax.scatter(x2, y2, s = 20, c = 'green', alpha =.5)
    #标题
    plt.title("DataSet")
    #绘制label
    plt.xlabel('x')
    plt.ylabel('y')
    #显示
    plt.show()

# plotDataSet()
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
    alpha = 0.001
    #最大迭代次数
    maxCycles = 500
    #初始化权重向量
    weights = ones((n, 1))
    for k in range(maxCycles):
        #梯度上升矢量化公式
        weights = weights + alpha * dataSetMat.transpose() * (labelsMat - sigmoid(dataSetMat * weights))
    return weights.getA(), weights

"""
函数名：plotBestFit(weights)
函数说明：绘制决策边界

Parameters:
    weights - 权重数组
Return:
    无
Modify:
    2017-11-26
"""
def plotBestFit(weights):
    #获取样本数据
    dataSet, labels = loadDataSet()
    #把矩阵转化为数组
    dataArr = array(dataSet)
    #分别存储正样本、负样本
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in range(len(labels)):
        if labels[i] == 1:
            x1.append(dataArr[i, 1])
            y1.append(dataArr[i, 2])
        else:
            x2.append(dataArr[i, 1])
            y2.append(dataArr[i, 2])
    #绘制空白图
    fig = plt.figure()
    #添加子图
    ax = fig.add_subplot(111)
    #画点
    ax.scatter(x1, y1, s = 20, marker = 's', c = 'red', alpha=.5)
    ax.scatter(x2, y2, s = 20, c = 'green', alpha=.5)
    # 画出直线，weights[0]*1.0+weights[1]*x+weights[2]*y=0
    #arange()类似于内置函数range()，通过指定开始值、终值和步长创建表示等差数列的一维数组
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] * 1.0 - weights[1] * x) / weights[2]
    ax.plot(x, y)
    #标题
    plt.title('BestFit')
    #定义label
    plt.xlabel("X1")
    plt.ylabel('X2')
    #显示
    plt.show()

"""
函数名：gradientAscent2(dataSet, labels, numIter = 150)
函数说明:改进的随机机梯度上升算法

Parameters:
    dataSet - 数据集
    labels - 数据标签
    numIter = 150 - 迭代次数
Return:
    weights - 权重数组
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
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = labels[randIndex] - h
            weights = weights + alpha * error *dataMatrix[randIndex]
            # weights = weights + alpha * dataMatrix[randIndex] * (labels[randIndex] - sigmoid(sum(dataMatrix[randIndex] * weights)))
            #删除已使用样本
            del(dataIndex[randIndex])
    #返回回归系数
    return weights

# dataSet, labels = loadDataSet()
# weights = gradientAscent2(array(dataSet), labels, numIter = 150)
# plotBestFit(weights)


