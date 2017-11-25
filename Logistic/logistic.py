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
    label = []
    for line in fr.readlines():
        #获取一行数据，按照空格分开
        lineArr = line.strip().split()
        #添加数据
        dataSet.append([1.0, float(lineArr[0]), float(lineArr[1])])
        #添加标签
        label.append(int(lineArr[2]))
    fr.close()
    #返回数据矩阵和标签数组
    return dataSet, label

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

plotDataSet()

