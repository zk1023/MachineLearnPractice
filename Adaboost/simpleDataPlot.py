# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: simpleDataPlot.py
@Date: 2018/1/16 16:13
"""
import numpy as np
import matplotlib.pyplot as plt

"""
创建单层决策树的数据集
Parameters:
    无
Returns:
    dataMat - 数据集
    classLabels - 数据标签
Modify:
    2018-1-16
"""
def loadSimpleData():
    dataMat = np.matrix([[1. , 2.1],
                         [1.5, 1.6],
                         [1.3, 1. ],
                         [1. , 1. ],
                         [2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

"""
数据可视化
Parameters:
    dataMat - 数据集
    labelMat - 数据标签
Returns:
    无
Modify:
    2018-1-16
"""
def showDataSet(dataMat, labelMat):
    data_plus = []    #存放正样本
    data_minus = []    #存放负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0 :
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)    #转化为numpy矩阵
    data_minus_np = np.array(data_minus)    #转化为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])    #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])    #正样本散点图
    plt.show()

if __name__ == '__main__':
    dataMat, classLabels = loadSimpleData()
    showDataSet(dataMat, classLabels)
