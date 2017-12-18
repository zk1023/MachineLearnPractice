# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: svmTest.py
@Date: 2017/12/14 11:17
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import types

"""
函数名：loadDataSet()
函数说明：获取数据

Parameters:
    无
Returns:
    dataSet - 数据集
    labels - 数据标签
Modify:
    2017-12-14
"""
def loadDataSet():
    fr = open("testSet.txt")
    dataSet = []
    labels = []
    readlines = fr.readlines()
    for line in readlines:
        arr = line.strip().split('\t')
        dataSet.append([float(arr[0]), float(arr[1])])
        labels.append(float(arr[2]))
    fr.close()
    return dataSet, labels


"""
函数名: showDataSet()
函数说明: 数据可视化 

Parameters:
    dataSet - 数据矩阵
    labels - 数据标签
Returns:
    无
Modify:
    2017-12-14
"""
def showDataSet(dataMat, labelMat):
    #正样本
    data_plus = []
    #负样本
    data_minus = []
    #遍历数据，获取正负样本集
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    #转化为numpy矩阵，，但是并不知道为啥要这样
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    #描点画图（散点图）
    #正样本
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    #负样本
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()

"""
函数名: selectJrand(i, m)
函数说明: 随机选取alpha 

Parameters:
    i - 第i个alpha
    m - alpha总数
Returns:
    j - 所选取的alpha序号
Modify:
    2017-12-14
"""
def selectJrand(i, m):
    j = i
    #选取一个与i不同的alpha
    while j == i :
        j = int(random.uniform(0, m))
    return j

"""
函数名: cliAlpha(aj, H, L)
函数说明: 修剪alpha 

Parameters:
    aj - alpha值
    H - alpha上限
    L - alpha下限
Returns:
    aj - alpha值
Modify:
    2017-12-14
"""
def cliAlpha(aj, H, L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj

"""
函数名: smoSimple(dataMatIn, classLabels, C, toler, maxIter)
函数说明: 简化版SMO算法

Parameters:
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛因子
    toler - 容错率
    maxIter - 最大迭代次数
Returns:
    b - 截距
    alphas - alphas值集合
Modify:
    2017-12-14
"""
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    #把数据集和标签转化为numpy的mat存储
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    #初始化b
    b = 0
    #统计dataMatrix维度
    m, n = np.shape(dataMatrix)
    #初始化alpha参数，设置为0
    alphas = np.mat(np.zeros((m, 1)))
    # print alphas
    #初始化迭代次数
    iter_num = 0
    while iter_num < maxIter:
        #记录alpha是否被优化
        alphaPairsChanged = 0
        for i in range(m):
            #步骤一 计算误差Ei
            fxi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b
            Ei = fxi - float(labelMat[i])
            #优化alpha 设定容错率
            if (labelMat[i] * Ei < -toler and alphas[i] < C) or (labelMat[i] * Ei > toler and alphas[i] > 0):
                #随机选择另一个alpha_j,成对优化alpha_i和alpha_j
                j = selectJrand(i, m)
                #步骤一 计算误差
                fxj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b
                Ej = fxj - float(labelMat[j])
                #保存更新前的alpha值，使用深拷贝
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #步骤二 计算上下界和H
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    print "L == H"
                    continue
                #步骤三 计算eta
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T \
                       - dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0:
                    print "eta >= 0"
                    continue
                #步骤四 更新alpha_j
                alphas[j] -= labelMat[j] * (Ei - Ej) /eta
                #步骤五 修剪alpha_j
                alphas[j] = cliAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001 :
                    print "alpha_j changed too small"
                    continue
                #步骤六 更新alpha_i
                alphas[i] += labelMat[j] * labelMat[i] *(alphaJold - alphas[j])
                #步骤七 更新b1 b2
                b1 = b - Ei -labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                #步骤八 更新b
                if (alphas[i] > 0) and (alphas[i] < C):
                    b = b1
                elif (alphas[j] > 0) and (alphas[i] < C):
                    b = b2
                else:
                    b = float(b1 + b2) / 2.0
                #统计优化次数
                alphaPairsChanged += 1
                #打印统计信息
                print "第%d次迭代 样本:%d,优化次数为%d" % (iter_num, i, alphaPairsChanged)
        #更新迭代次数
        if alphaPairsChanged == 0:
            iter_num += 1
        else:
            iter_num = 0
        print "迭代次数为%d" % iter_num
    return b, alphas

"""
函数名: get_w(dataMat, labelMat, alphas)
函数说明: 计算法向量

Parameters:
    dataMat - 数据矩阵
    labelMat - 标签矩阵
    alphas - alpha值
Returns:
    w - 直线法向量
Modify:
    2017-12-14
"""
def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()

"""
函数名: showClassifier(dataMat, w, b)
函数说明: 分类结果可视化

Parameters:
    dataMat - 数据矩阵
    w - 直线法向量
    b - 直线截距
Returns:
    无
Modify:
    2017-12-14
"""
def showClassifier(dataMat, w, b):
    # 正样本
    data_plus = []
    # 负样本
    data_minus = []
    # 遍历数据，获取正负样本集
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 转化为numpy矩阵，，但是并不知道为啥要这样
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    # 描点画图（散点图）
    # 正样本
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    # 负样本
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) /a2
    plt.plot([x1, x2], [y1, y2])
    print "w:",w
    print "b:",b
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], c='none', s=150, edgecolors='red', alpha=0.7, linewidths=1.5)
    plt.show()


if __name__ == '__main__':
    start = time.time()
    dataSet, labels = loadDataSet()
    b, alphas = smoSimple(dataSet, labels, 0.6, 0.001, 40)
    w = get_w(dataSet, labels, alphas)
    labelMat = np.mat(labels).transpose()
    showClassifier(dataSet, w, b)
    # showDataSet(dataSet, labels)
    end = time.time()
    elapsed = end - start
    print "Time taken: ", elapsed, "seconds."
