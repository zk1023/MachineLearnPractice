# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: predictDeathRate.py
@Date: 2017/11/27 21:19
"""
import logistic
import numpy
import compare

"""
函数名：colicTest()
函数说明：使用Logistic分类器进行预测

Parameters:
    无
Return:
    无
Modify:
    2017-11-27
"""
def colicTest():
    #获取训练集
    frTrain = open("horseColicTraining.txt")
    #获取测试集
    frTest = open("horseColicTest.txt")
    #训练集矩阵和标签数组
    trainMat = []; trainLabels = []
    #逐行遍历训练数据集
    for line in frTrain.readlines():
        #定义数组，存储每行元素
        lineArr = []
        #按照'\t'分割字符串
        currentLine = line.strip().split('\t')
        #把每行中的元素存到lineArr数组中
        for i in range(len(currentLine) - 1):
            lineArr.append(float(currentLine[i]))
        #构建标签数组
        trainLabels.append(float(currentLine[-1]))
        #构建训练矩阵
        trainMat.append(lineArr)
    #使用Logistic分类器进行训练
    #梯度上升算法
    # t, weights = logistic.gradientAscent(trainMat, trainLabels)
    #改进随机梯度上升算法
    weights = logistic.gradientAscent2(trainMat, trainLabels, 500)
    # print weights
    #记录错误数量及总数量
    errorCount = 0.0
    numTestVec = 0.0
    #构建测试矩阵和测试标签数组
    testMat = []; testLabels = []
    # 逐行遍历测试数据集
    for line in frTest.readlines():
        numTestVec += 1.0
        # 定义数组，存储每行元素
        lineArr = []
        # 按照'\t'分割字符串
        currentLine = line.strip().split('\t')
        # 把每行中的元素存到lineArr数组中
        for i in range(len(currentLine) - 1):
            lineArr.append(float(currentLine[i]))
        # print lineArr
        if int(classifyVector(lineArr, weights)) != int(currentLine[-1]):
            errorCount += 1.0
        #计算错误率
    # print numTestVec
    errorRate = float(errorCount) / float(numTestVec) * 100
    #打印
    print 'the errorRate is %.2f%%' % errorRate


"""
函数名：classifyVector(inX, weights)
函数说明：分类函数

Parameters:
    inX - 特征向量
    weights - 权重数组
Return:
    分类结果
Modify:
    2017-11-27
"""
def classifyVector(inX, weights):
    if logistic.sigmoid(sum(inX * weights)) > 0.5:
        return 1.0
    else:
        return 0.0

colicTest()
