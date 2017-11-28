# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: skLearnClassify.py
@Date: 2017/11/28 15:44
"""
import numpy
from sklearn.linear_model import LogisticRegression

"""
函数名：colicSklearn()
函数说明：使用sklearn构建逻辑回归分类器

Parameters:
    无
Return:
    分类准确率
Modify:
    2017-11-28
"""
def colicSklearn():
    #导入数据
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    #定义训练矩阵、标签数组
    trainMat = []; trainLabel = []
    # 定义测试矩阵、标签数组
    testMat = [];
    testLabel = []
    #遍历训练集
    for line in frTrain.readlines():
        #按照'\t'切分字符串
        currentLine = line.strip().split('\t')
        #保存每行元素
        lineArr = []
        #把元素放到数组中
        for i in range(len(currentLine) - 1):
            lineArr.append(float(currentLine[i]))
        trainLabel.append(float(currentLine[-1]))
        trainMat.append(lineArr)
    #遍历测试集
    for line in frTest.readlines():
        #按照'\t'切分字符串
        currentLine = line.strip().split('\t')
        #保存每行元素
        lineArr = []
        #把元素放到数组中
        for i in range(len(currentLine) - 1):
            lineArr.append(float(currentLine[i]))
        testLabel.append(float(currentLine[-1]))
        testMat.append(lineArr)
    #使用sklearn构建分类器
    classifier = LogisticRegression(solver='liblinear', max_iter=10).fit(trainMat, trainLabel)
    test_accuracy = classifier.score(testMat, testLabel) * 100
    print 'the correct rate is %f%%' % test_accuracy


colicSklearn()
