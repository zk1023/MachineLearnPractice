# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: simpleClassify.py
@Date: 2018/1/19 20:41
"""
import numpy as np
import simpleDataPlot
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.font_manager import  FontProperties



"""
加载数据集
Parameters:
    filename - 文件名（路径）
Returns:
    dataMat - 数据集
    classLabels - 数据标签
Modify:
    2018-1-22
"""
def loadSimpleData(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currentLine) - 1):
            lineArr.append(float(currentLine[i]))
        labelMat.append(float(currentLine[-1]))
        dataMat.append(lineArr)
    return dataMat, labelMat

"""
单层决策树分类
Paramters:
    dataMatrix - 数据矩阵
    dimen - 第dimen列，即第dimen个特征
    threshVal - 阈值
    threshIn - 标志
Returns:

Modify:
    2018-1-19
"""
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    #初始化类别都为1，若小于阈值或者大于阈值 使其为-1
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == "lt":
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

"""
单层决策树分类函数
Parameters:
    dataArr - 数据矩阵
    classLabels - 数据标签
    D - 样本权重
Returns:
    bestStump - 最佳单层决策树信息
    minError - 最小误差
    bestClasEst - 最佳分类结果
Modify:
    2018-1-19
"""
def buildStump(dataArr, classLabels, D):
    #转化为矩阵
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    #获取矩阵规格
    m, n = np.shape(dataArr)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    #最小化误差初始化为无穷大
    minError = float('inf')
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        #计算步长
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                #计算阈值
                threshVal = (rangeMin + float(j) * stepSize)
                #计算分类结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.ones((m, 1))
                #分类正确，赋值为0
                errArr[predictedVals == labelMat] = 0
                #计算误差
                weightedError = D.T * errArr
                print "split: dim%d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return  bestStump, minError, bestClasEst

"""
adaboost迭代分类
Parameters:
    dataArr - 数据矩阵
    classLabels - 数据标签
    numIt - 迭代次数
Returns:
    weakClassArr - 
    aggClassEst - 权重
Modify:
    2018-1-22
"""
def adaBoostTrainDS(dataArr, classLabels, numIt = 50):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    #初始化权重
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        #构建单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D:", D.T
        #计算若学习算法的权重，使error不为0
        alpha = float(0.5 * np.log((1 - error) / max(error, 1e-16)))
        #存储弱学习算法权重
        bestStump['alpha'] = alpha
        #存储单层决策树
        weakClassArr.append(bestStump)
        print "classEst:", classEst.T
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        #计算adaboost误差，当误差为0时 跳出循环
        aggClassEst += alpha * classEst
        print "aggClassEst:", aggClassEst
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate
        if errorRate == 0.0:
            break
    return  weakClassArr, aggClassEst

"""
adaboost分类函数
Parameters:
    dataToClass - 待分类样例
    classifierArr - 训练好的分类器
Returns:
    分类结果
Modify:
    2018-1-22
"""
def adaClassify(dataToClass, classifierArr):
    dataMatrix = np.mat(dataToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return np.sign(aggClassEst)

"""
绘制ROC曲线
Parameters:
    predStrengths - 分类器的预测强度
    classLabels - 类别
Returns:
    无
Modify:
    2018-1-22
"""
def plotROC(predStrengths, classLabels):
    font = FontProperties(fname="C:\Windows\Fonts\simsun.ttc", size=14)
    #绘制光标的位置
    cur = (1.0, 1.0)
    #计算AUC
    ySum = 0.0
    #统计正类数量
    numPosClas = np.sum(np.array(classLabels) == 1.0)
    #x轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)
    #y轴步长
    yStep = 1 / float(numPosClas)
    #预测强度 排序
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0,1], [0,1], 'b--')
    plt.title("AdaBoost", FontProperties = font)
    plt.xlabel('jiayanglv', FontProperties = font)
    plt.ylabel('zhenyanglv', FontProperties = font)
    ax.axis([0, 1, 0, 1])
    print 'AUC面积为：', ySum * xStep
    plt.show()



# if __name__ == '__main__':
#     # dataMatrix, classLabels = simpleDataPlot.loadSimpleData()
#     # D = np.mat(np.ones((5, 1)) / 5)
#     # bestStump, minError, bestClaEst = buildStump(dataMatrix, classLabels, D)
#     # print 'bestStump:\n', bestStump
#     # print 'minError:\n', minError
#     # print 'bestClasEst:\n', bestClaEst
#     # weakClassArr, aggClassEst = adaBoostTrainDS(dataMatrix, classLabels)
#     # print weakClassArr
#     # print aggClassEst
#     # print adaClassify([[0, 0], [5, 5]], weakClassArr)
#     dataArr, labelArr = loadSimpleData("horseColicTraining2.txt")
#     weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr)
#     testArr, testlabelArr = loadSimpleData("horseColicTest2.txt")
#     print weakClassArr
#     predictions = adaClassify(dataArr, weakClassArr)
#     errArr = np.mat(np.ones((len(dataArr), 1)))
#     print 'trainSet errorRate is %.3f%%' % float(errArr[predictions != np.mat(labelArr).T].sum() / len(dataArr) * 100)
#     predictions = adaClassify(testArr, weakClassArr)
#     errArr = np.mat(np.ones((len(testArr), 1)))
#     print 'testSet errorRate is %.3f%%' % float(errArr[predictions != np.mat(testlabelArr).T].sum() / len(testArr) * 100)
#
# if __name__ == '__main__':
#     dataArr, labelArr = loadSimpleData("horseColicTraining2.txt")
#     testArr, testlabelArr = loadSimpleData("horseColicTest2.txt")
#     bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=10)
#     bdt.fit(dataArr, labelArr)
#     predictions = bdt.predict(dataArr)
#     errArr = np.mat(np.ones((len(dataArr), 1)))
#     print 'trainSet errorRate is %.3f%%' % float(errArr[predictions != labelArr].sum() / len(dataArr) * 100)
#     predictions = bdt.predict(testArr)
#     errArr = np.mat(np.ones((len(testArr), 1)))
#     print 'testSet errorRate is %.3f%%' % float(errArr[predictions != testlabelArr].sum() / len(testArr) * 100)


if __name__ == '__main__':
    dataArr, labelArr = loadSimpleData("horseColicTraining2.txt")
    weakClasArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
    plotROC(aggClassEst.T, labelArr)