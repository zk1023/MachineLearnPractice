# -*- coding: utf-8 -*-：
"""
@author: Zhangkai
@file: kNN8.py
@Date: 2017/9/11 9:37
"""
from numpy import *
import operator

def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        vote = labels[sortDistances[i]]
        classCount[vote] = classCount.get(vote, 0) + 1
    sortResult = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)
    return sortResult[0][0]
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    index = 0
    classLabelVector = []
    for line in arrayOfLines:
        # strip() 方法用于移除字符串头尾指定的字符（默认为空格）
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        index += 1
        classLabelVector.append(int(listFromLine[-1]))
    return returnMat, classLabelVector
def autoNum(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    print normDataSet
    return normDataSet, ranges, minVals
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNum(datingDataMat)
    m = normMat.shape[0]
    numTestVec = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVec):
        classifierResult = classify0(datingDataMat[i,:], datingDataMat[numTestVec:m,:], datingLabels[numTestVec:m], 3)
        print "the classifier came back with: %d, the real answer is: %d"%(classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print "the total error rate is :%f"%(errorCount/float(numTestVec))
datingClassTest()

