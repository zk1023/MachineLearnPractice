# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: kNN9.py
@Date: 2017/9/14 8:48
"""
from numpy import *
import operator
def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = dataSet - tile(intX, (dataSetSize, 1))
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortDist = distances.argsort()
    classCount = {}
    for i in range(k):
        vote = labels[sortDist[i]]
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedCountLables = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)
    return sortedCountLables[0][0]
def classify1(intX, dataSet, labels, k):
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
    classLabelVectors = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLines = line.split('\t')
        returnMat[index,:] = listFromLines[0:3]
        index = index + 1
        classLabelVectors.append(int(listFromLines[-1]))
    return returnMat, classLabelVectors
def file3matrix(filename):
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
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    m = dataSet.shape[0]
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
def autoNorm1(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLables = file2matrix('datingTestSet2.txt')
    normDataMat, ranges, minVals = autoNorm(datingDataMat)
    m = normDataMat.shape[0]
    number = int(hoRatio*m)
    error = 0.0
    for i in range(number):
        classifyResult = classify1(normDataMat[i,:], normDataMat[number:m,:], datingLables[number:m], 3)
        print "the classify return result is %d, the real answer is %d"%(classifyResult, datingLables[i])
        if classifyResult != datingLables[i]:
            error = error + 1
    print "the ratio is %f"%(error/float(number))
    print error

# datingClassTest()
def datingClassTest1():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVec = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVec):
        classifierResult = classify1(normMat[i,:], normMat[numTestVec:m,:], datingLabels[numTestVec:m], 3)
        print "the classifier came back with: %d, the real answer is: %d"%(classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount = errorCount + 1
    print "the total error rate is :%f"%(errorCount/float(numTestVec))
    print errorCount
datingClassTest()
