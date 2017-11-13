# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: createBranch.py
@Date: 2017/9/24 20:34
"""
from math import log

#计算商
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    shannonEnt = 0.0
    for featVec in dataSet:
        vote = featVec[-1]
        labelCounts[vote] = labelCounts.get(vote, 0) + 1
        # if vote not in labelCounts.keys():
        #     labelCounts[vote] = 0
        # labelCounts[vote] += 1
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt = shannonEnt - prob*log(prob,2)
    return shannonEnt

#创建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
dataSet, labels = createDataSet()
print dataSet, labels
print calcShannonEnt(dataSet)

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        print featVec
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# result = splitDataSet(dataSet, 0, 0)
# print result

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return i