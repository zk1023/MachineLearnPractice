# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: createBranch2.py
@Date: 2017/9/25 16:17
"""
from math import *
import operator
#创建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
dataSet, labels = createDataSet()
print dataSet, labels

#计算上
def calcShannonEnt(dataSet):
    shannonEnt = 0.0
    length = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        vote = featVec[-1]
        labelCount[vote] = labelCount.get(vote, 0) + 1
    for key in labelCount:
        prob = labelCount[key]/float(length)
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt

#划分
def spiltDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            currentDataSet = featVec[:axis]
            currentDataSet.extend(featVec[axis+1:])
            retDataSet.append(currentDataSet)
    return retDataSet

#选择
def chooseBestFeatureToSpilt(dataSet):
    num = len(dataSet[0]) - 1
    baseShannonEnt = calcShannonEnt(dataSet)
    bestShannonEnt = 0.0
    bestFeature = -1
    for i in range(num):
        shannonEnt = 0.0
        list = [example[i] for example in dataSet]
        total = set(list)
        for value in total:
            retDataSet = spiltDataSet(dataSet,i,value)
            prob = len(retDataSet)/float(len(dataSet))
            shannonEnt += prob*calcShannonEnt(retDataSet)
        infoGain = baseShannonEnt - shannonEnt
        if infoGain > bestShannonEnt:
            bestShannonEnt = infoGain
            bestFeature = i
    return bestFeature
print chooseBestFeatureToSpilt(dataSet)

#标签分类
def majorityCnt(labels):
    classCount = {}
    for vote in labels:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
print majorityCnt(labels)

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSpilt(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(spiltDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
print createTree(dataSet,labels)



#计算商
# def calcShannonEnt(dataSet):
#     numEntries = len(dataSet)
#     shannonEnt = 0.0
#     labelsCount = {}
#     for featVec in dataSet:
#         currentLabel = featVec[-1]
#         labelsCount[currentLabel] = labelsCount.get(currentLabel, 0) + 1
#     for key in labelsCount:
#         prop = float(labelsCount[key])/numEntries
#         shannonEnt -= prop*log(prop, 2)
#     return shannonEnt
# print calcShannonEnt(dataSet)
#
# #划分数据集
# def splitDataSet(dataSet, axis, value):
#     retDataSet = []
#     for list in dataSet:
#         if list[axis] == value:
#             reducedFeatVec = list[:axis]
#             reducedFeatVec.extend(list[axis+1:])
#             retDataSet.append(reducedFeatVec)
#     return retDataSet
#
# #选择最好方法
# def chooseBestFeatureToSplit(dataSet):
#     numFeatures = len(dataSet[0]) - 1
#     baseEntries = calcShannonEnt(dataSet)
#     bestInfoGain = 0.0
#     bestFeature = -1
#     for i in range(numFeatures):
#         featList = [example[i] for example in dataSet]
#         uniqueVals = set(featList)
#         newEntropy = 0.0
#         for value in uniqueVals:
#             subDataSet = splitDataSet(dataSet,i,value)
#             prob = len(subDataSet)/float(len(dataSet))
#             newEntropy += prob*calcShannonEnt(subDataSet)
#         infoGain = baseEntries - newEntropy
#         print infoGain
#         if infoGain > bestInfoGain:
#             bestInfoGain = infoGain
#             bestFeature = i
#     return bestFeature
#
# def majorityCnt(classList):
#     classCount = {}
#     for vote in classList:
#         classCount[vote] = classCount.get(vote, 0) + 1
#     sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
#     return  sortedClassCount[0][0]
#
# def createTree(dataSet, labels):
#     classList = [example[-1] for example in dataSet]
#     if classList.count(classList[0]) == len(classList):
#         return classList[0]
#     if len(dataSet[0]) == 1:
#         return majorityCnt(classList)
#     bestFeat = chooseBestFeatureToSplit(dataSet)
#     bestFeatLabel = labels[bestFeat]
#     myTree = {bestFeatLabel:{}}
#     del(labels[bestFeat])
#     featValues = [example[bestFeat] for example in dataSet]
#     uniqueVals = set(featValues)
#     for value in uniqueVals:
#         subLabels = labels[:]
#         myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
#     return myTree
# myTree = createTree(dataSet, labels)
# print myTree