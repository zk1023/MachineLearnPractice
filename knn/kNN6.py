"""  
@author: Zhangkai
@file: kNN6.py
@Date: 2017/9/7 10:12
"""
from numpy import *
import operator
import matplotlib.pyplot as plt
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    index = 0
    classLabelVector = []
    # lables = {'didntLike':1, 'smallDoses':2, 'largeDoses':3}
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        index += 1
        classLabelVector.append(int(listFromLine[-1]))
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,  (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedClassCount = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabels = labels[sortedClassCount[i]]
        classCount[voteIlabels] = classCount.get(voteIlabels, 0) + 1
    sortedCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedCount[0][0]

def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    m = normDataSet.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classfierResult = classify0(normDataSet[i,:], normDataSet[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d  the real answer is: %d " % (classfierResult, datingLabels[i])
        if classfierResult != datingLabels[i]:
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount, numTestVecs
datingClassTest()



