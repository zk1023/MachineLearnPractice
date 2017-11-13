# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: kNN10.py
@Date: 2017/9/15 17:02
"""
from numpy import *
from os import listdir
import operator

def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = dataSet - tile(intX, (dataSetSize,1))
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDist = distances.argsort()
    classCount = {}
    for i in range(k):
        vote = labels[sortedDist[i]]
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedResult = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedResult[0][0]

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
# datingMat = img2vector('digits/trainingDigits/0_13.txt')
# print datingMat[0,32:63]

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        hwLabels.append(int(classNumStr))
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s'%fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifyResult = classify0(vectorUnderTest, trainingMat, hwLabels, 10)
        if classifyResult != classNumStr:
            errorCount += 1.0
            print "the classifier came back with %d, the real answer is %d" % (classifyResult, classNumStr)
    print "the total number of errors is: %d" % errorCount
    print "the total error rate is: %f" % (errorCount/float(mTest))
handwritingClassTest()

