"""  
@author: Zhangkai
@file: kNN2.py
@Date: 2017/9/4 20:06
"""
from numpy import *
import operator

def createDataSet():
    groups = array([[1.0, 1.0], [1.0, 1.1], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return groups, lables

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
def main():
    groups,labels = createDataSet()
    result = classify0([3, 3], groups, labels, 3)
    print result
main()