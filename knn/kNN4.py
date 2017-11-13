"""  
@author: Zhangkai
@file: kNN4.py
@Date: 2017/9/6 10:53
"""
from numpy import *
import matplotlib.pyplot as plt
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    int = {'didntLike':1 ,'smallDoses':2, 'largeDoses':3}
    for line in arrayOfLines:
        line = line.strip()
        listFromline = line.split('\t')
        returnMat[index,:] = listFromline[0:3]
        classLabelVector.append(int[listFromline[-1]])
        index += 1
    return returnMat, classLabelVector
datingDataMat, datingLabel = file2matrix("datingTestSet.txt")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabel), 15.0*array(datingLabel))
plt.show()
