"""
@author: Zhangkai
@file: kNN3.py
@Date: 2017/9/5 09:00
"""
from numpy import  *
import matplotlib.pyplot as plt

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    number0fLines = len(arrayOLines)
    print number0fLines
    returnMat = zeros((number0fLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if listFromLine[-1] == 'largeDoses':
            k = 3
        elif listFromLine[-1] == 'smallDoses':
            k = 2
        elif listFromLine[-1] == 'didntLike':
            k = 1
        else:
            k = 0
        classLabelVector.append(k)
        index += 1
    dictClassLabel = set(classLabelVector)
    print dictClassLabel
    return returnMat,classLabelVector
datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
print datingDataMat
print datingLabels[0:20]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()