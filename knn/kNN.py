from numpy import *


def createDataSet():
    groups = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return groups, labels
jj = mat([[1, 2, 3, 0], [8, 8, 8, 8], [9, 1, 0, 4]])
# print jj.min(0)
# print jj.max(0)
ranges = jj.max(0) - jj.min(0)
# print ranges
# print jj[:,0]
# int = {'a':1, 'b':2, 'c':3}
# print int['a']
# j = [1, 2, 3, 4, 5]
# print j[-2]