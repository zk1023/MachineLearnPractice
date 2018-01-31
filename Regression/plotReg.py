import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    fr = open(filename)
    numFeat =  len(fr.readline().split('\t')) - 1
    xArr = []
    yArr = []
    for line in fr.readlines():
        lineArr = []
        curArr = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curArr[i]))
        xArr.append(lineArr)
        yArr.append(float(curArr[-1]))
    return xArr, yArr

def plotDataSet():
    xArr, yArr = loadDataSet("ex0.txt")
    n = len(xArr)
    xcord = []
    ycord = []
    for i in range(n):
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s = 20, c = 'blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

if __name__ == '__main__':
    plotDataSet()



