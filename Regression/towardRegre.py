import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import Regression.regressions as regression
import Regression.redigReg as redig

#标准化
def regularize(xMat, yMat):
    dataMat = xMat.copy()
    labelMat = yMat.copy()
    xMean = np.mean(dataMat)
    xVar = np.var(dataMat, axis=0)
    dataMat = (dataMat - xMean) / xVar
    yMean = np.mean(dataMat, axis=0)
    labelMat = labelMat - yMean
    return dataMat, labelMat

#计算平方误差
def rssError(yArr, yHatArr):
    return  ((yArr - yHatArr)**2).sum()

#前进逐步线性回归
def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    dataMat, labelMat = regularize(xMat, yMat)
    m, n = np.shape(dataMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        lowestError = float('inf')
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = dataMat * wsTest
                rssE = rssError(labelMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

def plotstageWiseMat():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    dataMat, labelMat = regression.loadDataSet('abalone.txt')
    returnMat = stageWise(dataMat, labelMat, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title(u'前向逐步回归：迭代次数与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()
if __name__ == '__main__':
    plotstageWiseMat()