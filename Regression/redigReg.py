import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import Regression.regressions as regression

def ridgeRegres(dataMat, labelMat, lam = 0.2):
    xTx = dataMat.T * dataMat
    denom = xTx + np.eye(np.shape(dataMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("非奇异矩阵，不可求逆")
        return
    ws = denom.I * (dataMat.T * labelMat)
    return ws
def ridgeTest(dataMat, labelMat):
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).T
    xMean = np.mean(dataMat, axis=0)
    xVar = np.var(dataMat, axis=0)
    yMeam = np.mean(labelMat, axis=0)
    labelMat = labelMat - yMeam
    dataMat = (dataMat - xMean) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(dataMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(dataMat, labelMat, np.exp(i - 10))
        wMat[i,:] = ws.T
    return wMat
def plotwMat():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    dataMat, labelMat = regression.loadDataSet('abalone.txt')
    redgeWeights = ridgeTest(dataMat, labelMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size = 20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size = 10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()
if __name__ == '__main__':
    plotwMat()