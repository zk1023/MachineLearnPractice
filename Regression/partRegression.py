import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import Regression.regressions as regression

def lwlr(testPoint, dataMat, labelMat, k = 1.0):
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).T
    m = np.shape(dataMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - dataMat[j,:]
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))
    xTx = dataMat.T * (weights * dataMat)
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = xTx.I * (dataMat.T * (weights * labelMat))
    return testPoint * ws

def lwlrTest(testArr, dataMat, labelMat, k = 1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], dataMat, labelMat, k)
    return yHat

def plotlwlrRegression():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    dataMat, labelMat = regression.loadDataSet('ex0.txt')
    yHat_1 = lwlrTest(dataMat, dataMat, labelMat, 1.0)
    yHat_2 = lwlrTest(dataMat, dataMat, labelMat, 0.01)
    yHat_3 = lwlrTest(dataMat, dataMat, labelMat, 0.003)
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat)
    srtInd = dataMat[:,1].argsort(0)
    xSort = dataMat[srtInd][:,0,:]
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10,8))
    axs[0].plot(xSort[:,1], yHat_1[srtInd], c='red')
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c='red')
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c='red')
    axs[0].scatter(dataMat[:,1].flatten().A[0], labelMat.flatten().A[0], s = 20, c = 'blue', alpha=.5)
    axs[1].scatter(dataMat[:, 1].flatten().A[0], labelMat.flatten().A[0], s=20, c='blue', alpha=.5)
    axs[2].scatter(dataMat[:, 1].flatten().A[0], labelMat.flatten().A[0], s=20, c='blue', alpha=.5)
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线，k=1.0',FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线，k=0.01', FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线，k=0.003', FontProperties=font)
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()

if __name__ == '__main__':
    plotlwlrRegression()


