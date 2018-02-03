from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import Regression.regressions as regression
import Regression.partRegression as part

def rssError(lableMat, yHatArr):
    return  ((lableMat - yHatArr) ** 2).sum()

if __name__ == '__main__':
    dataMat, labelMat = regression.loadDataSet('abalone.txt')
    print('训练集与测试集相同：局部加权线性回归，核k的大小对预测的影响：')
    yHat01 = part.lwlrTest(dataMat[0:99], dataMat[0:99], labelMat[0:99], 0.1)
    yHat1 = part.lwlrTest(dataMat[0:99], dataMat[0:99], labelMat[0:99], 1)
    yHat10 = part.lwlrTest(dataMat[0:99], dataMat[0:99], labelMat[0:99], 10)
    print('k=0.1时，误差大小为:',rssError(labelMat[0:99], yHat01.T))
    print('k=1时，误差大小为:', rssError(labelMat[0:99], yHat1.T))
    print('k=10时，误差大小为:', rssError(labelMat[0:99], yHat10.T))
    print('')
    print('训练集与测试集不同:局部加权线性回归，核k的大小是越小越好吗？更换数据集，测试结果如下：')
    yHat01 = part.lwlrTest(dataMat[100:199], dataMat[0:99], labelMat[0:99], 0.1)
    yHat1 = part.lwlrTest(dataMat[100:199], dataMat[0:99], labelMat[0:99], 1)
    yHat10 = part.lwlrTest(dataMat[100:199], dataMat[0:99], labelMat[0:99], 10)
    print('k=0.1时，误差大小为:', rssError(labelMat[0:99], yHat01.T))
    print('k=1时，误差大小为:', rssError(labelMat[0:99], yHat1.T))
    print('k=10时，误差大小为:', rssError(labelMat[0:99], yHat10.T))
    print('')
    print('训练集与测试集不同：简单线性回归与k=1时的局部加权线性回归对比：')
    print('k=1时，误差大小为：', rssError(labelMat[100:199], yHat1.T))
    ws = regression.standRegres(dataMat[0:99], labelMat[0:99])
    yHat = np.mat(dataMat[100:199]) * ws
    print('简单线性回归误差大小为 ', rssError(labelMat[100:199], yHat.T.A))

