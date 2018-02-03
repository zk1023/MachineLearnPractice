import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    fr = open(filename)
    numFeat = len(fr.readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    for line in fr.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(numFeat):
            lineArr.append(float(currentLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currentLine[-1]))
    return dataMat, labelMat

def standRegres(dataMat, labelMat):
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).T
    xTx = dataMat.T * dataMat
    if np.linalg.det(xTx) == 0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = xTx.I * (dataMat.T * labelMat)
    return ws

def plotRegression():
    dataMat, labelMat = loadDataSet("ex0.txt")
    ws = standRegres(dataMat, labelMat)
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat)
    xCopy = dataMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:,1], yHat, c = 'red')
    ax.scatter(dataMat[:,1].flatten().A[0], labelMat.flatten().A[0], s=20, c = 'blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


# if __name__ == '__main__':
#     dataMat, lableMat = loadDataSet("ex0.txt")
#     ws = standRegres(dataMat, lableMat)
#     dataMat = np.mat(dataMat)
#     lableMat = np.mat(lableMat)
#     yHat = dataMat * ws
#     print(np.corrcoef(yHat.T, lableMat))

if __name__ == '__main__':
    plotRegression()