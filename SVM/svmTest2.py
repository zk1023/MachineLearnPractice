# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: svmTest2.py
@Date: 2017/12/15 17:15
"""
import matplotlib.pyplot as plt
import numpy as np
import random

"""
数据结构，维护所有需要操作的值

Parameters:
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
Modify:
    2017-12-15
"""
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn #数据矩阵
        self.labelMat = classLabels #数据标签
        self.C = C #松弛变量
        self.tol = toler #容错率
        self.m = np.shape(dataMatIn)[0] #矩阵行数
        self.alphas = np.mat(np.zeros((self.m,1))) #初始化alpha参数为0
        self.b = 0 #初始化b为0
        # 根据矩阵行数 初始化误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值
        self.eCache = np.mat(np.zeros((self.m,2)))

"""
函数说明:
    读取数据
Parameters:
    filename - 文件名
Returns:
    dataMat - 数据矩阵
    labelMat- 数据标签
Modify:
    2017-12-15
"""
def loadDataSet(filename):
    fr = open(filename)
    dataMat = []
    labelMat = []
    readlines = fr.readlines()
    for line in readlines:
        arr = line.strip().split('\t')
        dataMat.append((float(arr[0]), float(arr[1])))
        labelMat.append(float(arr[2]))
    fr.close()
    return dataMat, labelMat

"""
函数说明:
    计算误差
Parameters:
    oS - 数据结构
    k - 标号为k的数据
Returns:
    Ek - 标号为k的数据误差
Modify:
    2017-12-15
"""
def calcEk(oS, k):
    fxk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T) + oS.b)
    Ek = fxk - float(oS.labelMat[k])
    return Ek
"""
函数说明:
    随机选择alpha_j的索引值
Parameters:
    i - alpha_i的索引值
    m - alpha参数总数
Returns:
    j - alpha_j的索引值
Modify:
    2017-12-15
"""
def selectJrand(i, m):
    j = i
    while j == i :
        j = int(random.uniform(0, m))
    return j
"""
函数说明:
    内循环启发方式2
Parameters:
    i - 标号为i的数据索引值
    oS - 数据结构
    Ei - 标号为i的数据误差
Returns:
    j, maxK - 标号为j 或者 maxK的数据索引值
    Ej- 标号为j的数据误差
Modify:
    2017-12-15
"""
def selectJ(i, oS, Ei):
    #初始化
    maxK = -1; maxDeltaE = 0; Ej = 0
    #根据Ei更新误差缓存
    oS.eCache[i] = [1, Ei]
    #返回误差不为0的数据索引值
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    #存在不为0的误差
    if len(validEcacheList) > 1:
        #遍历找到最大的Ek
        for k in validEcacheList:
            #不再计算Ei
            if k == i:
                continue
            #计算Ek
            Ek = calcEk(oS, k)
            #计算|Ei - Ek|
            deltaE = abs(Ei - Ek)
            #比较找到最大Ek
            if deltaE > maxDeltaE:
                maxDeltaE = deltaE
                maxK = k
                Ej = Ek
        return maxK, Ej
    #若不存在不为0的误差，随机选择alpha_j的值
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej
"""
函数说明:
    计算Ek 并更新误差缓存
Parameters:
    oS - 数据结构
    k - 标号为k的数据索引值
Returns:
    无
Modify:
    2017-12-15
"""
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]
"""
函数说明:
    修剪alpha
Parameters:
    aj - alpha值
    H - alpha上限
    L - alpha下限
Returns:
    aj - 修剪后alpha的值
Modify:
    2017-12-15
"""
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj
"""
函数说明:
    优化的SMO算法
Parameters:
    i - 标号为i的数据索引值
    oS - 数据结构
Returns:
    1 - 有优化出现
    0- 无优化
Modify:
    2017-12-15
"""
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labelMat[i] * Ei < - oS.tol and oS.alphas[i] < oS.C) or (oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
            H = min(oS.C, oS.alphas[i] + oS.alphas[j])
        if L == H:
            print ("L == H")
            return 0
        eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
        if eta >= 0:
            print ("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        print (oS.alphas[j])
        print (alphaJold)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print ("alpha_j changed too small")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,:] * \
            oS.X[i,:].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i,:] * oS.X[j,:].T
        b2 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * \
                         oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (oS.alphas[i] > 0) and (oS.alphas[i] < oS.C):
            oS.b = b1
        elif (oS.alphas[j] > 0) and (oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0
"""
函数说明：完整的线性SMO算法
Patamters:
    dataMatIn - 数据矩阵 
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
    maxInter - 最大迭代次数
"""
def smoP(dataMatIn, classLabels, C, toler, maxIter):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while iter < maxIter and alphaPairsChanged > 0 or entireSet:
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print ("全体样本遍历：第%d次迭代 样本:%d， alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print ("非边界遍历:第%d次迭代 样本:%d,alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet =False
        elif alphaPairsChanged == 0:
            entireSet = True
        print ("迭代次数:%d" % iter)
    return oS.b, oS.alphas
"""
函数说明：计算w
Parameters:
    dataArr - 数据矩阵
    classLabels - 数据标签
    alphas - alphas值
"""
def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i,:].T)
    return w
"""
函数说明：分类数据可视化
Parameters:
    dataMat - 数据矩阵
    classLabels - 数据标签
    w - 直线法向量
    b - 直线截距
Returns:
    无
"""
def showClassifier(dataMat, classLabels, w, b):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[1]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1,x2], [y1,y2])
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidths=1.5, edgecolors='red')
    plt.show()

if __name__ == '__main__':
    dataArr, classLabels = loadDataSet("testSet.txt")
    b, alphas = smoP(dataArr, classLabels, 0.6, 0.001, 40)
    w = calcWs(alphas, dataArr, classLabels)
    showClassifier(dataArr, classLabels, w, b)




