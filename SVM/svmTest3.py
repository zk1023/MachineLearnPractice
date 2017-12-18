# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: svmTest3.py
@Date: 2017/12/17 18:53
"""
import matplotlib.pyplot as plt
import numpy as np
import svmTest2
import time
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
        j = svmTest2.selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej
"""
函数说明：初始化 数据结构，维护所有需要操作的值
Parameters:
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
    kTup - 包含核函数信息的元组，第一个参数放核函数类别，第二个参数存放必要的核函数需要用到的参数
"""
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))#初始化核k
        #计算所有数据的核k
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

"""
函数说明：通过核函数将数据转换到更高维的空间
Parameters:
    X - 数据矩阵
    A - 单个数据向量
    kTup - 包含核函数信息的元组
Returns:
    k - 计算的核K
"""
def kernelTrans(X, A, kTup):
    """
    通过核函数将数据转换更高维的空间
    Parameters：
        X - 数据矩阵
        A - 单个数据的向量
        kTup - 包含核函数信息的元组
    Returns:
        K - 计算的核K
    """
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin': K = X * A.T                       #线性核函数,只进行内积。
    elif kTup[0] == 'rbf':                                 #高斯核函数,根据高斯核函数公式进行计算
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))                     #计算高斯核K
    else: raise NameError('核函数无法识别')
    return K
def kernelTrans1(X, A, kTup):
    #获取矩阵的维度
    m, n = np.shape(X)
    #初始化
    K = np.mat(np.zeros((m, 1)))
    #如果是线性核函数，进行内机运算
    if kTup[0] == 'lin':
        K = X * A.T
    #高斯核函数，根据高斯核函数公式计算
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / -1 * kTup[1] ** 2)
    else:
        raise NameError("核函数无法识别")
    return K
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
            print "L == H"
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print "eta >= 0"
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = svmTest2.clipAlpha(oS.alphas[j], H, L)
        svmTest2.updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print "alpha_j changed too small"
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        svmTest2.updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (oS.alphas[i] > 0) and (oS.alphas[i] < oS.C):
            oS.b = b1
        elif (oS.alphas[j] > 0) and (oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
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
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while iter < maxIter and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print "全体样本遍历：第%d次迭代 样本:%d， alpha优化次数:%d" % (iter, i, alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print "非边界遍历:第%d次迭代 样本:%d,alpha优化次数:%d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet =False
        elif alphaPairsChanged == 0:
            entireSet = True
        print "迭代次数:%d" % iter
    return oS.b, oS.alphas
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
    fxk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fxk - float(oS.labelMat[k])
    return Ek
"""
函数说明：测试函数
Parameters:
    k1 - 使用高斯核函数时的表示到达率
Returns： 
    无
"""
def testRbf(k1 = 1.2):
    dataMat, labelMat = svmTest2.loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataMat, labelMat, 200, 0.0001, 100, ('rbf', k1))
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()
    #获取支持向量
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print "支持向量的个数：%d" % np.shape(sVs)[0]
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelMat[i]):
            errorCount += 1
    print "训练集错误率为：%.2f%%" % ((float(errorCount) / m) * 100)
    dataMat, labelMat = svmTest2.loadDataSet('testSetRBF2.txt')
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelMat[i]):
            errorCount += 1
    print "测试集错误率为：%.2f%%" % ((float(errorCount) / m) * 100)
    return (float(errorCount) / m) * 100

if __name__ == '__main__':
    start = time.time()
    count = 0
    testRbf()
    end = time.time()
    elapsed = end - start
    print "Time taken: ", elapsed, "seconds."

