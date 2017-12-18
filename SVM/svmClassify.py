# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: svmClassify.py
@Date: 2017/12/18 16:16
"""
import numpy as np
import time
import operator
from os import listdir
from sklearn.svm import SVC
"""
函数说明：把32×32的图像转化为1×1024的向量

Parameters:
    filename - 文件名
Returns:
    returnVect - 二进制图像的1×1024向量
Modify:
    2017-12-18
"""
def img2vectoer(filename):
    #创建1×1024向量
    returnVect = np.zeros((1, 1024))
    #读取文件内容
    fr = open(filename)
    #逐行写入到向量中
    for i in range(32):
        #读取一行数据
        lineStr = fr.readline()
        #把很一行的32个数据添加到1×1024向量中
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    #返回向量
    return returnVect
"""
函数说明：手写数字分类测试

Parameters:
    无
Returns:
    无
Modify:
    2017-12-28
"""
def handwritingClassTest():
    #保存训练集标签
    hwLabel = []
    #获取训练集文件夹下所有文件
    trainingFileList = listdir('trainingDigits')
    #获取文件个数
    m = len(trainingFileList)
    #初始化训练矩阵
    trainingMat = np.zeros((m, 1024))
    #构建训练矩阵,并从文件名中解析标签
    for i in range(m):
        #获取文件名
        filename = trainingFileList[i]
        #获取文件路径
        filepath = 'trainingDigits/%s' % trainingFileList[i]
        #把32×32的图像转化为1×1024的向量
        returnVect = img2vectoer(filepath)
        #添加到训练矩阵中
        trainingMat[i,:] = returnVect
        #获取标签
        label = int(filename.split('_')[0])
        #添加到标签数组中
        hwLabel.append(label)
    #进行训练
    clf = SVC(C=20000, kernel='rbf')
    clf.fit(trainingMat, hwLabel)
    # 获取测试集文件夹下所有文件
    testFileList = listdir('testDigits')
    # 获取文件个数
    m = len(testFileList)
    #记录错误个数
    errorCount = 0
    #利用模型进行分类
    for i in range(m):
        #获取文件名
        filename = testFileList[i]
        #获取文件路径
        filepath = 'testDigits/%s' %testFileList[i]
        #把32×32的图像转化为1×1024的向量
        returnVect = img2vectoer(filepath)
        #获取预测结果
        result = int(clf.predict(returnVect))
        #获取真实结果
        label = int(filename.split('_')[0])
        if label != result:
            errorCount += 1
    errorRate = float(errorCount) / float(m)
    print '分类错误率为：%.2f%%' % (errorRate * 100)

start = time.time()
handwritingClassTest()
end = time.time()
print end - start




