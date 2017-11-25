# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: emailClassify.py
@Date: 2017/11/17 9:51
"""
from numpy import *
import re
import bayes_test
import email

"""
函数名：textParse(bigString)
函数说明：接收一个大字符串并解析为字符串列表

Parameter:
    bigString - 大字符串
Return:
    list - 字符串列表
Modify:
    2017-11-17
"""
def textParse(bigString):
    #把特殊符号（非字母非数字）作为切分标志来切分字符串
    listOfTokens = re.split(r'\W*', bigString)
    #把词汇列表中长度大于2的留下来，并统一设定为小写字母
    list = [tok.lower() for tok in listOfTokens if len(tok) > 2]
    return list

"""
函数名：spamTest()
函数说明：对贝叶斯垃圾分类器进行自动化处理

Parameter:
    无
Return:
    无（输出分类效果）
Modify:
    2017-11-17
"""
def spamTest():
    #保存文档
    docList = []
    #保存文档对应类别
    classList = []
    #记录所有词条
    fullTest = []
    #遍历25个文件
    for i in range(1,26):
        #读取每封垃圾邮件
        content = open('D:\Study\github\MachineLearnPractice\email\spam\%d.txt'%i).read()
        #把邮件内容转化为词汇列表
        document = textParse(content)
        #构建词汇列表集合
        docList.append(document)
        #记录邮件类别（垃圾邮件）
        classList.append(1)
        #记录邮件中所有单词
        fullTest.extend(document)

        # 读取每封非垃圾邮件
        content = open('D:\Study\github\MachineLearnPractice\email\ham\%d.txt'%i).read()
        # 把邮件内容转化为词汇列表
        document = textParse(content)
        # 构建词汇列表集合
        docList.append(document)
        # 记录邮件类别（非垃圾邮件）
        classList.append(0)
        # 记录邮件中所有单词
        fullTest.extend(document)
    #构建词汇表（所有不重复单词）
    vocabList = bayes_test.createVocabList(docList)
    #采用留存交叉验证进行分类
    #随机选出十个邮件作为测试集，剩下的作为训练集
    #构建存储训练集索引值的列表
    trainSet = range(50)
    #构建存储测试集索引值的列表
    testSet = []
    for i in range(10):
        #产生从0到训练集长度的随机数
        randIndex = int(random.uniform(0, len(trainSet)))
        #选取测试集
        testSet.append(trainSet[randIndex])
        #从训练集中去除测试集
        del(trainSet[randIndex])
    #构建贝叶斯分类器
    #建立训练矩阵
    trainMat = []
    #用于存储类别标签
    trainClass = []
    #遍历训练集
    for i in trainSet:
        #构建词向量训练矩阵
        trainMat.append(bayes_test.bagOfWords2VecMN(vocabList, docList[i]))
        #添加邮件向量对应的类别
        trainClass.append(classList[i])
    #获取垃圾邮件条件概率、非垃圾邮件的条件概率和邮件属于垃圾邮件的概率
    p0V, p1V, pSpam = bayes_test.trainNB0(trainMat, trainClass)
    #记录邮件分类错误率
    errorCount = 0
    #开始测试
    for i in testSet:
        #获取分类器输出标签
        label = bayes_test.classifyNB(bayes_test.bagOfWords2VecMN(vocabList, docList[i]), p0V, p1V, pSpam)
        #与真实类别进行比较，并记录错误数量
        if label != classList[i]:
            errorCount += 1
            print 'classify error: ', docList[i]
    #输出错误率
    result = float(errorCount) / len(testSet)
    return result

if __name__ == '__main__':
    count = 0.0
    for i in range(10):
        count += spamTest()
    print 'the error rate is %.2f%%'%(float(count)*10)

