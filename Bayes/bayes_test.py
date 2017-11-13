# -*- coding: utf-8 -*-：
"""
@author: Zhangkai
@file: bayes_test.py
@Date: 2017/11/12 11:25
"""
import dataSet
from numpy import *

"""
函数名:createVocabList(dataSet)
函数说明:将切分的样本词条整理成不重复的词条列表

Parameters:
    dataSet1 - 词条切分后的文档集合
Returns:
    vocabSet - 不重复词条的列表(词汇表)
Modify:
    2017-11-12
"""
def createVocabList(dataSet1):
    #创建一个空集
    vocabSet = set([])
    #遍历文档集合
    for document in dataSet1:
        #创建两个集合的并集
        vocabSet = vocabSet | set(document)
    #返回词汇表
    return list(vocabSet)

# postingList, classVct = dataSet.loadDataSet()
# print createVocabList(postingList)

"""
函数名:setOfWords2Vec(vocabList, inputSet)
函数说明:把文档转化为词条向量

Parameters:
    vocabList - 词汇表
    inputSet - 词条文档
Returns:
    returnVec - 词条向量
Modify:
    2017-11-12
"""
def setOfWords2Vec(vocabList, inputSet):
    #定义一个值为全0的词条向量
    returnVec = [0]*len(vocabList)
    #遍历每个词条
    for word in inputSet:
        #如果词条在词汇表中,词条向量中的对应值设为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in the vocabulary"%word
    #返回词条向量
    return returnVec

"""
函数名:trainNB0(trainMatrix,trainCategory)
函数说明:朴素贝叶斯训练函数

Parameters:
    trainMatrix - 各个文档的词向量构成的矩阵
    trainCategory - 类标签向量
Returns:
    p0Vect - 侮辱类词条的条件概率
    p1Vect - 非侮辱类词条的条件概率
    pAbusive - 侮辱类文档所占的概率
Modify:
    2017-11-13
"""
def trainNB0(trainMatrix, trainCategory):
    #获取训练文档的数量
    numTrainDocs = len(trainMatrix)
    #获取词向量的长度（单词个数）,即词汇表长度
    numWords = len(trainMatrix[0])
    #计算侮辱类文档所占概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #构造侮辱类词向量，并初始化为0
    p1Num = zeros(numWords)
    #用来记录侮辱类单词总数量
    p1Denom = 0.0
    #构造非侮辱类词向量，并初始化为0
    p0Num = zeros(numWords)
    #用来记录非侮辱类单词总数量
    p0Denom = 0.0
    #遍历文档
    for i in range(numTrainDocs):
        #类别为侮辱类文档
        if trainCategory[i] == 1:
            #统计侮辱类词向量
            p1Num = p1Num + trainMatrix[i]
            #统计侮辱类词条总数
            p1Denom += sum(trainMatrix[i])
        else:
            # 统计非侮辱类词向量
            p0Num = p0Num + trainMatrix[i]
            # 统计非侮辱类词条总数
            p0Denom += sum(trainMatrix[i])
    #计算侮辱类词条的条件概率（p(w|c)）
    p1Num = p1Num/p1Denom
    #计算非侮辱类词条的条件概率
    p0Num = p0Num/p0Denom
    #返回非侮辱类词条的条件概率、侮辱类词条的条件概率、侮辱类词条总数所占概率
    return p0Num, p1Num, pAbusive

"""
函数名:classifyNB(vec2Classify, p0Vec, p1Vec, pClass1)
函数说明:朴素贝叶斯分类函数

Parameters:
    vec2Classify - 待分类文档的词向量
    p0Vec - 非侮辱类词条的条件概率数组
    p1Vec - 侮辱类词条的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    0 - 文档属于非侮辱类
    1 - 文档属于侮辱类
Modify:
    2017-11-13
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec)*pClass1
    p0 = sum(vec2Classify*p0Vec)*(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

"""
函数名:testingNB()
函数说明:测试朴素贝叶斯分类器

Parameters:
    无
Returns:
    无
Modify:
    2017-11-13
"""
def testingNB():
    #获取实验样本数据
    listOPosts, listClasses = dataSet.loadDataSet()
    #获取词汇表
    myVocabList = createVocabList(listOPosts)
    #构建训练矩阵
    trainMat = []
    for document in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, document))
    #获取非侮辱类词条的条件概率、侮辱类词条的条件概率、文档属于侮辱类文档的概率
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    #测试文档1
    testEntry = ['love', 'my', 'dalmation']
    #将测试文档进行向量化
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    # 测试文档2
    # testEntry = ['stupid', 'garbages']
    # thisDoc = setOfWords2Vec(myVocabList, testEntry)
    #获取分类结果
    label = classifyNB(thisDoc, p0V, p1V, pAb)
    #设置结果标签
    result = ['侮辱类文档','非侮辱类文档']
    #打印输出结果
    print testEntry, 'classify as:', result[label]

testingNB()