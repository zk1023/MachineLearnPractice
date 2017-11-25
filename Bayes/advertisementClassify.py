# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: advertisementClassify.py
@Date: 2017/11/18 10:47
"""
import feedparser
import bayes_test
import emailClassify
from numpy import *

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
ny['entries']
print len(ny['entries'])
# print ny['entries']

"""
函数名：calcMostFreq(vocabList, fullText)
函数说明：选择高频词

Parameter:
    vocabList - 词汇表
    fullText - 所有单词
Return:
    sortedFreq[:30] - 返回出现次数最高的前三十个单词
Modify:
     2017-11-18
"""
def calcMostFreq(vocabList, fullText):
    #导入操作符
    import operator
    #创建新词典
    freqDict = {}
    #遍历词条列表中的每一个词
    for token in vocabList:
        #记录每个单词出现的次数
        freqDict[token] = fullText.count(token)
    #按照词条出现次数进行从大到小排序
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=1)
    #返回出现次数最多的前三十个单词
    return sortedFreq[:30]

"""
函数名：localWords(feed1, feed0)
函数说明：从RSS源获取数据, 并分类测试

Parameter:
    feed0 - RSS数据源
    feed1 - RSS数据源
Return:
    vocabList - 词汇表
    p0V - newyork城市广告中词条的条件概率
    p1V - sfbay城市广告中词条的条件概率
Modify:
    2017-11-18
"""
def localWords(feed1, feed0):
    #保存各个文档
    docList = []
    #记录所有单词
    fullText = []
    #记录每个文档对应的类别
    classList = []
    #获取条目数较少的RSS源的条目数
    minLen = min(len(feed0['entries']), len(feed1['entries']))
    # print len(feed1['entries']), feed1['entries']
    #遍历每个条目
    for i in range(minLen):
        #解析获取相应的数据
        wordList  = emailClassify.textParse(feed1['entries'][i]['summary'])
        print wordList
        #构建词汇列表集合
        docList.append(wordList)
        #记录所有单词
        fullText.extend(wordList)
        #添加类标签
        classList.append(1)

        # 解析获取相应的数据
        wordList = emailClassify.textParse(feed0['entries'][i]['summary'])
        print wordList
        # 构建词汇列表集合
        docList.append(wordList)
        # 记录所有单词
        fullText.extend(wordList)
        # 添加类标签
        classList.append(0)
    #创建词汇表
    vocabList = bayes_test.createVocabList(docList)
    #找到词汇表中出现频率最高的30个单词
    top30Words = calcMostFreq(vocabList, fullText)
    # print top30Words
    #据说去除高频词后，错误率会下降，试一试
    #从词汇表中去除这些单词
    for word in top30Words:
        if word[0] in vocabList:
            vocabList.remove(word[0])
            # 采用留存交叉验证进行分类
    # 随机选出十个邮件作为测试集，剩下的作为训练集
    # 构建存储训练集索引值的列表
    trainSet = range(minLen*2)
    # 构建存储测试集索引值的列表
    testSet = []
    for i in range(20):
        # 产生从0到训练集长度的随机数
        randIndex = int(random.uniform(0, len(trainSet)))
        # 选取测试集
        testSet.append(trainSet[randIndex])
        # 从训练集中去除测试集
        del (trainSet[randIndex])
    # 构建贝叶斯分类器
    # 建立训练矩阵
    trainMat = []
    # 用于存储类别标签
    trainClass = []
    # 遍历训练集
    for i in trainSet:
        # 构建词向量训练矩阵
        trainMat.append(bayes_test.bagOfWords2VecMN(vocabList, docList[i]))
        # 添加邮件向量对应的类别
        trainClass.append(classList[i])
    # 获取垃圾邮件条件概率、非垃圾邮件的条件概率和邮件属于垃圾邮件的概率
    p0V, p1V, pSpam = bayes_test.trainNB0(trainMat, trainClass)
    # 记录邮件分类错误率
    errorCount = 0
    # 开始测试
    for i in testSet:
        # 获取分类器输出标签
        label = bayes_test.classifyNB(bayes_test.bagOfWords2VecMN(vocabList, docList[i]), p0V, p1V, pSpam)
        # 与真实类别进行比较，并记录错误数量
        if label != classList[i]:
            errorCount += 1
            print 'classify error: ', docList[i]
    # 输出错误率
    result = float(errorCount) / len(testSet)
    print result
    return vocabList, p0V, p1V

"""
函数名：
函数说明：

Parameter:

Return:

Modify:
    2017-11-18
"""
def getTopWords(ny, sf):
    #利用RSS源分类器获取词汇表，及分类中各个词条的条件概率
    vocabList, p0V, p1V = localWords(ny, sf)
    #构造两个元组列表
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        #往对应元组中添加概率大于某个阈值的词及其概率值组成的二元列表
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    #按照二元组中的概率值进行从大到小排序
    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF"
    for item in sortedSF:
        print item[0]
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY"
    # 按照二元组中的概率值进行从大到小排序
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    for item in sortedNY:
        print item[0]

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
getTopWords(ny, sf)
