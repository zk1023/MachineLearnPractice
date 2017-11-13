# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: dataSet.py
@Date: 2017/11/12 10:59
"""
#导入numpy
from numpy import *

#要从文本中获取特征，需要先拆分文本，这里特征是指来自文本的词条，每个词
#条是字符的任意组合。词条可以理解为单词，当然也可以是非单词词条，比如URL
#IP地址或者其他任意字符串

"""
函数名:loadDataSet()
函数说明:创建实验样本

Parameters:
    无
Returns:
    postingList - 实验样本切分的词条
    classVec - 类别标签向量
Modify:
    2017-11-12
"""
def loadDataSet():
    # 词条(单词)切分后的文档集合，列表每一行代表一个文档
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    #类标签 1 代表有侮辱性文字,0代表正常言论
    classVec = [0,1,0,1,0,1]
    #返回词条切分后的文档集合和类别标签
    return postingList,classVec

if __name__ == '__main__':
    postingList,classVec = loadDataSet()
    i = 0
    for each in postingList:
        print each, classVec[i]
        i = i + 1
