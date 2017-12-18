# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: getData.py
@Date: 2017/12/14 16:19
"""
"""
函数名：loadDataSet()
函数说明：加载数据

Parameters:
    无
Return:
    dataSet - 数据列表
    label - 标签列表
Modify:
    2017-12-14

"""
def loadDataSet(filename):
    #打开文件
    fr = open("testSet.txt")
    #数据矩阵
    dataSet = []
    #标签数组
    labels = []
    for line in fr.readlines():
        #获取一行数据，按照空格分开
        lineArr = line.strip().split()
        #添加数据
        dataSet.append([float(lineArr[0]), float(lineArr[1])])
        #添加标签
        labels.append(int(lineArr[2]))
    #关闭文件
    fr.close()
    #返回数据矩阵和标签数组
    return dataSet, labels