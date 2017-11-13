# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: test.py
@Date: 2017/9/13 21:43
"""
import xlwt
from numpy import *
def f():
    fr = open("C:/Users/Administrator/Desktop/test.txt")
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    workbook = xlwt.Workbook
    workbook = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = workbook.add_sheet('students', cell_overwrite_ok=True)
    index = 1
    sheet.write(0, 0, "学号")
    sheet.write(0, 1, "姓名")
    sheet.write(0, 2, "性别")
    sheet.write(0, 3, "专业")
    sheet.write(0, 4, "班级")
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split(' ')
        for i in range(5):
            sheet.write(index, i, listFromLine[i])
            # sheet.write(index, i, 'a')
        index += 1
    workbook.save('C:/Users/Administrator/Desktop/1703students.xls')
f()