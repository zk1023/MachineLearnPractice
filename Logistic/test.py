# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: test.py
@Date: 2017/11/25 11:01
"""
"""
函数名：GradientAscentTest()
函数说明：梯度上升算法测试函数

Parameter:
    无
Return:
    无
Modify:
    2017-11-25
"""
def GradientAscentTest():
    #测试函数为 f = -x^2+4x
    #定义求导函数
    def f_prime(x_old):
        return -2 * x_old + 4
    #梯度上升算法初始值，即从(0,0)开始
    x_new = 0.0
    #初始值，小于x_new
    x_old = -1
    #步长,控制更新幅度
    alpha = 0.01
    #精确度
    precious = 0.00000001
    while abs(x_new - x_old) > precious :
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    print x_new
# GradientAscentTest()


