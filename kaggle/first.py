# -*- coding: utf-8 -*-： 
"""  
@author: Zhangkai
@file: first.py
@Date: 2017/12/19 21:32
"""
import win32api
import win32con

win32api.keybd_event(17,0,0,0)  #ctrl键位码是17
win32api.keybd_event(16,0,0,0)  #shift键位码是86
win32api.keybd_event(16,0,win32con.KEYEVENTF_KEYUP,0) #释放按键
win32api.keybd_event(17,0,win32con.KEYEVENTF_KEYUP,0)

num = [87, 69, 78, 71, 69, 32, 72, 65, 79, 32]
for i in range(len(num)):
    win32api.keybd_event(num[i], 0, 0, 0)#按键
    win32api.keybd_event(num[i], 0, win32con.KEYEVENTF_KEYUP, 0)#释放按键

