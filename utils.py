#!/usr/bin/python
# -*- coding:utf-8 -*-
################################################################################
"""
NSH主线挂自动迭代项目 -- 离线训练模块，utils: 日志组件 / 时间组件 / 进度条组件

Usage: 监督模型基类
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019/02/13
"""

import sys
import time

_logging_fname, _logging_file = None, None
_logging = False


def set_log():
    global _logging_fname, _logging_file, _logging
    start_time = get_time()
    _logging = True
    _logging_fname = 'log/Preprocessing_{time}.log'.format(time=start_time)
    _logging_file = open(_logging_fname, 'w')


def log(msg):
    print(msg)
    if _logging:
        _logging_file.write(msg + '\n')
        _logging_file.flush()


def get_time():
    return time.asctime(time.localtime(time.time()))


class ProgressBar(object):
    def __init__(self, total, bar_length=40):
        self._total = total
        self._bar_length = bar_length
        self._status = ''

    def updateBar(self, cur, info=''):
        progress = cur*1./self._total
        if progress >= 1.:
            progress, self._status = 1, '\r\n'
        block = int(round(self._bar_length * progress))
        text = "\r[{}] {:.0f}% {}  {}".format(
                "#" * block + "-" * (self._bar_length - block), round(progress * 100, 1),
                self._status, info)
        sys.stdout.write(text)
        sys.stdout.flush()


