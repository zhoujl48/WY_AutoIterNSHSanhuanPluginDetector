#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The NSH Anti-Plugin Project
################################################################################
"""
NSH三环挂自动迭代项目 -- HBase序列拉取模块

Usage: python dataloader.py --ds_start $ds_pred --ds_num 1
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019/02/13
"""

import os
import sys
import argparse
import logging
import logging.handlers
from queue import Queue
import _thread
import threading
import json
import requests
from time import sleep
from random import sample, seed
from datetime import datetime, timedelta
from config import SAVE_DIR_BASE, THREAD_NUM, HBASE_URL

lock = _thread.allocate_lock()


# 时间统计格式
TIME_FORMAT = '%Y-%m-%d'
# LOG配置
LOG_FILE = 'logs/dataloader_hbase.log'  # 日志文件
SCRIPT_FILE = 'dataloader_hbase'  # 脚本文件
LOG_LEVEL = logging.DEBUG  # 日志级别
LOG_FORMAT = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'  # 日志格式


def parse_args():
    parser = argparse.ArgumentParser('Pulling data sequneces'
                                     'Usage: python dataloader.py --end_grade 41 --ds_start 20181215 --ds_num 7')
    parser.add_argument('--mode', default='daily', help='\'daily\', \'pos\' or \'neg\'')
    parser.add_argument('--ds_start', type=str)
    parser.add_argument('--ds_num', type=int, default=1)

    return parser.parse_args()


def init_log():
    handler = logging.handlers.RotatingFileHandler(LOG_FILE)  # 实例化handler
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")  # 实例化formatter
    handler.setFormatter(formatter)  # 为handler添加formatter
    logger = logging.getLogger(SCRIPT_FILE)  # 获取logger
    logger.addHandler(handler)  # 为logger添加handler
    logger.setLevel(LOG_LEVEL)
    return logger


def get_ids(filepath, max_num):
    """从 trigger file 获取样本id和ds

    Args:
        path_ids: trigger file 的地址
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    num = min(len(results), max_num) if max_num > 0 else len(results)
    seed(0)
    results = sample(results, num)
    return results


class SequenceDataReader(threading.Thread):
    """序列数据读取类

    Attributes:
        logger: 日志
        queue: 多线程队列
        save_dir: 序列保存路径
    """
    def __init__(self, logger, queue, save_dir):
        threading.Thread.__init__(self)
        self.logger = logger
        self.queue = queue
        self.save_dir = save_dir

    def read_data(self, role_id, ds):
        """从hbase拉取数据

        Args:
            role_id: 样本用户ID
            ds: 样本用户日期

        Return:
            seq: 样本用户行为序列
        """
        ts_start = ds + ' 00:00:00'
        ts_end = ds + ' 23:59:59'
        url = HBASE_URL.format(st=ts_start, ed=ts_end, ids=role_id)
        response = requests.post(url, timeout=600)
        results = response.json()
        result = results[0]

        logids = result['role_seq']
        seq = [k['logid'] for k in logids]

        return seq

    def save_to_file(self, role_id, ds, seq):
        """保存行为序列

        Args:
            role_id: 样本用户ID
            seq: 样本用户行为序列
        """

        filename = os.path.join(self.save_dir, role_id + '_' + ds)
        with open(filename, 'w') as f:
            json.dump(seq, f, indent=4, sort_keys=True)

    def run(self):
        """多线程拉取运行接口

        遍历队列中的样本ID，拉取行为序列，并保存至相应目录
        """

        global lock

        # 循环读取queue中数据
        while True:
            if self.queue.qsize() % 1000 == 0:
                self.logger.info('{} id left'.format(self.queue.qsize()))

            lock.acquire()
            if self.queue.empty():
                lock.release()
                return
            role_id, ds = self.queue.get()
            lock.release()

            try:
                # 读取数据
                seq = self.read_data(role_id, ds)
                # 存入文件
                self.save_to_file(role_id, ds, seq)
                sleep(0.5)
            except Exception as e:
                self.logger.error('error with id = {}, error = {}'.format(role_id, e))
                # print('error with id = {}, error = {}'.format(role_id, e))
                # 若失败则重新放入队列
                lock.acquire()
                self.queue.put([role_id, ds])
                lock.release()


# 主函数
def main(argv):
    """主函数

    拉取指定用户ID对应的某段时间区间内行为序列并保存
    """

    args = parse_args()
    ds_start = args.ds_start
    ds_num = args.ds_num
    mode = args.mode



    # logger
    _logger = init_log()

    # 队列，用于多线程
    queue = Queue()

    # 创建数据目录，读取id
    if mode == 'daily':
        souce_dir = os.path.join(SAVE_DIR_BASE, datetime.strptime(ds_start, '%Y%m%d').strftime('%Y-%m-%d'))
        if not os.path.exists(souce_dir):
            os.mkdir(souce_dir)
        filepath_total = os.path.join(SAVE_DIR_BASE, 'trigger', '{}_{}_total'.format(ds_start, ds_num))
        results = get_ids(filepath_total, 0)
    elif mode in ['pos', 'neg']:
        ds_range = '{}_28'.format((datetime.strptime(ds_start, '%Y%m%d') - timedelta(days=21)).strftime('%Y%m%d'))
        souce_dir = os.path.join(SAVE_DIR_BASE, 'train_data', ds_range, mode)
        if not os.path.exists(souce_dir):
            os.makedirs(souce_dir)
        if mode == 'pos':
            filepath_ban = os.path.join(SAVE_DIR_BASE, 'trigger', '{}_{}_ban'.format(ds_start, ds_num))
            filepath_punish = os.path.join(SAVE_DIR_BASE, 'trigger', '{}_{}_punish'.format(ds_start, ds_num))
            # TODO: add specific trigger_samples provied by Cehua such as {}_spec
            filepath_cehua_xinyabiao = os.path.join(SAVE_DIR_BASE, 'trigger', '{}_{}_xinyabiao'.format(ds_start, ds_num))
            if os.path.exists(filepath_cehua_xinyabiao):
                results = get_ids(filepath_ban, 20000) + get_ids(filepath_punish, 20000) + get_ids(filepath_cehua_xinyabiao, 20000)
            else:
                results = get_ids(filepath_ban, 25000) + get_ids(filepath_punish, 25000)
        else:
            filepath_neg = os.path.join(SAVE_DIR_BASE, 'trigger', '{}_{}_neg'.format(ds_start, ds_num))
            results = get_ids(filepath_neg, 50000)
    else:
        raise Exception('Unvalid mode: {}'.format(mode))

    # 去除已经拉去的新seq
    existed_files = set(os.listdir(souce_dir))
    week_files = set(['{}_{}'.format(role_id, ds) for role_id, ds in results])
    new_files = [item.split('_') for item in list(week_files - existed_files)]

    # 需拉取序列id放入队列
    for role_id, ds in new_files:
        queue.put([str(role_id), ds])

    print('Start pulling data to dir: {}'.format(souce_dir))

    # 线程
    thread_list = []
    thread_num = THREAD_NUM
    for i in range(thread_num):
        _logger.info('init thread = {}'.format(i))
        thread = SequenceDataReader(_logger, queue, souce_dir)
        thread_list.append(thread)
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()

    print('Finish pulling total sequence on date: {}'.format(ds_start))


if __name__ == '__main__':
    main(sys.argv)
