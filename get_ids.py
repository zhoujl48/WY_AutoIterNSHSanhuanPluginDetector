#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The NSH Anti-Plugin Project
################################################################################
"""
NSH主线挂自动迭代项目 -- Hive拉取ids
从Hive获取指定等级、开始和结束日期的正负样本ID

Usage: python get_ids.py pos --ds_start 20181215 --ds_num 7
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019/02/13
"""

import os
import argparse
import json
import logging
import log
from datetime import datetime, timedelta
from config import SAVE_DIR_BASE, PROJECT_DIR
from config import QUERY_DICT
from HiveUtils import get_ids, connect_hive, hive_ip, hive_port


def parse_args():
    parser = argparse.ArgumentParser("Run trigger_sanhuan"
                                     "Usage: python get_ids.py pos  --ds_start 20181215 --ds_num 7")
    parser.add_argument('label', help='\'pos\' or \'total\'')
    parser.add_argument('--ds_start', type=str)
    parser.add_argument('--ds_num', type=int)

    return parser.parse_args()


def fetch_ids(sql, filepath):
    """HIVE拉取id和ds

    Args:
        sql: Query查询语句
        filepath: triggrt file 保存路径
    """

    results = connect_hive(hive_ip=hive_ip, hive_port=hive_port, sql=sql)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


if __name__ == '__main__':

    # 输入参数：等级，开始和结束日期
    args = parse_args()
    label = args.label
    ds_start = '{}-{}-{}'.format(args.ds_start[:4], args.ds_start[4:6], args.ds_start[6:])
    ds_end = (datetime.strptime(ds_start, '%Y-%m-%d') + timedelta(days=args.ds_num)).strftime('%Y-%m-%d')
    ds_end_7 = (datetime.strptime(ds_start, '%Y-%m-%d') + timedelta(days=args.ds_num + 7)).strftime('%Y-%m-%d')

    # logging
    log.init_log(os.path.join(PROJECT_DIR, 'logs', 'get_ids'))

    # 创建trigger目录
    trigger_dir = os.path.join(SAVE_DIR_BASE, 'trigger')
    if not os.path.exists(trigger_dir):
        os.mkdir(trigger_dir)

    # query
    if label == 'pos':
        filepath_ban = os.path.join(SAVE_DIR_BASE, 'trigger', '{}_{}_ban'.format(args.ds_start, args.ds_num))
        filepath_punish = os.path.join(SAVE_DIR_BASE, 'trigger', '{}_{}_punish'.format(args.ds_start, args.ds_num))
        if not os.path.exists(filepath_ban):
            sql_ban = QUERY_DICT['ban'].format(ds_start=ds_start, ds_end=ds_end, ds_ban_start=ds_start, ds_ban_end=ds_end_7)
            logging.info('Start pulling {} ids to file: {}'.format(label, filepath_ban))
            logging.info(sql_ban)
            fetch_ids(sql=sql_ban, filepath=filepath_ban)
        if not os.path.exists(filepath_punish):
            sql_punish = QUERY_DICT['punish'].format(ds_start=ds_start, ds_end=ds_end, ds_punish_start=ds_start, ds_punish_end=ds_end_7)
            logging.info('Start pulling {} ids to file: {}'.format(label, filepath_punish))
            logging.info(sql_punish)
            fetch_ids(sql=sql_punish, filepath=filepath_punish)
    elif label == 'total' and args.ds_num == 1:
        filepath_total = os.path.join(SAVE_DIR_BASE, 'trigger', '{}_{}_total'.format(args.ds_start, args.ds_num))
        if not os.path.exists(filepath_total):
            sql_total = QUERY_DICT['total'].format(ds_start=ds_start)
            logging.info('Start pulling {} ids to file: {}'.format(label, filepath_total))
            logging.info(sql_total)
            fetch_ids(sql=sql_total, filepath=filepath_total)
    elif label == 'neg':
        filepath_neg = os.path.join(SAVE_DIR_BASE, 'trigger', '{}_{}_neg'.format(args.ds_start, args.ds_num))
        if not os.path.exists(filepath_neg):
            sql_neg = QUERY_DICT['neg'].format(ds_start=ds_start, ds_end=ds_end, ds_ban_start=ds_start, ds_ban_end=ds_end_7)
            logging.info('Start pulling {} ids to file: {}'.format(label, filepath_neg))
            fetch_ids(sql=sql_neg, filepath=filepath_neg)
    else:
        results = list()
        logging.error('params error!')

