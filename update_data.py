#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The NSH Anti-Plugin Project
################################################################################
"""
NSH三环挂自动迭代项目 -- 训练样本更新模块
训练样本包含四周数据，每周将最老一周的数据更新为最新一周的数据，
其中重复的三周数据将被复制到新目录下，避免重复拉取

Usage: python update_data.py --old_dir 20181226_28 --ds_start 20190102 --ds_num 28
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019/02/13
"""

import os
import argparse
import json
import shutil
from random import sample, seed
from datetime import datetime, timedelta
from config import SAVE_DIR_BASE
from config import QUERY_DICT


def parse_args():
    parser = argparse.ArgumentParser("Run trigger_sanhuan"
                                     "Usage: python get_ids.py pos  --ds_start 20181215 --ds_num 7")
    parser.add_argument('label', type=str)
    parser.add_argument('--old_dir', type=str, default='20181212_28')
    parser.add_argument('--ds_start', type=str)
    parser.add_argument('--ds_num', type=int)

    return parser.parse_args()



if __name__ == '__main__':

    # 输入参数
    args = parse_args()
    label = args.label
    ds_start = datetime.strptime(args.ds_start, '%Y%m%d').strftime('%Y-%m-%d')
    ds_end = (datetime.strptime(ds_start, '%Y-%m-%d') + timedelta(days=args.ds_num)).strftime('%Y-%m-%d')
    new_dir = '{}_{}'.format(args.ds_start, args.ds_num)
    new_source = os.path.join(SAVE_DIR_BASE, 'train_data', new_dir, args.label)
    old_source = os.path.join(SAVE_DIR_BASE, 'train_data', args.old_dir, args.label)

    # 创建新目录
    if not os.path.exists(new_source):
        os.makedirs(new_source)

    # 复制已有数据至新目录
    datetime_start = datetime.strptime(ds_start, '%Y-%m-%d')
    datetime_end = datetime.strptime(ds_end, '%Y-%m-%d')
    filenames_can_use = list()
    for filename in os.listdir(old_source):
        datetime_seq = datetime.strptime(filename.split('_')[-1], '%Y-%m-%d')
        if datetime_seq >= datetime_start and datetime_seq <= datetime_end:
            filenames_can_use.append(filename)
    seed(0)
    num_sample = min(len(filenames_can_use), 100000)
    for filename in sample(filenames_can_use, num_sample):
        old_file_path = os.path.join(old_source, filename)
        new_file_path = os.path.join(new_source, filename)
        if not os.path.exists(new_file_path):
            shutil.copy(old_file_path, new_file_path)
    print('{} files copied to new source.'.format(len(os.listdir(new_source))))






