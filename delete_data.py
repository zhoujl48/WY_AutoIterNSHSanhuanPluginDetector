#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The NSH Anti-Plugin Project
################################################################################
"""
NSH三环挂自动迭代项目 -- 数据删除模块
仅保留两周内的每天全量数据，每天删除多余的旧数据

Usage: python delete_data.py --date 2019-01-24
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019/02/13
"""

import os
import shutil
import argparse
import logging
import dateutil.parser
import log
from datetime import datetime, timedelta
from config import SAVE_DIR_BASE, PROJECT_DIR


def parse_args():
    parser = argparse.ArgumentParser("Delete data"
                                     "Usage: python delete_data.py --date 2019-01-24")
    parser.add_argument('--date', type=str)
    return parser.parse_args()


if __name__ == '__main__':

    # 参数
    args = parse_args()
    date = args.date
    date = dateutil.parser.parse(date).strftime('%Y-%m-%d')     # 统一日期格式
    data_dir = os.path.join(SAVE_DIR_BASE, date)                # 数据目录


    # logging
    log.init_log(os.path.join(PROJECT_DIR, 'logs', 'delete_data'))

    # 删除指定日期的全天数据
    if os.path.exists(data_dir):
        try:
            logging.info('Start deleting data on date {}'.format(date))
            shutil.rmtree(data_dir)
            logging.info(('Successfully delete data on date {}'.format(date)))
        except Exception as e:
            logging.error(data_dir, e)
    else:
        logging.warning('No such dir: {}'.format(data_dir))
