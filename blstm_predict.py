#!/usr/bin/python
# -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The NSH Anti-Plugin Project
################################################################################
"""
NSH三环挂自动迭代项目 -- BLSTM模型预测脚本

Usage: python $ds_pred --ds_start $ds_start --ds_num 28
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019/02/13
"""

import argparse
import os
import gc
import logging
import log
from datetime import datetime, timedelta
from FeatureEngineering import EvseqLoader_hbase_pred
from BLSTMModel import BLSTMModel
from config import SAVE_DIR_BASE, PROJECT_DIR

if __name__ == '__main__':

    parser = argparse.ArgumentParser('BLSTM Model Train, feature generation and model train. \n'
                                     'Usage: python BLSTM path grade ..')
    parser.add_argument('ds_pred', help='date of predicting')
    parser.add_argument('--ds_start', type=str)
    parser.add_argument('--ds_num', type=int)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()
    ds_pred = args.ds_pred
    logid_path = os.path.join(PROJECT_DIR, 'logid', 'logid_selected')
    print(logid_path)
    ds_start_num = '{}_{}'.format(args.ds_start, args.ds_num)
    start = args.start
    end = args.end

    # logging
    log.init_log(os.path.join(PROJECT_DIR, 'logs', 'blstm_predict'))

    # 加载数据
    data_path = os.path.join(SAVE_DIR_BASE, datetime.strptime(ds_pred, '%Y%m%d').strftime('%Y-%m-%d'))
    data = EvseqLoader_hbase_pred(data_path, logid_path=logid_path, sampling_type='up', test_size=0.0, max_num=0, start=start, end=end)
    data.run_load()

    # 加载模型，预测
    logging.info(os.path.join(SAVE_DIR_BASE, 'model', ds_start_num))

    # TODO: sorted(xx)[0] means loading worst performance model?
    model_name = sorted(os.listdir(os.path.join(SAVE_DIR_BASE, 'model', ds_start_num)))[0]
    model_path = os.path.join(SAVE_DIR_BASE, 'model', ds_start_num, model_name)
    logging.info('Loading model: {}'.format(model_path))
    model = BLSTMModel(train_data=data.total_data, test_data=data.total_data, feature_type='seq', ids=data.ids)
    results = model.run_predict(model_path=model_path, ds_pred=ds_pred)
    logging.info('Done predicting ids on date: {}'.format(ds_pred))
    logging.info(results[:10])

    del data, model
    gc.collect()
