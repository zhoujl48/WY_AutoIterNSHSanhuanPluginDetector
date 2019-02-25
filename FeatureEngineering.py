#!/usr/bin/python
# -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The NSH Anti-Plugin Project
################################################################################
"""
NSH三环挂自动迭代项目 -- 离线训练模块，特征工程类
提供 Sequence Length / EvFreq / EvFreqGrade / EvTime / EvTimeGrade / EvSeq / EvTimeSeq 特征

Usage: 供调用
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019/02/13
"""

import os
import re
import numpy as np
from sklearn import model_selection
import random
from utils import log, get_time, ProgressBar
import json
from abc import ABCMeta, abstractmethod
from datetime import datetime
from random import sample
import concurrent.futures as futures
import threading


class DataLoader(object):
    __metaclass__ = ABCMeta

    def __init__(self, source_path, logid_path, label_tags, test_size, sampling_type, max_num, start=0, end=-1):
        assert sampling_type in ('up', 'down')
        assert 0.0 <= test_size <= 1.0
        assert os.path.exists(source_path)
        assert os.path.exists(logid_path)

        self._log_path = logid_path
        self.log_id_dict = self._get_logid_dict()

        self._source_path = source_path
        self._test_size = test_size
        self._sampling_type = sampling_type
        self._label_tags = label_tags
        self._max_num = max_num
        self._start = start
        self._end = end

        self._feature_data = list()
        self._label_data = list()
        self._feature_train = list()
        self._feature_test = list()
        self._label_train = list()
        self._label_test = list()

        self._data_ids = list()

    def _get_logid_dict(self):
        logid_dict = dict()
        with open(self._log_path) as f:
            for i, line in enumerate(f):
                logid_dict[line.strip()] = i+2
        return logid_dict

    @abstractmethod
    def _sequence2feature(self, log_sequence):
        raise NotImplementedError

    def _load(self):
        log('[{time}] loading from {path}'.format(time=get_time(), path=self._source_path))
        for i, label_tag in enumerate(self._label_tags):
            path = os.path.join(self._source_path, label_tag)
            files = sample(os.listdir(path)[self._start:self._end], self._max_num) if self._max_num > 0 else os.listdir(path)[self._start:self._end]
            print('start: {}, end: {}'.format(self._start, self._end))
            print(len(files))
            pbar = ProgressBar(len(files))
            for j, filename in enumerate(files):
                filepath = os.path.join(path, filename)
                try:
                    with open(filepath, 'r') as f:
                        log_sequence = json.load(f)
                        feature = self._sequence2feature(log_sequence)
                        self._data_ids.append(filepath.split('/')[-1].split('.')[0].split('_')[0])
                        self._feature_data.append(feature)
                        self._label_data.append(i)
                except:
                    log('[{time}] Failed to load file {filepath}'
                        .format(time=get_time(), filepath=filepath))
                    print('[{time}] Failed to load file {filepath}'
                        .format(time=get_time(), filepath=filepath))
                pbar.updateBar(j)



    def _sampling(self):
        log('[{time}] {type} sampling'.format(time=get_time(), type=self._sampling_type))

        self._n_label = len(set(self._label_data))
        labels_index = [[] for _ in range(self._n_label)]
        for i, label in enumerate(self._label_data):
            labels_index[label].append(i)

        if self._sampling_type == 'up':
            sampling_idx = list(range(len(self._label_data)))
            max_labels_index_len = max(map(len, labels_index))
            for i, label_index in enumerate(labels_index):
                sampling_idx.extend([label_index[random.randint(0, len(label_index)-1)]
                                     for _ in range(max_labels_index_len-len(label_index))])
        else:
            sampling_idx = list()
            min_labels_index_len = min(map(len, labels_index))
            for i, label_index in enumerate(labels_index):
                sampling_idx.extend([label_index[idx] for idx in
                                     random.sample(range(len(label_index)), min_labels_index_len)])
        self._feature_data = [self._feature_data[idx] for idx in sampling_idx]
        self._label_data = [self._label_data[idx] for idx in sampling_idx]

    def _shuffle(self):
        log('[{time}] shuffling...'.format(time=get_time()))
        samp = random.sample(range(len(self._label_data)), len(self._label_data))
        self._feature_data = [self._feature_data[idx] for idx in samp]
        self._label_data = [self._label_data[idx] for idx in samp]

    def _split(self):
        log('[{time}] splitting...'.format(time=get_time()))
        self._feature_train, self._feature_test, self._label_train, self._label_test = \
            model_selection.train_test_split(self._feature_data, self._label_data, test_size=self._test_size)
        del self._feature_data, self._label_data

    # 读取数据运行接口，供外部调用
    def run(self):
        self._load()
        self._sampling()
        self._shuffle()
        self._split()

    # 主要用于读取单类数据，无sampling/shuffle，主要供OCSVM调用
    def run_load(self):
        self._load()

    @property
    def total_data(self):
        return self._feature_data, self._label_data

    @property
    def train_data(self):
        return self._feature_train, self._label_train

    @property
    def test_data(self):
        return self._feature_test, self._label_test

    @property
    def ids(self):
        return self._data_ids


class SqeuenceLengthLoader(DataLoader):
    def __init__(self, source_path, grade, logid_path, label_tags=list(['normal', 'waigua']), test_size=0.2, sampling_type='up'):
        DataLoader.__init__(self, source_path=source_path, logid_path=logid_path, grade=grade, label_tags=label_tags, test_size=test_size,
                            sampling_type=sampling_type)

    def _sequence2feature(self, player_sequence):
        return len(player_sequence)


class EvfreqLoader(DataLoader):
    def __init__(self, source_path, logid_path, label_tags=list(['normal', 'waigua']), test_size=0.2, sampling_type='up'):
        DataLoader.__init__(self, source_path=source_path, logid_path=logid_path, label_tags=label_tags, test_size=test_size,
                            sampling_type=sampling_type)

    def _sequence2feature(self, player_sequence):
        feature = [0] * (len(self.log_id_dict) + 1)
        for item in player_sequence:
            feature[int(self.log_id_dict[item.split('#')[3]]) - 1 if item.split('#')[3] in self.log_id_dict else 0] += 1
        return feature


class EvfreqLoader_detail(DataLoader):
    def __init__(self, source_path, logid_path, label_tags=list(['normal', 'waigua']), test_size=0.2, sampling_type='up'):
        DataLoader.__init__(self, source_path=source_path, logid_path=logid_path, label_tags=label_tags, test_size=test_size,
                            sampling_type=sampling_type)

    def _sequence2feature(self, player_sequence):
        feature = [0] * (len(self.log_id_dict) + 1)
        for item in player_sequence:
            feature[int(self.log_id_dict['#'.join(item.split('#')[3:])]) - 1 if '#'.join(item.split('#')[3:]) in self.log_id_dict else 0] += 1
        return feature

class EvfreqLoader_detail_new(DataLoader):
    def __init__(self, source_path, logid_path, label_tags=list(['normal', 'waigua']), test_size=0.2, sampling_type='up'):
        DataLoader.__init__(self, source_path=source_path, logid_path=logid_path, label_tags=label_tags, test_size=test_size,
                            sampling_type=sampling_type)

    def _sequence2feature(self, player_sequence):
        feature = [0] * (len(self.log_id_dict) + 1)
        for item in player_sequence:
            feature[int(self.log_id_dict['#'.join(item.split('#')[2:])]) - 1 if '#'.join(item.split('#')[2:]) in self.log_id_dict else 0] += 1
        return feature


class EvfreqgLoader(DataLoader):
    def __init__(self, source_path, grade, logid_path, label_tags=list(['normal', 'waigua']), test_size=0.2, sampling_type='up'):
        DataLoader.__init__(self, source_path=source_path, logid_path=logid_path, grade=grade, label_tags=label_tags, test_size=test_size,
                            sampling_type=sampling_type)

    def _sequence2feature(self, player_sequence):
        grades = [int(item.split('#')[2]) for item in player_sequence]
        # 找到第一个非0等级
        first_grade = 0
        for g in grades:
            if g > first_grade:
                first_grade = g
                break
        # 按级别将索引分批
        cur_seq_index = 0
        grade_by_index = list()
        sequence_len = len(grades)
        for g in range(1, int(self._grade) + 1):
            this_grade_dist = list()
            while cur_seq_index < sequence_len:
                if grades[cur_seq_index] == g or grades[cur_seq_index] == 0:
                    this_grade_dist.append(cur_seq_index)
                    cur_seq_index += 1
                else:
                    break
            grade_by_index.append(this_grade_dist)
        # 分级别计算频数
        feature = list()
        for g in grade_by_index:
            this_grade_feature = [0] * (len(self.log_id_dict) + 1)
            for idx in g:
                this_grade_feature[
                    int(self.log_id_dict[player_sequence[idx].split('#')[3]]) - 1 if
                    player_sequence[idx].split('#')[3] in self.log_id_dict else 0] += 1
            feature.extend(this_grade_feature)
        return feature


class EvseqLoader(DataLoader):
    def __init__(self, source_path, logid_path, label_tags=list(['normal', 'waigua']), test_size=0.2, sampling_type='up'):
        DataLoader.__init__(self, source_path=source_path, logid_path=logid_path, label_tags=label_tags, test_size=test_size,
                            sampling_type=sampling_type)

    def _sequence2feature(self, player_sequence):
        player_feature = [str(self.log_id_dict[item.split('#')[3]]) if item.split('#')[3] in self.log_id_dict else str(1)
                   for item in player_sequence]
        return player_feature

class EvseqLoader_detail(DataLoader):
    def __init__(self, source_path, logid_path, label_tags=list(['normal_old_sampled', 'waigua_old_sampled']), test_size=0.2, sampling_type='up'):
        DataLoader.__init__(self, source_path=source_path, logid_path=logid_path, label_tags=label_tags, test_size=test_size,
                            sampling_type=sampling_type)

    def _sequence2feature(self, player_sequence):
        player_feature = [str(self.log_id_dict['#'.join(item.split('#')[3:])]) if '#'.join(item.split('#')[3:]) in self.log_id_dict else str(1)
                    for item in player_sequence]
        return player_feature

class EvseqLoader_hbase(DataLoader):
    def __init__(self, source_path, logid_path, label_tags=list(['neg', 'pos']), test_size=0.2, sampling_type='up', max_num=0):
        DataLoader.__init__(self, source_path=source_path, logid_path=logid_path, label_tags=label_tags, test_size=test_size,
                            sampling_type=sampling_type, max_num=max_num)

    def _sequence2feature(self, player_sequence):
        player_feature = [str(self.log_id_dict['#'.join(item.split('#')[2:])]) if '#'.join(item.split('#')[2:]) in self.log_id_dict else str(1)
                    for item in player_sequence]
        return player_feature

class EvseqLoader_hbase_pred(DataLoader):
    def __init__(self, source_path, logid_path, label_tags=list(['']), test_size=0.2, sampling_type='up', max_num=0, start=0, end=-1):
        DataLoader.__init__(self, source_path=source_path, logid_path=logid_path, label_tags=label_tags, test_size=test_size,
                            sampling_type=sampling_type, max_num=max_num, start=start, end=end)

    def _sequence2feature(self, player_sequence):
        player_feature = [str(self.log_id_dict['#'.join(item.split('#')[2:])]) if '#'.join(item.split('#')[2:]) in self.log_id_dict else str(1)
                    for item in player_sequence]
        return player_feature


class EvtimeLoader(DataLoader):
    def __init__(self, source_path, grade, logid_path, label_tags=list(['normal', 'waigua']), test_size=0.2, sampling_type='up'):
        DataLoader.__init__(self, source_path=source_path, logid_path=logid_path, grade=grade, label_tags=label_tags, test_size=test_size,
                            sampling_type=sampling_type)

    def _sequence2feature(self, player_sequence):
        format = '%Y-%m-%d %H:%M:%S'
        n_logid_index = len(self.log_id_dict) + 1
        player_feature, player_feature_count = [0] * n_logid_index, [0] * n_logid_index
        player_timestamp = [item.split('#')[1] for item in player_sequence]
        player_logid_index = [self.log_id_dict[item.split('#')[3]] for item in player_sequence if item.split('#')[3] in self.log_id_dict]
        for i in xrange(1, len(player_logid_index)):
            index = player_logid_index[i] - 1
            player_feature_count[index] += 1
            player_feature[index] += (datetime.strptime(player_timestamp[i], format) -
                                      datetime.strptime(player_timestamp[i - 1], format)).total_seconds()
        return map(lambda x, y: x * 1. / y if y else 0, player_feature, player_feature_count)


class EvtimegLoader(DataLoader):
    def __init__(self, source_path, grade, logid_path, label_tags=list(['normal', 'waigua']), test_size=0.2, sampling_type='up'):
        DataLoader.__init__(self, source_path=source_path, logid_path=logid_path, grade=grade, label_tags=label_tags, test_size=test_size,
                            sampling_type=sampling_type)

    def _sequence2feature(self, player_sequence):
        format = '%Y-%m-%d %H:%M:%S'
        n_logid_index = len(self.log_id_dict) + 1
        player_feature = [0] * (n_logid_index * int(self._grade))
        player_feature_count = [0] * (n_logid_index * int(self._grade))
        player_timestamp = [item.split('#')[1] for item in player_sequence]
        player_logid_index = [self.log_id_dict[item.split('#')[3]] for item in player_sequence if item.split('#')[3] in self.log_id_dict]
        player_grade = [int(item.split('#')[2]) for item in player_sequence]

        # 找到第一个不为0的等级
        cur_grade = 0
        for grade in player_grade:
            if grade != 0:
                cur_grade = grade
                break
        for i in xrange(1, len(player_logid_index)):
            index = player_logid_index[i] - 1
            cur_grade = max(cur_grade, player_grade[i])
            player_feature_count[(cur_grade - 1)*len(self.log_id_dict)+index] += 1
            player_feature[(cur_grade - 1)*len(self.log_id_dict)+index] += (datetime.strptime(player_timestamp[i], format) -
                                                     datetime.strptime(player_timestamp[i - 1], format)).total_seconds()
        return map(lambda x, y: x * 1. / y if y else 0, player_feature, player_feature_count)


class EvtseqLoader(DataLoader):
    def __init__(self, source_path, logid_path, label_tags=list(['normal', 'waigua']), test_size=0.2, sampling_type='up'):
        DataLoader.__init__(self, source_path=source_path, logid_path=logid_path, label_tags=label_tags, test_size=test_size,
                            sampling_type=sampling_type)

    def _sequence2feature(self, player_sequence):
        player_time = list()
        for index in range(len(player_sequence)):
            if index == 0:
                player_time.append(str(0))
            else:
                player_time.append(str((datetime.strptime(player_sequence[index].split('#')[1], '%Y-%m-%d %H:%M:%S')
                                        - datetime.strptime(player_sequence[index - 1].split('#')[1],
                                                            '%Y-%m-%d %H:%M:%S')).seconds))
        player_feature = [
            str(self.log_id_dict[item.split('#')[3]]) if item.split('#')[3] in self.log_id_dict else str(1) for item in
            player_sequence]
        return player_feature, player_time


if __name__ == '__main__':
    pass
