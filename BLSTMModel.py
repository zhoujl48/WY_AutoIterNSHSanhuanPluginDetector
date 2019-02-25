#!/usr/bin/python
# -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The NSH Anti-Plugin Project
################################################################################
"""
NSH三环挂自动迭代项目 -- BLSTM模型训练模块

Usage: python BLSTMModel.py --ds_start $ds_start --ds_num 28 ...
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019/02/13
"""

import os
import argparse
import numpy as np
from datetime import datetime
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout, Flatten, concatenate, Reshape
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from SupervisedModel import SupervisedModel
from FeatureEngineering import EvseqLoader_hbase
from config import SAVE_DIR_BASE, PROJECT_DIR


class BLSTMModel(SupervisedModel):
    """BLSTM模型

    Attributes:
        _feature_train: 训练数据
        _label_train: 训练标签
        _feature_test: 测试数据
        _feature_label: 测试标签
        _feature_type: 特征提取方式
        _ids: 样本ID
        _max_len: 最大长度限制
        _embedding_size: embedding大小
        _embedding_dropout_size: embedding的dropout大小
        _dense_size: dense层大小
        _dense_dropout_size: dense层的dropout大小
        _lstm_size: lstm层大小
        _lstm_dropout_size: lstm层的dropout大小
        _apply_attention_before_lstm: 是否在lstm层之前添加attention机制
        _attention_dropout_size: attention层的dropout大小
        _max_logid: logid最大数量限制
        _model_file: 模型保存路径
    """
    def __init__(self, train_data, test_data, feature_type, max_len=2000, epoch=30, batch_size=128, dropout_size=0.2,
                 regular=0.002, embedding_size=32, dense_size=128, lstm_size=64, apply_attention_before_lstm=False,
                 single_attention_vector=False, max_logid=450, ids=None, save_path=''):
        SupervisedModel.__init__(self, epoch=epoch, batch_size=batch_size, regular=regular)

        # data
        assert len(train_data) == 2
        self._feature_train, self._label_train = train_data
        self._feature_test, self._label_test = test_data
        self._feature_type = feature_type
        self._ids = ids

        # network
        self._max_len = max_len
        self._embedding_size = embedding_size
        self._embedding_dropout_size = dropout_size
        self._dense_size = dense_size
        self._dense_dropout_size = dropout_size
        self._lstm_size = lstm_size
        self._lstm_dropout_size = dropout_size
        self._apply_attention_before_lstm = apply_attention_before_lstm
        self._attention_dropout_size = dropout_size
        self._single_attention_vector = single_attention_vector
        self._max_logid = max_logid
        self._model_file = os.path.join(save_path, 'blstm_{feature}_len_{max_len}_logid_{max_logid}_embedding_{embedding_size}_dense_{dense_size}'.format(
                                        max_len=self._max_len,
                                        feature=self._feature_type,
                                        max_logid=self._max_logid,
                                        embedding_size=self._embedding_size,
                                        dense_size=self._dense_size))

    def _padding_sequence(self):
        """padding样本序列序列
        """
        if self._feature_type == 'tseq':
            self._time_train = [self._feature_train[i][1] for i in range(len(self._feature_train))]
            self._feature_train = [self._feature_train[i][0] for i in range(len(self._feature_train))]
            self._time_test = [self._feature_test[i][1] for i in range(len(self._feature_test))]
            self._feature_test = [self._feature_test[i][0] for i in range(len(self._feature_test))]
            self._time_train = sequence.pad_sequences(self._time_train, maxlen=self._max_len)
            self._time_test = sequence.pad_sequences(self._time_test, maxlen=self._max_len)
            self._time_train = np.reshape(self._time_train, (len(self._time_train), self._max_len, 1))
            self._time_test = np.reshape(self._time_test, (len(self._time_test), self._max_len, 1))
            self._feature_train = sequence.pad_sequences(self._feature_train, maxlen=self._max_len)
            self._feature_test = sequence.pad_sequences(self._feature_test, maxlen=self._max_len)
        else:
            self._feature_train = sequence.pad_sequences(self._feature_train, maxlen=self._max_len)
            self._feature_test = sequence.pad_sequences(self._feature_test, maxlen=self._max_len)

    def model(self):
        """Model定义及训练
        """
        print('Building model...')
        logid_input = Input(shape=(self._max_len,), dtype='int32', name='logid_input')
        logid_embedding = Embedding(output_dim=self._embedding_size, input_dim=self._max_logid,
                                    input_length=self._max_len)(logid_input)
        logid_embedding_dropout = Dropout(self._embedding_dropout_size)(logid_embedding)
        if self._feature_type == 'tseq':
            time_input = Input(shape=(self._max_len, 1,), dtype='float32', name='time_input')
            input_concat = concatenate([logid_embedding_dropout, time_input], axis=2)
            input_concat_reshape = Reshape((self._max_len, self._embedding_size + 1))(input_concat)
            input_lstm = Bidirectional(LSTM(units=self._lstm_size, return_sequences=True))(input_concat_reshape)
        else:
            input_lstm = Bidirectional(LSTM(units=self._lstm_size, return_sequences=True))(logid_embedding_dropout)
        dropout_lstm = Dropout(self._lstm_dropout_size)(input_lstm)
        dropout_lstm = Flatten()(dropout_lstm)
        dropout_attention = Dropout(self._attention_dropout_size)(dropout_lstm)
        dense = Dense(self._dense_size, activation='relu', name='dense',
                      kernel_regularizer=regularizers.l2(self._regular))(dropout_attention)
        dropout_dense = Dropout(self._dense_dropout_size)(dense)
        output = Dense(1, activation='sigmoid', name='output')(dropout_dense)
        if self._feature_type == 'tseq':
            model_final = Model(inputs=[logid_input, time_input], outputs=[output])
        else:
            model_final = Model([logid_input], [output])
        model_final.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy', self.precision, self.recall, self.f1_score])
        print(model_final.summary())
        # checkpoint
        checkpoint = ModelCheckpoint(self._model_file + '.weights.{epoch:03d}-{val_f1_score:.4f}.hdf5',
                                     monitor='val_f1_score', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        print('Train...')
        if self._feature_type == 'tseq':
            model_final.fit([self._feature_train, self._time_train], self._label_train,
                            batch_size=self._batch_size,
                            epochs=self._epoch,
                            callbacks=callbacks_list,
                            validation_data=([self._feature_test, self._time_test], self._label_test))
        else:
            model_final.fit([self._feature_train], self._label_train,
                            batch_size=self._batch_size,
                            epochs=self._epoch,
                            callbacks=callbacks_list,
                            validation_data=([self._feature_test], self._label_test))

    def run(self):
        """离线训练调用接口
        """
        self._padding_sequence()
        self.model()


    def run_predict(self, model_path, ds_pred):
        """预测调用接口(离线评估用)

        Args:
            model_path: 模型保存路径
            ds_pred: 预测日期

        Return:
            results: 预测结果，[[role_id, suspect_score]...]
        """

        # 加载模型
        self._padding_sequence()
        model = load_model(model_path, compile=False)

        # 预测
        suspect_scores = model.predict(self._feature_train)
        print(len(self._ids), len(suspect_scores))

        # 保存文件，返回预测结果
        result_file = 'sanhuan_sl_blstm_{ds_pred}.csv'.format(ds_pred=datetime.strptime(ds_pred, '%Y%m%d').strftime('%Y_%m_%d'))
        pred_dir = os.path.join(SAVE_DIR_BASE, 'classification')
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        results = list()
        with open(os.path.join(pred_dir, result_file), 'w') as f:
            for i in range(len(self._ids)):
                role_id = str(self._ids[i])
                suspect_score = str(suspect_scores[i][0])
                f.write(role_id + ',' + suspect_score + '\n')
                results.append([role_id, suspect_score])
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BLSTM Model Train, feature generation and model train. \n'
                                     'Usage: python BLSTMModel path ..')
    parser.add_argument('--ds_start', help='data source')
    parser.add_argument('--ds_num', help='data source')
    parser.add_argument('--feature', help='set specified feature generated for training. available: \'seq\', \'tseq\'', default='seq')
    parser.add_argument('--max_len', help='set padding length if shorter or splitting length if longer', default=2000, type=int)
    parser.add_argument('--max_num', help='max num of data of each label', default=0, type=int)
    parser.add_argument('--max_logid', help='max len of logid', default=3800, type=int)
    parser.add_argument('--epoch', help='set the training epochs', default=30, type=int)
    parser.add_argument('--batch_size', help='set the training batch size', default=128, type=int)
    parser.add_argument('--dropout_size', help='set the dropout size for fc layer or lstm cells', default=0.2, type=float)
    parser.add_argument('--regular', help='set regularization', default=0.0, type=float)
    parser.add_argument('--embedding_size', help='set embedding size', default=64, type=int)
    parser.add_argument('--dense_size', help='set dense size', default=64, type=int)
    parser.add_argument('--lstm_size', help='set lstm cell size', default=64, type=int)
    parser.add_argument('--test_size', help='set test ratio when splitting data sets into train and test', default=0.2, type=float)
    parser.add_argument('--sampling_type', help='set sampling type, \'up\' or \'down\'', default='up')
    args = parser.parse_args()
    max_logid = args.max_logid
    ds_start = args.ds_start
    ds_num = args.ds_num

    # 数据路径
    logid_path = os.path.join(PROJECT_DIR, 'logid', 'logid_selected')

    # 模型保存路径
    PATH_MODEL_SAVE = os.path.join(SAVE_DIR_BASE, 'model', ds_start + '_' + ds_num)
    if not os.path.exists(PATH_MODEL_SAVE):
        os.makedirs(PATH_MODEL_SAVE)

    # 导入数据，训练
    source_path = os.path.join(SAVE_DIR_BASE, 'train_data', ds_start + '_' + ds_num)
    print(source_path)
    data = eval('Ev{feature}Loader_hbase(source_path, logid_path=logid_path, sampling_type=args.sampling_type, '
                'test_size=args.test_size, max_num=args.max_num)'.format(feature=args.feature))
    data.run()
    model = BLSTMModel(max_len=args.max_len, train_data=data.train_data, test_data=data.test_data,
                       epoch=args.epoch, batch_size=args.batch_size, dropout_size=args.dropout_size,
                       regular=args.regular, embedding_size=args.embedding_size, dense_size=args.dense_size,
                       lstm_size=args.lstm_size, feature_type=args.feature, max_logid=max_logid, save_path=PATH_MODEL_SAVE)
    model.run()
