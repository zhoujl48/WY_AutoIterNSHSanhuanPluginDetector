#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2018, The NSH Anti-Plugin Project
# 
"""
mysql tool, 使用NSH线上数据库

Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019/1/31
"""
import pymysql

"""
mysql
"""
HOST = '***.***.***.***'
USER = '***'
PASSWORD = '***'
DATABASE = '***'
PORT = 3306

class MySQL(object):
    """
    mysql class
    usage:
        mysql = MySQL()
        result = mysql.query('select * from mid_role_portrait_all_d limit 10;')
    return None if failed
    """

    def __init__(self, host=HOST, user=USER, password=PASSWORD, database=DATABASE, port=PORT):
        '''
         mincached: minimum free connection number
         maxcached: maximum free connection number
        '''

        self.conn = pymysql.connect(host=host, port=port, user=user, passwd=password, db=database)

    def query(self, sql):
        '''
        query data by sql
        :param sql:
        :return: the data stored in mysql
        '''
        ret = None
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
                ret = cursor.fetchall()
        finally:
            cursor.close()
        return ret

    def execute(self, sql):
        '''
        execute command by sql
        :param sql:
        :return:
        '''
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
            self.conn.commit()
        finally:
            cursor.close()

    def batch_insert(self, table, columns, data):
        '''
        批量插入数据库
        :param table:
        :param columns: ['id', 'name']
        :param data: [[1,'Alice'], [2, 'bob']]
        :return:
        '''

        cols = ','.join(columns)
        placeholders = ','.join(['%s'] * len(columns))
        sql = 'insert into {table}({cols}) values ({holders})'.format(table=table, cols=cols, holders=placeholders)

        try:
            with self.conn.cursor() as cursor:
                cursor.executemany(sql, data)
            self.conn.commit()
        finally:
            cursor.close()
