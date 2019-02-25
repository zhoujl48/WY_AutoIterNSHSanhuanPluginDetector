#!/usr/bin/python
# -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The NSH Anti-Plugin Project
################################################################################
"""
NSH三环挂自动迭代项目 -- 配置模块

Usage: 供调用
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019/02/13
"""

################################################################################

# 项目根目录
PROJECT_DIR = '/home/zhoujialiang/online_sanhuan'

# 数据目录
SAVE_DIR_BASE = '/srv/nsh-antiplugin-dataset/sanhuangua/'

################################################################################

# Hive Config
HIVE_HOST_IP = '***.***.***.***'
HIVE_HOST_PORT = 10000
HIVE_HOST_USER = '***'
HIVE_HOST_PASSWORD = '***'
HIVE_TARGET_DB = '***'
HIVE_DB_DIR = '/user/hive/warehouse/%s.db/' % HIVE_TARGET_DB
HDFS_HOST_IP = '***.***.***.***'
HDFS_HOST_PORT = 14000

# Query，获取三环挂模型训练正样本id中的封杀样本
QUERY_SQL_PUNISH = """
SELECT sanhuan.role_id, sanhuan.ds
FROM (
    SELECT role_id, ds
    FROM (
        SELECT role_id, split(role_scene_id, '-')[0] AS scene_id, ds
        FROM luoge_nsh_ods.ods_nsh_entermap
        WHERE ds >= '{ds_start}' AND ds <= '{ds_end}'
    ) enter
    WHERE scene_id IN ('16100060', '16100061', '16101518', '16101519')
) sanhuan
JOIN (
    SELECT role_id
    FROM luoge_nsh_ods.ods_nsh_gmpunishplayer punish
    WHERE punish.punishment_type = 4
        AND punish.ds >= '{ds_punish_start}'
        AND punish.ds <= '{ds_punish_end}'
) pos_punish
ON sanhuan.role_id = pos_punish.role_id
GROUP BY sanhuan.role_id, sanhuan.ds
"""

# Query，获取三环挂模型训练正样本id中的封停样本
QUERY_SQL_BAN = """
SELECT sanhuan.role_id, sanhuan.ds
FROM (
    SELECT role_id, ds
    FROM (
        SELECT role_id, split(role_scene_id, '-')[0] AS scene_id, ds
        FROM luoge_nsh_ods.ods_nsh_entermap
        WHERE ds >= '{ds_start}' AND ds <= '{ds_end}'
    ) enter
    WHERE scene_id IN ('16100060', '16100061', '16101518', '16101519')
) sanhuan
JOIN (
    SELECT portrait.role_id
    FROM luoge_nsh_mid.mid_role_portrait_all_d portrait
    JOIN luoge_nsh_dwd.dwd_nsh_account_ban_add_d ban
    ON portrait.role_account_name = ban.role_account_name
    WHERE portrait.ds = '{ds_end}'
        AND ban.opr_type = 'banLogin'
        AND ban.ds >= '{ds_ban_start}'
        AND ban.ds <= '{ds_ban_end}'
) pos_ban
ON sanhuan.role_id = pos_ban.role_id
GROUP BY sanhuan.role_id, sanhuan.ds
"""

# Query，获取三环挂模型训练负样本id
QUERY_SQL_NEG = """
SELECT sanhuan.role_id, sanhuan.ds
FROM (
    SELECT role_id, ds
    FROM (
        SELECT role_id, split(role_scene_id, '-')[0] AS scene_id, ds
        FROM luoge_nsh_ods.ods_nsh_entermap
        WHERE ds >= '{ds_start}' 
            AND ds <= '{ds_end}'
    ) enter
    WHERE scene_id IN ('16100060', '16100061', '16101518', '16101519')
) sanhuan
LEFT JOIN (
    SELECT role_id
    FROM luoge_nsh_ods.ods_nsh_gmpunishplayer punish
    WHERE punish.punishment_type = 4
        AND punish.ds >= '{ds_ban_start}'
        AND punish.ds <= '{ds_ban_end}'
) pos_punish
ON sanhuan.role_id = pos_punish.role_id
LEFT JOIN (
    SELECT portrait.role_id
    FROM luoge_nsh_mid.mid_role_portrait_all_d portrait
    JOIN luoge_nsh_dwd.dwd_nsh_account_ban_add_d ban
    ON portrait.role_account_name = ban.role_account_name
    WHERE portrait.ds = '{ds_ban_end}'
        AND ban.opr_type = 'banLogin'
        AND ban.ds >= '{ds_ban_start}'
        AND ban.ds <= '{ds_ban_end}'
) pos_ban
ON sanhuan.role_id = pos_ban.role_id
WHERE pos_punish.role_id IS NULL
    AND pos_ban.role_id IS NULL
ORDER BY rand()
LIMIT 200000
"""

# Query，获取三环挂模型预测单天全量样本id
QUERY_SQL_DATE = """
SELECT role_id, ds
    FROM (
        SELECT role_id, split(role_scene_id, '-')[0] AS scene_id, ds
        FROM luoge_nsh_ods.ods_nsh_entermap
        WHERE ds = '{ds_start}'
    ) enter
    WHERE scene_id IN ('16100060', '16100061', '16101518', '16101519')
  GROUP BY role_id, ds
"""

# Query字典
QUERY_DICT = {
    'ban': QUERY_SQL_BAN,
    'punish': QUERY_SQL_PUNISH,
    'total': QUERY_SQL_DATE,
    'neg': QUERY_SQL_NEG
}

################################################################################

# hbase序列拉取线程数
THREAD_NUM = 100
# hbase链接
HBASE_URL = 'http://***.***.***.22***8:8080/roleseq/time?start_time={st}&end_time={ed}&game=nsh&role_info={ids}'

################################################################################

# MySQL Config
MySQL_HOST_IP = '***.***.***.***'
MySQL_HOST_PORT = 3306
MySQL_HOST_USER = '***'
MySQL_HOST_PASSWORD = '***'
MySQL_TARGET_DB = '***'

# time format
TIME_FORMAT = '%Y%m%d %H:%M:%S'

# 预测模块预测频率
MINUTE_DELTA = 15
