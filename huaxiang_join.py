#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The NSH Anti-Plugin Project
################################################################################
"""
NSH三环挂自动迭代项目 -- 关联画像模块
步骤：
    1. 预测全量结果上传MySQL的tmp表
    2. MySQL与Hive跨数据库JOIN，获取阈值筛选后的用户画像
    3. 调整字段，画像上传FTP
    4. 调整字段，画像上传MySQL

Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019/1/30
"""
import os
import csv
import codecs
import requests
import logging
import argparse
from datetime import datetime, timedelta
from dateutil.parser import parse
import impala.dbapi
import log
from mysql import MySQL
from config import PROJECT_DIR, SAVE_DIR_BASE


hive_ip = '***'
hive_port = 36003

SHUYUAN_URL = "http://***.***.***.***:49349/runsql"

# 画像列
PROFILE_COLS = ['suspect_score', 'role_account_name', 'role_name', 'role_gender', 'role_class', 'nie_lian_time', 'server',
                'create_date', 'create_time', 'phone_num', 'del_status', 'del_date', 'role_level', 'role_total_online_time',
                'role_base_score', 'role_talisman_score', 'role_skill_score', 'role_practice_score', 'role_equip_score',
                'role_total_score', 'role_wuxue_score', 'not_binded_charged_yuanbao', 'hstb', 'bdyb', 'yin_liang',
                'yin_piao', 'ck_yin_liang', 'bhgx', 'sjtxjf', 'kjjf', 'liang_shi', 'jj', 'ttsw', 'qldb', 'dsn', 'idn',
                'mac', 'mid', 'smb', 'cpu_name', 'device_id', 'device_name', 'display_memory_localized_size',
                'dx_version', 'os_name', 'os_version', 'physical_memory_size', 'vendor_id', 'role_cor', 'role_agi',
                'role_int', 'role_sta', 'role_str', 'finished_task_acm_cnt', 'watch_movie_acm_pct_avg',
                'npc_interact_acm_tms', 'latest_log_time', 'punish_cnt', 'first_punish_time', 'latest_punish_time',
                'latest_undo_punish_time', '2w_d_avg_lw_time', '2w_d_avg_sjtx_time', '2w_d_avg_zjx_time',
                '2w_d_avg_wl_time', '2w_d_avg_wyc_time', '2w_d_avg_wxl_time', '2w_d_avg_cy_time', '2w_d_avg_qm_time',
                '2w_d_avg_qm_yx_time', '2w_d_avg_jd_time', '2w_d_avg_szqx_time', '2w_d_avg_shl_time',
                '2w_d_avg_zy_time', '2w_d_avg_zl_time', '2w_d_avg_mh_time', '45level_f_bl_task_acm_num',
                '60level_f_bl_task_acm_num', 'f_bl_task_acm_num', 'latest_ml_chapter', '45level_latest_ml_chapter',
                '60level_latest_ml_chapter', '2w_d_avg_f_ml_task_cnt', '2w_d_avg_f_bl_task_cnt',
                '2w_d_avg_f_bt_task_cnt', '2w_d_avg_f_jj_task_cnt', '2w_d_avg_f_bh_task_cnt', '2w_d_avg_f_task_cnt',
                '2w_get_bhgx_amt', '2w_d_avg_get_bhgx_amt', '2w_get_wulin_amt', '2w_d_avg_get_wulin_amt',
                '2w_get_xiayi_amt', '2w_d_avg_get_xiayi_amt', '2w_get_liangshi_amt', '2w_d_avg_get_liangshi_amt',
                '2w_get_recharge_yuanbao_amt', '2w_d_avg_get_recharge_yuanbao_amt', '2w_use_bhgx_amt',
                '2w_d_avg_use_bhgx_amt', '2w_use_wulin_amt', '2w_d_avg_use_wulin_amt', '2w_use_xiayi_amt',
                '2w_d_avg_use_xiayi_amt', '2w_use_liangshi_amt', '2w_d_avg_use_liangshi_amt',
                '2w_use_recharge_yuanbao_amt', '2w_d_avg_use_recharge_yuanbao_amt', '2w_use_bdyb_amt',
                '2w_d_avg_use_bdyb_amt', '2w_use_free_yuanbao_amt', '2w_d_avg_use_free_yuanbao_amt',
                '2w_use_huoyue_amt', '2w_d_avg_use_huoyue_amt', '2w_use_tili_amt', '2w_d_avg_use_tili_amt',
                '2w_use_huoli_amt', '2w_d_avg_use_huoli_amt', '2w_get_skillexp_amt', '2w_d_avg_get_skillexp_amt',
                '2w_get_exp_amt', '2w_d_avg_get_exp_amt', 'acm_get_free_yuanbao_amt', 'acm_get_recharge_yuanbao_amt',
                'acm_get_charge_yuanbao_amt', 'acm_get_yl_amt', 'acm_get_yp_amt', 'acm_get_yl_yp_amt',
                'acm_use_free_yuanbao_amt', 'acm_use_recharge_yuanbao_amt', 'acm_use_charge_yuanbao_amt',
                'acm_use_yl_amt', 'acm_use_yp_amt', 'acm_use_yl_yp_amt', 'max_equip_score', 'max_skill_score',
                'max_practice_score', 'max_talisman_score', 'max_total_score', 'max_base_score',
                'learned_production_ids', 'learned_production_num', '2w_active_days', '2w_d_avg_team_chat_cnt',
                '2w_d_avg_guild_chat_cnt', '2w_d_avg_create_team_cnt', '2w_d_avg_join_team_cnt',
                '2w_d_avg_leave_team_cnt', '2w_d_avg_kickout_team_cnt', '2w_d_avg_match_team_cnt',
                '2w_d_avg_match_waiting_time', '1st_apprentice_level', '1st_apprentice_date', 'late_apprentice_level',
                'late_apprentice_date', 'acm_apprentice_cnt', 'chushi_status', 'chushi_level', 'chushi_date',
                '2w_bfr_baishi_d_avg_onl_tm', '1w_bfr_chushi_d_avg_onl_tm', '1w_aft_chushi_d_avg_onl_tm',
                '2w_get_yl_amt', '2w_d_avg_get_yl_amt', '2w_get_yp_amt', '2w_d_avg_get_yp_amt', '2w_trade_get_yl_amt',
                '2w_d_avg_trade_get_yl_amt', '2w_yabiao_get_yl_amt', '2w_d_avg_yabiao_get_yl_amt',
                '2w_yabiao_get_yp_amt', '2w_d_avg_yabiao_get_yp_amt', '2w_sellgold_get_yl_amt',
                '2w_d_avg_sellgold_get_yl_amt', '2w_cangbaoge_get_yl_amt', '2w_d_avg_cangbaoge_get_yl_amt',
                '2w_bid_get_yl_amt', '2w_d_avg_bid_get_yl_amt', '2w_task_get_yl_amt', '2w_d_avg_task_get_yl_amt',
                '2w_task_get_yp_amt', '2w_d_avg_task_get_yp_amt', '2w_use_yl_amt', '2w_d_avg_use_yl_amt',
                '2w_use_yp_amt', '2w_d_avg_use_yp_amt', '2w_trade_use_yl_amt', '2w_d_avg_trade_use_yl_amt',
                '2w_shop_buy_use_yl_amt', '2w_d_avg_shop_buy_use_yl_amt', '2w_merge_equip_use_yl_amt',
                '2w_d_avg_merge_equip_use_yl_amt', '2w_merge_equip_use_yp_amt', '2w_d_avg_merge_equip_use_yp_amt',
                'acm_use_wulin_amt', '2w_shop_buy_equip_use_yl_amt', '2w_d_avg_shop_buy_equip_use_yl_amt',
                '2w_buyitem_buy_equip_use_yl_amt', '2w_d_avg_buyitem_buy_equip_use_yl_amt', 'acm_use_jieyusha_amt',
                '2w_d_avg_onl_time', 'acm_upgrade_skill_use_yl_amt', 'acm_upgrade_skill_use_yp_amt',
                'acm_merge_equip_use_yl_amt', 'acm_merge_equip_use_yp_amt', 'acm_upgrade_skill_use_yl_rto',
                'acm_upgrade_skill_use_yp_rto', 'acm_merge_equip_use_yl_rto', 'acm_merge_equip_use_yp_rto',
                'acm_up_9_level_skill_amt', 'skill_avg_level', 'bfr_45_level_total_onl_tm', '2w_merge_equip_cnt',
                '2w_d_avg_merge_equip_cnt', '2w_purple_merge_equip_cnt', '2w_d_avg_purple_merge_equip_cnt',
                '2w_xilian_equip_cnt', '2w_d_avg_xilian_equip_cnt', '2w_get_red_equip_amt',
                '2w_d_avg_get_red_equip_amt', '2w_get_purple_equip_amt', '2w_d_avg_get_purple_equip_amt',
                'acm_get_hero_card_amt', 'acm_guild_create_cnt', 'acm_guild_join_cnt', 'acm_guild_leave_cnt',
                'acm_guild_kickout_cnt', '2w_task_accept_cnt', '2w_d_avg_task_accept_cnt', '2w_task_finish_cnt',
                '2w_d_avg_task_finish_cnt', '2w_task_giveup_cnt', '2w_d_avg_task_giveup_cnt', 'ml_24_chapter_f_level',
                'bl_yaji_f_level', 'first_ban_time', 'first_unban_time', 'late_ban_time', 'late_unban_time', 'ban_status']

# MySQL上传画像字段
PROFILE_COLS_MYSQL = ['role_id', 'suspect_score', 'method', 'start_time', 'end_time'] + PROFILE_COLS[1:]

# 预测结果全天全量文件
PREDICTED_FILE_DIR = os.path.join(SAVE_DIR_BASE, 'classification')

# 关联画像疑似度阈值
THRESHOLD =0.8

# FTP画像HEAD
HEAD_FILE = os.path.join(PROJECT_DIR, 'huaxiang_head')

# FTP目录
FTP_BASE_DIR= '/home/ftp/sanhuan_sl_blstm_huaxiang_{ds}__.csv'


def connect_hive(hive_ip, hive_port):
    # 验证
    os.system('kinit -kt ***.keytab ***@***')
    # 执行
    conn = impala.dbapi.connect(host=hive_ip, port=hive_port, auth_mechanism='GSSAPI',
                                kerberos_service_name='hive', database="default")
    cursor = conn.cursor()
    cursor.execute("show databases")
    print(cursor.fetchall())


def sanhuangua_join_profile(data, ds):
    mysql = MySQL()

    # delete
    delete_sql = 'delete from nsh_sanhuangua_tmp'
    mysql.execute(delete_sql)
    # insert
    mysql.batch_insert('nsh_sanhuangua_tmp', ['role_id', 'suspect_score'], data)

    PROFILE_SQL = """
    select a.suspect_score, b.*
    from anti_plugin.nsh_sanhuangua_tmp a 
    join luoge_nsh_mid.mid_role_portrait_all_d b on a.role_id = b.role_id
    where b.ds = '{ds}'
    """

    sql = PROFILE_SQL.format(ds=ds)
    logging.info(sql)

    params = {
        'sql': sql,
        'needReturn': 'true'
    }

    # 关联画像请求，返回结果
    result = requests.post(SHUYUAN_URL, timeout=6000, json=params)

    # json转成字典
    id_profile_dict = {}
    for line in result.json()['data']:
        profile_dict = {}
        for k, v in line.items():
            k = k.split('.')[-1]
            v = '0' if v is None else v
            profile_dict[k] = v

        profiles = [profile_dict.get(col, '') for col in PROFILE_COLS]
        role_id = profile_dict['role_id']
        id_profile_dict[role_id] = profiles

    # 结果字典
    profile_data = list()
    for role_id, profiles in id_profile_dict.items():
        row = [role_id] + profiles
        profile_data.append(row)

    return profile_data


def load_predicted_data(file_path, threshold):

    # 读取预测结果
    data = list()
    with open(file_path, 'r') as f:
        for line in f:
            role_id, score = line.strip().split(',')
            if float(score) > threshold:
                data.append([int(role_id), float(score)])

    return data


def save_ftp(profile_data, file_path, head_file):

    with open(head_file, 'r', encoding='utf-8') as f:
        head = f.read().strip().split('\t')

    data = [head] + profile_data

    with open(file_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def upload_to_mysql(data, cols):

    mysql = MySQL()

    mysql.batch_insert('nsh_sanhuangua', cols, data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Connect precition to profile'
                                     'Usage: python huaxiang_join.py 2019-02-17')
    parser.add_argument('ds')
    args = parser.parse_args()
    ds = args.ds

    # Init logging
    log.init_log(os.path.join(PROJECT_DIR, 'logs', 'huaxiang_join'))

    # 加载预测数据
    predicted_file_path = os.path.join(PREDICTED_FILE_DIR, 'sanhuan_sl_blstm_{ds}.csv'.format(ds=parse(ds).strftime('%Y_%m_%d')))
    logging.info('Loading predicted data, file: {}'.format(predicted_file_path))
    data = load_predicted_data(file_path=predicted_file_path, threshold=THRESHOLD)
    logging.info('Length of data above threshold {} is {}'.format(THRESHOLD, len(data)))

    # 关联画像
    logging.info('Start joining profile...')
    profile_data = sanhuangua_join_profile(data, parse(ds).strftime('%Y-%m-%d'))
    logging.info('Done joining profile...')
    logging.info(profile_data[:3])


    # 保存FTP
    ftp_path = FTP_BASE_DIR.format(ds=parse(ds).strftime('%Y_%m_%d'))
    logging.info('Start saving to FTP {}'.format(ftp_path))
    save_ftp(profile_data, ftp_path, HEAD_FILE)
    logging.info('Done saving to FTP {}'.format(ftp_path))


    # 上传画像结果至MySQL
    start_time = parse(ds).strftime('%Y-%m-%d %H:%M:%S')
    end_time = (parse(ds) + timedelta(hours=23, minutes=59, seconds=59)).strftime('%Y-%m-%d %H:%M:%S')
    method = 'nsh_sanhuangua_model'
    add_col = lambda row: row[:2] + [method, start_time, end_time] + row[2:]
    profile_data_mysql = list(map(add_col, profile_data))
    logging.info('Start uploading profile to MySQL...')
    upload_to_mysql(profile_data_mysql, PROFILE_COLS_MYSQL)
    logging.info('Done uploading')




