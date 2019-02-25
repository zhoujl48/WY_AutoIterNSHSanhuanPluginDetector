#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2018, The NSH Anti-Plugin Project
# 
"""
module doc here

Authors: Zhou Jialiang
Email: zjl_Sempre@163.com
Date: 2019/1/30
"""
import os
import impala.dbapi
import json

hive_ip = '***'
hive_port = 36003


def connect_hive(hive_ip, hive_port, sql):
    # 验证
    os.system('kinit -kt ***.keytab ***@***')
    # 执行
    conn = impala.dbapi.connect(host=hive_ip, port=hive_port, auth_mechanism='GSSAPI',
                                kerberos_service_name='hive', database="default")
    cursor = conn.cursor()
    cursor.execute("show databases")

    cursor.execute(sql)
    results = cursor.fetchall()


    return results

# 若本地不存在，则从HIVE拉取，并保存至本地
def get_ids(sql, ids_path):
    results = connect_hive(hive_ip=hive_ip, hive_port=hive_port, sql=sql)
    ids = [str(item[0]) for item in results]
    with open(ids_path, 'w') as f:
        json.dump(ids, f, indent=4, sort_keys=True)
    return ids


if __name__ == '__main__':


    sql = """
    SELECT DISTINCT grade.role_id
    FROM luoge_nsh_ods.ods_nsh_upgrade grade 
    WHERE grade.role_level = 41
    AND grade.ds >= '2019-01-20'
	AND grade.ds <= '2019-01-30'
    """

    ids = get_ids(sql, 'tmp_hive')