#!/usr/bin/env bash
# Usage_1: bash train.sh

# 工作目录
WORK_DIR=/home/zhoujialiang/online_sanhuan

# 定义参数
last_wednesday=`date -d "wednesday -2 weeks" +%Y%m%d`       # 过去最近的周三
ds_start_old=`date -d "$last_wednesday -35 days" +%Y%m%d`   # 旧模型的数据样本开始日期
ds_start=`date -d "$last_wednesday -28 days" +%Y%m%d`       # 新模型的数据样本开始日期
ds_start_new=`date -d "$last_wednesday -7 days" +%Y%m%d`    # 新模型需更新的数据样本开始日期


# 复制重复数据至新目录
echo /usr/bin/python3 $WORK_DIR/update_data.py pos --old_dir ${ds_start_old}_28 --ds_start $ds_start --ds_num 28 &&
/usr/bin/python3 $WORK_DIR/update_data.py pos --old_dir ${ds_start_old}_28 --ds_start $ds_start --ds_num 28 &&
echo /usr/bin/python3 $WORK_DIR/update_data.py neg --old_dir ${ds_start_old}_28 --ds_start $ds_start --ds_num 28 &&
/usr/bin/python3 $WORK_DIR/update_data.py neg --old_dir ${ds_start_old}_28 --ds_start $ds_start --ds_num 28 &&

# 获取新id
echo /usr/bin/python3 $WORK_DIR/get_ids.py pos --ds_start $ds_start_new --ds_num 7 &&
/usr/bin/python3 $WORK_DIR/get_ids.py pos --ds_start $ds_start_new --ds_num 7 &&
echo /usr/bin/python3 $WORK_DIR/get_ids.py neg --ds_start $ds_start_new --ds_num 7 &&
/usr/bin/python3 $WORK_DIR/get_ids.py neg --ds_start $ds_start_new --ds_num 7 &&

# 获取新序列
echo /usr/bin/python3 $WORK_DIR/dataloader.py --ds_start $ds_start_new --ds_num 7 --mode neg &&
/usr/bin/python3 $WORK_DIR/dataloader.py --ds_start $ds_start_new --ds_num 7 --mode neg &&
echo /usr/bin/python3 $WORK_DIR/dataloader.py --ds_start $ds_start_new --ds_num 7 --mode pos &&
/usr/bin/python3 $WORK_DIR/dataloader.py --ds_start $ds_start_new --ds_num 7 --mode pos &&

# 滑窗迭代模型
echo /usr/bin/python3 $WORK_DIR/BLSTMModel.py --ds_start $ds_start --ds_num 28 &&
/usr/bin/python3 $WORK_DIR/BLSTMModel.py --ds_start $ds_start --ds_num 28
